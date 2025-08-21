import math
import torch
from torch.optim.optimizer import Optimizer

class TTAdamW(Optimizer):
    r"""
    TTAdamW = AdamW + Two-Timescale sensitivity gate (+ optional LAMB-style trust ratio, + optional Gradient Centralization)

    Per-parameter state:
        m_t = β1 m_{t-1} + (1-β1) g_t
        v_t = β2 v_{t-1} + (1-β2) g_t^2              (short-term 2nd moment, Adam)
        u_t = β3 u_{t-1} + (1-β3) g_t^2              (long-term "Fisher-ish" sensitivity)

    Bias-corrected:
        m̂_t = m_t / (1-β1^t),   v̂_t = v_t / (1-β2^t),   û_t = u_t / (1-β3^t)

    Precondition + gate:
        pgrad = m̂_t / (sqrt(v̂_t) + eps)
        τ_t   = 1 / (1 + γ * sqrt(û_t))               # temper updates where long-term sensitivity is large
        upd   = τ_t * pgrad

    Optional layer-wise trust ratio (LAMB-style):
        trust = clip( ||w|| / (||upd|| + 1e-12), [trust_min, trust_max] )
        upd   = trust * upd

    Decoupled weight decay (AdamW):
        w ← w - lr * (upd + wd * w)

    Args:
        params (iterable): model parameters
        lr (float): learning rate
        betas (Tuple[float,float]): Adam betas for (m, v)
        beta3 (float): long-term EMA factor for u_t (e.g. 0.9999)
        eps (float)
        weight_decay (float): decoupled weight decay factor
        gamma (float): strength of the sensitivity gate τ (higher = stronger tempering)
        layerwise_adaptation (bool): enable LAMB-style trust ratio
        trust_clip (Tuple[float,float]): clamp for trust ratio if enabled
        gradient_centralization (bool): subtract grad mean over non-batch dims for tensors with dim>1
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta3=0.9999, eps=1e-8,
                 weight_decay=0.01, gamma=0.5, layerwise_adaptation=True,
                 trust_clip=(0.01, 10.0), gradient_centralization=True):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f"Invalid beta3: {beta3}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if trust_clip[0] <= 0.0 or trust_clip[1] <= 0.0 or trust_clip[0] > trust_clip[1]:
            raise ValueError(f"Invalid trust_clip: {trust_clip}")

        defaults = dict(lr=lr, betas=betas, beta3=beta3, eps=eps, weight_decay=weight_decay,
                        gamma=gamma, layerwise_adaptation=layerwise_adaptation,
                        trust_clip=trust_clip, gradient_centralization=gradient_centralization)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr     = group['lr']
            beta1, beta2 = group['betas']
            beta3  = group['beta3']
            eps    = group['eps']
            wd     = group['weight_decay']
            gamma  = group['gamma']
            use_lw = group['layerwise_adaptation']
            tmin, tmax = group['trust_clip']
            use_gc = group['gradient_centralization']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # Optional Gradient Centralization (for weight tensors with dim > 1)
                if use_gc and grad.dim() > 1:
                    dims = tuple(range(1, grad.dim()))
                    grad = grad - grad.mean(dim=dims, keepdim=True)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # m
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # v
                    state['long_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # u

                m = state['exp_avg']
                v = state['exp_avg_sq']
                u = state['long_avg_sq']

                state['step'] += 1
                t = state['step']

                # Adam-style EMA updates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Long-term sensitivity (very slow EMA of squared grads)
                u.mul_(beta3).addcmul_(grad, grad, value=1 - beta3)

                # Bias correction
                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                bias_c3 = 1 - beta3 ** t

                m_hat = m / bias_c1
                v_hat = v / bias_c2
                u_hat = u / bias_c3

                # Precondition + sensitivity gate
                denom = v_hat.sqrt().add_(eps)
                pgrad = m_hat / denom

                # τ = 1 / (1 + γ * sqrt(û))
                tau = 1.0 / (1.0 + gamma * (u_hat.sqrt()))
                upd = pgrad * tau

                # Optional layer-wise trust ratio (LAMB-ish)
                if use_lw:
                    w_norm = p.norm(p=2)
                    u_norm = upd.norm(p=2)
                    if w_norm > 0 and u_norm > 0:
                        trust = torch.clamp(w_norm / (u_norm + 1e-12), min=tmin, max=tmax)
                        upd = upd * trust

                # Apply decoupled weight decay
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

                # Parameter update
                p.add_(upd, alpha=-lr)

        return loss