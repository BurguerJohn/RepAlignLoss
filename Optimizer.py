import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable, Tuple, Optional, Union

class WNGradW(Optimizer):
    """
    Weight-Norm-Scaled normalized gradient descent with decoupled weight decay.

    Core idea per parameter tensor p:
      - Optionally center grads along non-output dims (gradient centralization).
      - Compute a momentum-smoothed direction m_t.
      - Normalize the direction per 'vector' (rows/channels) to have unit L2 norm.
      - Maintain an EMA of weight^2 (per element) in fp32; reduce over the same dims
        to get a per-vector weight scale.
      - Scale the step by either denom (LARS-like) or 1/denom (inverse-trust, your default).
      - Apply AdamW-style decoupled weight decay.

    This preserves your behavior by default (inverse_trust=True), but adds:
      - fp32 state for stability
      - momentum on gradients
      - bias correction on the weight EMA
      - safer numerics and no in-place grad edits

    Notes:
      - For linear [out, in] or conv [out, in, ...], the reduction dims are (1..ndim-1),
        i.e., we treat each output channel/row as a vector.
      - For 1D params (bias, LayerNorm gamma/beta), reduction dims is (0,).
      - For 0D params (rare), treat as a scalar.

    Args:
      params: iterable of parameters
      lr: learning rate
      weight_decay: decoupled weight decay coefficient
      eps: numerical epsilon
      beta_m: momentum coefficient for gradient direction (0 disables momentum if set to 0)
      beta_w: EMA coefficient for weight^2
      center_grad: subtract mean over non-output dims for tensors with ndim > 1
      normalize_grad: L2-normalize the direction per vector
      inverse_trust: if True, scale updates by 1/||w|| (your current behavior);
                     if False, scale by ||w|| (LARS-style relative step)
      bias_correction: apply bias correction to the weight-EMA scale early in training
      max_trust: optional clamp on the scaling factor to bound step magnitudes
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        eps: float = 1e-8,
        beta_m: float = 0.0,
        beta_w: float = 0.0,
        center_grad: bool = True,
        normalize_grad: bool = True,
        inverse_trust: bool = False,
        bias_correction: bool = False,
        max_trust: Optional[float] = None,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not (0.0 <= beta_m < 1.0):
            raise ValueError(f"Invalid beta_m: {beta_m}")
        if not (0.0 <= beta_w < 1.0):
            raise ValueError(f"Invalid beta_w: {beta_w}")
        if max_trust is not None and max_trust <= 0:
            raise ValueError(f"Invalid max_trust: {max_trust}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            beta_m=beta_m,
            beta_w=beta_w,
            center_grad=center_grad,
            normalize_grad=normalize_grad,
            inverse_trust=inverse_trust,
            bias_correction=bias_correction,
            max_trust=max_trust,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta_m = group["beta_m"]
            beta_w = group["beta_w"]
            center_grad = group["center_grad"]
            normalize_grad = group["normalize_grad"]
            inverse_trust = group["inverse_trust"]
            bias_correction = group["bias_correction"]
            max_trust = group["max_trust"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("WNGradW does not support sparse gradients.")

                g = p.grad
                state = self.state[p]

                # Choose reduction dims: treat dim 0 as output; reduce over the rest
                if p.ndim == 0:
                    reduce_dims: Optional[Tuple[int, ...]] = None
                elif p.ndim == 1:
                    reduce_dims = (0,)
                else:
                    reduce_dims = tuple(range(1, p.ndim))

                if len(state) == 0:
                    state["step"] = 0
                    # Momentum on gradients in fp32
                    if beta_m > 0.0:
                        state["m"] = torch.zeros_like(p, dtype=torch.float32, device=p.device)
                    # EMA of weight^2 in fp32
                    if beta_w > 0.0:
                        state["ema_w2"] = torch.zeros_like(p, dtype=torch.float32, device=p.device)

                if beta_m > 0.0:
                    m = state["m"]
                if beta_w > 0.0:
                    ema_w2 = state["ema_w2"]
                state["step"] += 1
                t = state["step"]

                # Gradient centralization (avoid in-place on the shared grad buffer)
                if center_grad and p.ndim > 1:
                    g_eff = g - g.mean(dim=reduce_dims, keepdim=True)
                else:
                    g_eff = g

                # Momentum on the direction (fp32)
                if beta_m > 0.0:
                    m.mul_(beta_m).add_(g_eff.to(torch.float32), alpha=1.0 - beta_m)
                    dir_vec = m
                else:
                    dir_vec = g_eff.to(torch.float32)

                # Normalize per vector (robustly)
                if normalize_grad:
                    if reduce_dims is None:
                        g_norm = dir_vec.abs().clamp_min(eps)
                        u = dir_vec / g_norm
                    else:
                        g_norm = dir_vec.norm(p=2, dim=reduce_dims, keepdim=True).clamp_min(eps)
                        u = dir_vec / g_norm
                else:
                    u = dir_vec

                # Update EMA of weight^2 using current weights (fp32)
                w_fp32 = p.detach().to(torch.float32) ** 2
                if beta_w > 0.0:
                    ema_w2.mul_(beta_w).add_(w_fp32, value=1.0 - beta_w)
                    w_fp32 = ema_w2

                # Per-vector weight scale: sqrt(sum(ema_w2)) ~= ||w||_2 smoothed
                if reduce_dims is None:
                    denom = w_fp32.sqrt().clamp_min(eps)
                else:
                    denom = w_fp32.sum(dim=reduce_dims, keepdim=True).sqrt_().clamp_min(eps)

                # Bias correction for early steps (optional)
                if bias_correction:
                    corr = math.sqrt(1.0 - beta_w ** t)
                    denom = denom / max(corr, eps)

                # Trust factor: either inverse (your original behavior) or direct (LARS-like)
                trust = denom.reciprocal() if inverse_trust else denom

                if max_trust is not None:
                    trust = trust.clamp(max=max_trust)

                # Decoupled weight decay
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

                # Apply update in parameter dtype
                p.add_((u.to(p.dtype) * trust.to(p.dtype)), alpha=-lr)

        return loss