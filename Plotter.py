import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Dict, Any
import torch

def plot_layer_loss_and_numel(
    loss_mod,                           # RepAlignLoss instance
    X_VAL: Sequence,                    # list/tuple of tensors per layer (student/model)
    Y_VAL: Sequence,                    # list/tuple of tensors per layer (teacher/target)
    *,
    log_numel: bool = False,            # set True if numel dwarfs loss scale
    start_index: int = 0                # set to 1 if you prefer 1-based layer indices
) -> Tuple[Any, Tuple[Any, Any], Dict[str, Any]]:
    """
    Returns:
      - fig: the matplotlib Figure
      - (ax_loss, ax_numel): the two axes (left = loss, right = numel)
      - data dict with 'layers', 'loss', 'numel'
    """
    assert len(X_VAL) == len(Y_VAL), "X_VAL and Y_VAL must have the same number of layers"

    per_layer_loss = []
    per_layer_numel = []
    per_layer_mean = []

    # Calculate per-layer metrics using the same routine your forward() loop uses.
    with torch.no_grad():
        for x_i, y_i in zip(X_VAL, Y_VAL):
            l_i, n_i = loss_mod.CalculateLoss(x_i, y_i)
            per_layer_loss.append(float(l_i))
            per_layer_numel.append(int(n_i))
            per_layer_mean.append(float(l_i) / n_i)

    layers = list(range(start_index, start_index + len(per_layer_loss)))

    # Plot
    fig, ax_loss = plt.subplots()
    ln1, = ax_loss.plot(layers, per_layer_mean, marker="o", label="Loss (mean)")
    ax_loss.set_xlabel("Layer index")
    ax_loss.set_ylabel("Loss (sum)")

    ax_numel = ax_loss.twinx()
    ln2, = ax_numel.plot(layers, per_layer_numel, marker="s", linestyle="--", label="Numel", color="tab:green")
    ax_numel.set_ylabel("Numel")
    if log_numel:
        ax_numel.set_yscale("log")

    ax_loss.grid(True, linestyle=":", which="both", axis="both")
    lines = [ln1, ln2]
    labels = [ln.get_label() for ln in lines]
    ax_loss.legend(lines, labels, loc="best")
    fig.tight_layout()

    return fig, (ax_loss, ax_numel), {"layers": layers, "loss": per_layer_loss, "numel": per_layer_numel}
