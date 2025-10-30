# RepAlignLoss

RepAlignLoss is a PyTorch toolbox for **representation alignment losses**. It lets you compare a student model's outputs to a ground-truth target in the feature space of a frozen, pretrained teacher network. The project now ships with utilities to capture teacher activations, visualize per-layer contributions, and experiment with a weight-norm-aware optimizer tailored to distillation-style training.

## Overview
- Extract intermediate activations from a frozen teacher (DINOv2, VGG, CLIP-style encoders, etc.) while keeping gradients flowing only through the student.
- Align representations by normalizing each layer's activations and measuring their mean-squared error.
- Optionally weight deep layers more heavily, or inspect layer-wise losses to refine which features you care about.
- Use the standalone example script to bootstrap experiments or as a reference for integrating into your own training loop.

## What's Included
- `RepAlignLoss.py`: main loss implementation with a demonstration entrypoint (set `_SELECTED_MODEL` to pick a teacher and `_PLOT_LAYERS` to enable plotting).
- `Optimizer.py`: `WNGradW`, a weight-norm-scaled optimizer with gradient centralization, momentum, and trust-ratio style scaling.
- `Plotter.py`: utilities to chart per-layer losses and activation sizes so you can see which layers dominate the objective.
- `setup.py`: minimal packaging script that reads this README as the long description.

## Installation

You only need PyTorch to instantiate the loss, but different teachers add extra requirements:

```bash
pip install torch torchvision
# Optional extras when needed
pip install transformers matplotlib
```

When working locally inside this repository you can install it in editable mode:

```bash
pip install -e .
```

> **Tip:** Some teacher choices (e.g. Perception Encoder models) expect additional repositories to be present. Check the comments around `_SELECTED_MODEL` inside `RepAlignLoss.py` before running the demo.

## Quick Start

```python
import torch
import torchvision.transforms as T
from transformers import Dinov2Model
from RepAlignLoss import RepAlignLoss

# 1. Load your frozen teacher and matching preprocessing.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = Dinov2Model.from_pretrained("facebook/webssl-dino300m-full2b-224").to(device).eval()
normalize = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# 2. Instantiate the loss. By default it hooks nn.Conv2d and nn.Linear layers.
rep_loss = RepAlignLoss(teacher, normalize, device=device, use_weight=True)

student = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1).to(device)
optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

# 3. During training, create activation banks for the student output and the target.
targets = torch.randn(4, 3, 140, 140, device=device)          # your ground truth
inputs = torch.randn(4, 3, 140, 140, device=device)           # data for the student

student_out = student(inputs)
student_acts = rep_loss.MakeData(student_out)                 # gradients enabled

with torch.no_grad():
    target_acts = rep_loss.MakeData(targets)                  # cache teacher features

loss = rep_loss(student_acts, target_acts)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Customizing the Hooks
- Set `use_weight=True` when constructing `RepAlignLoss` to linearly increase layer weights with depth.
- Edit the `sel_module` list in `RepAlignLoss.__init__` if you want to hook other layer types (e.g. attention blocks).
- Call `MakeData` inside a `with torch.no_grad()` block when you do not need gradients (e.g. for the target tensor) to save memory.

## Inspect Layer Contributions

After collecting activations you can visualize which layers dominate the objective:

```python
from Plotter import plot_layer_loss_and_numel
import matplotlib.pyplot as plt

fig, (ax_loss, ax_numel), data = plot_layer_loss_and_numel(
    rep_loss, student_acts, target_acts, log_numel=True
)
plt.show()
```

The helper returns both the Matplotlib figure/axes and a plain Python dictionary containing raw per-layer loss and element counts, making it easy to log the statistics elsewhere.

## Weight-Norm Gradient Optimizer (`Optimizer.WNGradW`)

`WNGradW` is a normalization-aware optimizer designed with representation alignment in mind:
- Gradient centralization keeps channels centred before measuring their norms.
- Optional momentum on the direction vector adds stability without altering gradients in-place.
- Trust-ratio style scaling uses the running norm of each parameter block; flip `inverse_trust` to switch between inverse scaling (recommended for distillation) and LARS-like behaviour.
- Works with any PyTorch module; drop it in place of your usual optimizer:

```python
from Optimizer import WNGradW
optimizer = WNGradW(student.parameters(), lr=1e-3, weight_decay=1e-4, beta_m=0.9)
```

## Demo Script

Run the bundled example to see the whole pipeline in action:

```bash
python RepAlignLoss.py
```

Tweak `_SELECTED_MODEL` inside the script to switch between:
- `0` - DINOv2 via `torch.hub`
- `1` - Hugging Face DINOv2 (default)
- `2` - Meta Perception Encoder (requires the external `perception_models` repo)
- `4` - VGG19 from `torchvision`

Set `_PLOT_LAYERS = True` to pop up the visualization window (requires Matplotlib).

## Practical Notes
- Teacher parameters are frozen automatically and run in evaluation mode; gradients only flow through the student.
- Activation banks can be large. Monitor GPU memory, especially with deep teachers and big batches.
- Ensure the `normalize` transform matches the teacher's expected input distribution (size, mean, and standard deviation).
- Sparse gradients are not supported by `WNGradW`.

## License

MIT License - see `LICENSE`.

## Author

`Gabriel Poetsch` - griskai.yt@gmail.com