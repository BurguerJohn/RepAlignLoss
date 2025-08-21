# RepAlignLoss (Representation Alignment Loss)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`RepAlignLoss` is a PyTorch loss function designed for guiding the training of a 'student' model by aligning the feature representations *of its output* with those of a *ground truth* image, as perceived by a pre-trained 'teacher' model. It encourages the student to generate outputs that are perceptually similar to the target by mimicking the internal activations of powerful foundation models (like DINOv2, VGG19, ResNet, etc.) when they process the student's output versus the target.

This approach falls under the umbrella of **feature-level knowledge distillation** or **perceptual loss**, specifically applied by comparing the teacher's interpretation of the student's output and the ground truth.

## Key Features

*   **Teacher-Student Framework:** Leverages a frozen, pre-trained model (teacher) to provide perceptual guidance signals.
*   **Layer-wise Activation Matching:** Uses PyTorch forward hooks to extract intermediate activations from specified layers (default: `nn.Conv2d`, `nn.Linear`) of the teacher model.
*   **Output-Target Alignment:** Calculates loss by comparing the teacher's activations produced when processing the *student's output* versus the teacher's activations produced when processing the *ground truth* image.
*   **Normalized Feature Comparison:** Uses L2 normalization followed by MSE loss for robust similarity measurement between activation vectors.
*   **Flexible:** Allows easy swapping of teacher models and configuration of which layer types to use for comparison. Requires providing an appropriate normalization transform matching the teacher's pre-training.

## How it Works

1.  **Teacher Model:** A pre-trained model (e.g., `dinov2_vits14_reg`) is loaded, set to evaluation mode (`eval()`), and its parameters are frozen (`requires_grad_(False)`).
2.  **Activation Hooking:** Forward hooks are registered on selected layers (by default `nn.Conv2d` and `nn.Linear`) within the teacher model. These hooks capture the output activations of these layers during a forward pass.
3.  **Activation Extraction (`MakeData` method):**
    *   Takes an input tensor (either the student model's output or the ground truth image).
    *   Applies the necessary normalization (`self.normalize`) expected by the teacher model.
    *   Performs a forward pass through the *teacher* model (`self.model(Y)`).
    *   The hooks capture the activations from the selected layers into a list (`self.activations`).
    *   This method is called *twice* per training step: once for the student's output and once for the ground truth target.
4.  **Loss Calculation (`forward` method):**
    *   Takes two lists of activation tensors: `X_VAL` (activations from teacher processing the *student's output*) and `Y_VAL` (activations from teacher processing the *ground truth*).
    *   Iterates through the corresponding activation tensors layer by layer from `X_VAL` and `Y_VAL`.
    *   For each pair of activation tensors (`x`, `y`):
        *   Flattens spatial/sequence dimensions: `(B, C, H, W)` -> `(B, C, H*W)`
        *   Applies L2 normalization along the last dimension (feature dimension)
        *   Calculates Mean Squared Error (MSE) loss between the normalized vectors
        *   Accumulates the loss and element count
    *   Optional weighting based on layer depth can be applied (`use_weight=True`), giving higher weights to deeper layers.
    *   The final loss is the total loss divided by the total number of elements across all layers.

### Feature Comparison Method

The current implementation uses a straightforward approach for comparing activations between student output and ground truth:

1.  **Flatten:** Spatial dimensions (if any) are flattened: `(Batch, Channels, Height, Width)` -> `(Batch, Channels, Height*Width)`
2.  **Normalize:** L2 normalization is applied along the last dimension (feature dimension), ensuring unit norm vectors
3.  **Compare:** MSE loss is calculated between the normalized activation vectors from student output and ground truth
4.  **Aggregate:** Losses from all layers are summed and divided by the total number of elements for averaging

This method ensures that the student model learns to produce outputs whose internal representations (as perceived by the teacher) are similar to those of the ground truth, focusing on the direction/pattern of activations rather than their magnitudes.

## Dependencies

*   PyTorch (`torch`)
*   TorchVision (`torchvision`) - for transforms and models
*   Access to `torch.hub` for loading models like DINOv2 (`facebookresearch/dinov2`)
*   Optional: `transformers` for HuggingFace models
*   Optional: Custom optimizers like `TTAdamW` (included in the repository)

You can typically install the core dependencies with:
```bash
pip install torch torchvision transformers
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/BurguerJohn/RepAlignLoss
    cd RepAlignLoss
    ```

## Usage

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from RepAlignLoss import RepAlignLoss

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load your Student Model
student_model = nn.Sequential(
    nn.Conv2d(3, 64, 3, 1, 1),
    nn.ReLU(),
    nn.Conv2d(64, 3, 3, 1, 1)
)
student_model.to(device)

# 2. Load the Teacher Model and Define Normalization
teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device).eval()

# Normalization specific to the teacher model
normalize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# 3. Instantiate the RepAlignLoss
loss_func = RepAlignLoss(
    sel_model=teacher,
    normalize=normalize,
    device=device,
    use_weight=False,  # Optional: Use layer weighting
    verbose=True       # Set to False during training loops
).to(device)

# 4. Setup Optimizer
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

# --- Training Loop Example ---
num_epochs = 10
dataloader = ...  # Your data loader providing (input_data, ground_truth_image) pairs

for epoch in range(num_epochs):
    student_model.train()
    for batch_idx, (input_data, ground_truth_image) in enumerate(dataloader):
        input_data = input_data.to(device)
        ground_truth_image = ground_truth_image.to(device)

        # Forward pass through student model
        student_output = student_model(input_data)

        # Get activations from teacher model
        # Process ground truth first (no gradients needed)
        with torch.no_grad():
            gt_activations = loss_func.MakeData(ground_truth_image)

        # Process student output (gradients needed for backprop)
        student_activations = loss_func.MakeData(student_output)

        # Calculate RepAlignLoss
        rep_align_loss = loss_func(student_activations, gt_activations)

        # Backpropagation
        optimizer.zero_grad()
        rep_align_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {rep_align_loss.item():.4f}")
```

## Supported Teacher Models

The repository includes examples for several pre-trained teacher models:

### DINOv2 (Default)
```python
teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
normalize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

### VGG19
```python
teacher = torchvision.models.vgg19(pretrained=True)
normalize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

### HuggingFace DINOv2
```python
from transformers import Dinov2Model
teacher = Dinov2Model.from_pretrained('facebook/webssl-dino300m-full2b-224')
normalize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

## Configuration Parameters

### `RepAlignLoss.__init__`

*   `sel_model` (torch.nn.Module): The pre-trained, frozen teacher model.
*   `normalize` (callable): A torchvision transform that preprocesses input tensors to match the teacher model's requirements.
*   `device` (torch.device, optional): The device where calculations should happen. Defaults to `None`.
*   `use_weight` (bool, optional): If `True`, applies linear weighting to deeper layers. Defaults to `False`.
*   `verbose` (bool, optional): If `True`, prints hooked layers during initialization. Defaults to `True`.

## TTAdamW Optimizer

The repository includes `TTAdamW`, an enhanced optimizer combining:
- AdamW with decoupled weight decay
- Two-timescale sensitivity gating
- Optional LAMB-style trust ratio
- Optional gradient centralization

```python
from TTAdamW import TTAdamW
optimizer = TTAdamW(student_model.parameters(), lr=1e-3)
```

## Important Notes

*   **Memory Usage:** Storing activations from multiple layers can be memory-intensive with large batch sizes.
*   **Computational Cost:** Two forward passes through the teacher model (student output + ground truth) add overhead.
*   **Normalization Matching:** The `normalize` transform must exactly match the teacher model's preprocessing requirements.
*   **Layer Selection:** By default, hooks target `nn.Conv2d` and `nn.Linear` layers. Modify `sel_module` in the code to change this.
*   **Teacher Model Frozen:** All teacher model parameters have `requires_grad=False` to prevent updates during training.

## Example Run

The `RepAlignLoss.py` file includes a complete example that can be run directly:

```bash
python RepAlignLoss.py
```

This example:
1. Creates a simple student model (single Conv2d layer)
2. Uses VGG19 as the teacher model (controlled by `_SELECTED_MODEL = 4`)
3. Generates random input and ground truth tensors
4. Computes the RepAlignLoss and demonstrates the training setup

## Author

**Gabriel Poetsch**  
Email: griskai.yt@gmail.com