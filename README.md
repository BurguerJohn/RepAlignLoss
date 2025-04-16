# RepAlignLoss  (Representation Alignment Loss)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`RepAlignLoss` is a PyTorch loss function designed for guiding the training of a 'student' model by aligning the feature representations *of its output* with those of a *ground truth* image, as perceived by a pre-trained 'teacher' model. It encourages the student to generate outputs that are perceptually similar to the target by mimicking the internal activations of powerful foundation models (like DINOv2, ResNet, EfficientNet, etc.) when they process the student's output versus the target.

This approach falls under the umbrella of **feature-level knowledge distillation** or **perceptual loss**, specifically applied by comparing the teacher's interpretation of the student's output and the ground truth.

## Key Features

*   **Teacher-Student Framework:** Leverages a frozen, pre-trained model (teacher) to provide perceptual guidance signals.
*   **Layer-wise Activation Matching:** Uses PyTorch forward hooks to extract intermediate activations from specified layers (default: `nn.Conv2d`, `nn.Linear`) of the teacher model.
*   **Output-Target Alignment:** Calculates loss by comparing the teacher's activations produced when processing the *student's output* versus the teacher's activations produced when processing the *ground truth* image.
*   **Localized Similarity Metric:** Employs a unique method to compare activations based on *local groups* of features within flattened activation maps, promoting finer-grained similarity.
*   **Flexible:** Allows easy swapping of teacher models and configuration of which layer types to use for comparison. Requires providing an appropriate normalization transform matching the teacher's pre-training.

## How it Works

1.  **Teacher Model:** A pre-trained model (e.g., `dinov2_vitb14_reg`) is loaded, set to evaluation mode (`eval()`), and its parameters are frozen (`requires_grad_(False)`).
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
        *   Flattens spatial/sequence dimensions: `(B, C, H, W)` -> `(B, C, H*W)` or `(B, Seq, Dim)` -> `(B, Seq, Dim)`.
        *   **Localized Similarity Calculation (`HandleData` -> `CalculateLoss`):**
            *   Pads the last dimension (flattened features) of both `x` and `y` so its size is divisible by a group size (hardcoded to `2` in `HandleData`).
            *   Reshapes the tensors to group features: `(B, C, Features)` -> `(B, C, Features // 2, 2)`.
            *   Applies L2 normalization *independently to each small group* (dimension of size 2).
            *   Calculates Mean Squared Error (MSE) loss between the corresponding normalized groups (`y` is detached).
        *   Sums the MSE loss across all elements for the layer pair.
    *   Optional weighting based on layer depth can be applied (`use_weight=True`), giving higher weights to deeper layers.
    *   The final loss is the aggregated, averaged similarity score across all selected layers and all elements.

### Localized Similarity

A key differentiator in `RepAlignLoss` is how similarity is calculated between corresponding layer activations (`x` from student output, `y` from ground truth):

1.  **Flatten:** Spatial dimensions (if any) are flattened, resulting in tensors like `(Batch, Channels, Features)`.
2.  **Pad & Group:** The last dimension (`Features`) is padded to be divisible by 2. It's then reshaped into groups of 2: `(Batch, Channels, Features_padded // 2, 2)`.
3.  **Normalize per Group:** L2 normalization is applied along the last dimension (the one of size 2). This normalizes each pair of features independently.
4.  **Compare Groups:** MSE loss is calculated between the normalized groups of `x` and `y`.
5.  **Effect:** Instead of calculating similarity (like cosine similarity via MSE on normalized vectors) across the *entire* feature vector for a given channel/location, this method enforces similarity between very localized *pairs* of features within the flattened activation map. This encourages the student model to generate outputs where finer-grained feature relationships, as perceived by the teacher, match those of the ground truth.

## Dependencies

*   PyTorch (`torch`)
*   TorchVision (`torchvision`) - for transforms, potentially models
*   Access to `torch.hub` for loading models like DINOv2 (`facebookresearch/dinov2`)

You can typically install the core dependencies with:
```bash
pip install torch torchvision
```

## Installation

1.  Clone the repository:
    ```bash
    # Replace with your actual repo URL if different
    git clone https://github.com/BurguerJohn/RepAlignLoss
    cd RepAlignLoss
    ```

## Usage

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from RepAlignLoss import RepAlignLoss # Assuming RepAlignLoss.py is in the same directory or installed

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load your Student Model
student_model = ... # Your model definition here
student_model.to(device)

# 2. Load the Teacher Model and Define Normalization
# Ensure the teacher model and normalization match!
teacher_model_name = 'dinov2_vits14_reg' # Example
teacher = torch.hub.load('facebookresearch/dinov2', teacher_model_name).to(device).eval()

# Normalization specific to the teacher model (DINOv2 example)
normalize = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), # Example size for ViT
    transforms.CenterCrop(224),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# 3. Instantiate the RepAlignLoss
# Note: `verbose=False` is usually preferred during training loops
loss_func = RepAlignLoss(
    sel_model=teacher,
    normalize=normalize,
    device=device,
    use_weight=False, # Optional: Use layer weighting
    verbose=False
).to(device)

# 4. Setup Optimizer
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

# --- Training Loop Example ---
num_epochs = ...
dataloader = ... # Your data loader providing (input_data, ground_truth_image) pairs

for epoch in range(num_epochs):
    student_model.train() # Set student model to training mode
    for batch_idx, (input_data, ground_truth_image) in enumerate(dataloader):
        input_data = input_data.to(device)
        ground_truth_image = ground_truth_image.to(device)

        # --- Forward Pass ---
        student_output = student_model(input_data)

        # --- Calculate RepAlignLoss ---
        # IMPORTANT: Process BOTH student output and ground truth through the teacher
        # using MakeData to get the respective activation lists.
        # No gradient needed for teacher forward passes when getting activations.
        with torch.no_grad():
            gt_activations = loss_func.MakeData(ground_truth_image)

        # Need gradients for student_output's activations as they depend on student params
        # loss_func.MakeData clears activations, so call it for student_output *after* gt
        student_output_activations = loss_func.MakeData(student_output)

        # Calculate the actual loss
        rep_align_loss = loss_func(student_output_activations, gt_activations)

        # --- (Optional) Combine with other losses ---
        # e.g., reconstruction_loss = nn.functional.mse_loss(student_output, ground_truth_image)
        # total_loss = rep_align_loss + lambda_rec * reconstruction_loss
        total_loss = rep_align_loss # Using only RepAlignLoss here

        # --- Backward Pass and Optimization ---
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}")

    # --- (Optional) Validation Step ---
    # ...

# --- Save Model ---
# torch.save(student_model.state_dict(), 'student_model.pth')

```

## Configuration Parameters (`RepAlignLoss.__init__`)

*   `sel_model` (torch.nn.Module): The pre-trained, frozen teacher model.
*   `normalize` (callable): A torchvision transform (or similar callable) that preprocesses input tensors to match the teacher model's requirements (e.g., resizing, normalization).
*   `device` (torch.device, optional): The device ('cpu' or 'cuda') where the teacher model resides and calculations should happen. Defaults to `None`.
*   `use_weight` (bool, optional): If `True`, applies an exponential weight to the loss contributions from different layers, giving more importance to deeper layers. Defaults to `False`.
*   `verbose` (bool, optional): If `True`, prints the layers being hooked during initialization and the total count. Defaults to `True`.

## Important Notes

*   **Memory Consumption:** Storing activations from multiple layers of potentially large teacher models can be memory-intensive, especially with large batch sizes.
*   **Computational Cost:** Performing forward passes through the teacher model twice (once for student output, once for ground truth) adds computational overhead to each training step.
*   **Normalization:** Ensure the `normalize` transform provided to `RepAlignLoss` is the *exact* preprocessing used when the `sel_model` (teacher) was originally trained. Incorrect normalization will lead to meaningless activation comparisons.
*   **Layer Selection:** By default, hooks are placed on `nn.Conv2d` and `nn.Linear` layers. You can modify the `sel_module` list inside `RepAlignLoss.py` to target different layer types (e.g., `nn.ReLU`, `nn.GELU` were commented out).
*   **Group Size:** The localized similarity currently uses a hardcoded group size of `2` within the `HandleData` method. This could be parameterized in the future if needed.
*   **Teacher Model Input:** The teacher model processes the *final output* of the student model, not its intermediate features directly. The loss guides the student to produce outputs that *look* similar to the ground truth *according to the teacher's internal representations*.