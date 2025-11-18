import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import warnings

class RepAlignLoss(torch.nn.Module):
    def __init__(self, sel_model, normalize, device=None, use_weight=False, verbose=True):
        super().__init__()
        self.model = sel_model
        self.normalize = normalize
        self.use_weight = use_weight
        self.device = device
        self.generator = torch.Generator(device=device)

        self.activations = []
        def getActivation():
            # the hook signature
            def hook(model, input, output):
                self.activations.append(output)
            return hook

        for param in self.model.parameters():
            param.requires_grad_(False)

        count = 0

        if verbose:
            print("Using Weights: ", self.use_weight)

        #Can also use more layers or all of them

        #sel_module = [nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU]
        sel_module = [nn.Linear, nn.Conv2d]
        use_all_layers = False

        def traverse_modules(module):
            nonlocal count
            for name, sub_module in module.named_children():
                traverse_modules(sub_module)

                if (use_all_layers and len(list(sub_module.named_children())) == 0) or isinstance(sub_module, tuple(sel_module)):
                  if verbose:
                    print("~~~~", count,  sub_module)
                  count += 1
                  sub_module.register_forward_hook(getActivation())

        traverse_modules(self.model)
        if verbose:
            print(f"Total Layers: {count}")


    def MakeData(self, Y):
        Y = self.normalize(Y)
        self.activations = []
        self.model(Y)
        Y_VAL = self.activations
        return Y_VAL

    
    
    def CalculateLoss(self, x, y):
        sel_dim = tuple(range(1, x.ndim))

        with torch.no_grad():
            x_p = (x.pow(2).sum(dim=sel_dim, keepdim=True).sqrt() + 1e-8)
            y_p = (y.pow(2).sum(dim=sel_dim, keepdim=True).sqrt() + 1e-8)

        x = x / x_p
        y = y / y_p

        loss = nn.functional.mse_loss(x,  y, reduction="none")

        loss = loss.mean(dim=sel_dim)
        
        return loss.sum(), loss.numel()
    
        
    def forward(self, X_VAL, Y_VAL):
        loss = 0
        elements = 0
        
        if self.use_weight:
            n = len(X_VAL)
            weights = [(i+ 1) / (n+1) for i in range(n)]

        for i in range(len(X_VAL)):
            l, s  =  self.CalculateLoss(X_VAL[i], Y_VAL[i]) 
            
            #Optional weight
            if self.use_weight:
                l = l * weights[i]
                
            loss += l 
            elements += s
            
        return loss / elements




_ALL_MODELS = ["dinov2_vits14_reg", "webssl-dino300m-full2b-224", "PE-Core-B16-224", "VGG19"]
_SELECTED_MODEL = 1
_PLOT_LAYERS = True

if __name__ == "__main__":
    device = torch.device("cpu")

    if _SELECTED_MODEL == 0:
        #https://github.com/facebookresearch/dinov2
        teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device).eval()
        norm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    if _SELECTED_MODEL == 1:
        #https://huggingface.co/facebook/webssl-dino300m-full2b-224
        from transformers import Dinov2Model
        teacher  = Dinov2Model.from_pretrained('facebook/webssl-dino300m-full2b-224').to(device).eval()
        norm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    if _SELECTED_MODEL == 2:
        #https://github.com/facebookresearch/perception_models
        import os, sys
        perception_models_path = os.path.abspath('./perception_models')
        sys.path.append(perception_models_path)
        os.chdir(perception_models_path)
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        teacher = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True).to(device).eval()

        norm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean=.5, std=.5),
        ])

    if _SELECTED_MODEL == 4:
        #https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
        teacher = torchvision.models.vgg19(pretrained=True).to(device).eval()
        norm = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


    loss_func = RepAlignLoss(teacher, norm, device, use_weight=True, verbose=True).to(device)

    #Since the teacher is dino V2, we need to feed it tensors [B,3,H,W] tensor with 14X14 patches.
    model_input = torch.randn(1, 3, 140, 140, requires_grad=True).to(device)
    training_model = torch.nn.Sequential(
        nn.Conv2d(3, 3, 3, 1, 1)
    )
    
    gt = torch.randn(1, 3, 140, 140).to(device)

    try:
        #Custom Optimizer that may work better for teacher/student learning
        from RPLOptimizer import WNGradW
        optimizer = WNGradW(training_model.parameters(), 1e-2)
        print(optimizer)
    except ImportError:
        pass
    
    model_output = training_model(model_input)
    #Feed both tensors to DinoV2 and store each layer data.
    output_data = loss_func.MakeData(model_output)
    with torch.no_grad():
        gt_data = loss_func.MakeData(gt)

    if _PLOT_LAYERS:
        from Plotter import plot_layer_loss_and_numel
        import matplotlib.pyplot as plt
        fig, (ax_loss, ax_numel), data = plot_layer_loss_and_numel(loss_func, output_data, gt_data, log_numel=True)
        plt.show()


    loss = loss_func(output_data, gt_data)
    print("Training Loss:", loss.item())
        
