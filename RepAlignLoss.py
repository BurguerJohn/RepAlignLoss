import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

class RepAlignLoss(torch.nn.Module):
    def __init__(self, sel_model, normalize, device=None, randomize_pairs=True, use_weight=False, verbose=True):
        super().__init__()
        self.model = sel_model
        self.normalize = normalize
        self.use_weight = use_weight
        self.randomize_pairs = randomize_pairs
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

    def HandleTensor(self, x, head=2):
        pad = (head - (x.shape[-1] % head)) % head
        x = F.pad(x, (0, pad))
        x = x.reshape(x.size(0), -1, head)
        return x

    def l2_normalize_groups(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
                      
        # Calculate sum of squares for elements in x and y per group
        # x_sq_sum and y_sq_sum will have shape (num_groups, 1)
        x_sq_sum = x.pow(2).sum(dim=-1, keepdim=True)
        y_sq_sum = y.pow(2).sum(dim=-1, keepdim=True)
    
        # Total sum of squares for the 4 elements in each group
        group_sum_sq = x_sq_sum + y_sq_sum
        group_norm = torch.sqrt(group_sum_sq)
    
        x_normalized = x / (group_norm + eps)
        y_normalized = y / (group_norm + eps)
    
        return x_normalized, y_normalized
    
    def CalculateLoss(self, x, y, heads):
        x = self.HandleTensor(x, heads)
        y = self.HandleTensor(y, heads)

        #x = nn.functional.softmax(x, dim=-1)
        #y = nn.functional.softmax(y, dim=-1)
        #loss = nn.functional.mse_loss(x, y.detach(), reduction="none")
        #loss = (1 - torch.nn.functional.cosine_similarity(x, y.detach(), dim=-1)).pow(2)
        
        #x = F.normalize(x, dim=-1)
        #y = F.normalize(y, dim=-1)

        x, y = self.l2_normalize_groups(x, y)

        loss = nn.functional.mse_loss(x, y.detach(), reduction="none")

        return loss.sum(), loss.numel()
        

    def HandleData(self, x, y):
        loss = 0
        elements = 0

        if x.ndim > 3:
            x = x.view(x.size(0), x.size(1), -1)
            y = y.view(y.size(0), y.size(1), -1)
            
        if self.randomize_pairs:
            perm = torch.randperm(x.size(-1), device=x.device, generator=self.generator)
            x = x[...,  perm]
            y = y[...,  perm]


        l, e = self.CalculateLoss(x, y, 2)

        loss += l 
        elements += e
        return loss, elements

        
    def forward(self, X_VAL, Y_VAL):
        loss = 0
        elements = 0
        
        #Improve, no need to calculate each loop.
        #Other option is use exp weights
        #weights = [math.exp(i * 0.1) for i in range(len(X_VAL))]
        weights = [i + 1 for i in range(len(X_VAL))]
        total_weight = sum(weights)

        for i in range(len(X_VAL)):
            #Eval mode, make generator deterministic.
            if not X_VAL[i].requires_grad or not torch.is_grad_enabled():
                self.generator.manual_seed(42)
                

            l, s  =  self.HandleData(X_VAL[i], Y_VAL[i]) 
            
            #Optional weight
            if self.use_weight:
                w = weights[i] / total_weight
                l = l * w
                
            loss += l 
            elements += s
            
        return loss / elements



_ALL_MODELS = ["dinov2_vits14_reg", "webssl-dino300m-full2b-224", "PE-Core-B16-224", "VGG19"]
_SELECTED_MODEL = 0

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


    loss_func = RepAlignLoss(teacher, norm, device, randomize_pairs=True, use_weight=False, verbose=True).to(device)

    #Since the teacher is dino V2, we need to feed it tensors [B,3,H,W] tensor with 14X14 patches.
    model_output = torch.randn(1, 3, 140, 140, requires_grad=True).to(device)
    gt = torch.randn(1, 3, 140, 140).to(device)

    #Feed both tensors to DinoV2 and store each layer data.
    output_data = loss_func.MakeData(model_output)
    with torch.no_grad():
        gt_data = loss_func.MakeData(gt)

    #Calculate loss
    #Training with random pairs make the loss change with the same data.
    for _ in range(3):
        loss = loss_func(output_data, gt_data)
        print("Training with random pairs:", loss.item())
    
    
    #Eval with random pairs. Loss does not change.
    for _ in range(3):
        with torch.no_grad():
            loss = loss_func(output_data, gt_data)
        print("Eval with random pairs:", loss.item())
