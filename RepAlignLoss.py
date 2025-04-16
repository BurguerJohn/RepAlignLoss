import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math

class RepAlignLoss(torch.nn.Module):
    def __init__(self, sel_model, normalize, device=None, use_weight=False, verbose=True):
        super().__init__()
        self.model = sel_model
        self.normalize = normalize
        self.use_weight = use_weight

        self.activations = []
        def getActivation():
            # the hook signature
            def hook(model, input, output):
                self.activations.append(output)
            return hook

        for param in self.model.parameters():
            param.requires_grad_(False)

        count = 0
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

    def HandleTensor(self, x, head=32):
        pad = (head - (x.shape[-1] % head)) % head
        x = F.pad(x, (0, pad))
        x = x.reshape(x.size(0), x.size(1),  x.size(2) // head, head)
        return x

    def CalculateLoss(self, x, y, heads):
        x = self.HandleTensor(x, heads)
        y = self.HandleTensor(y, heads)
        x = nn.functional.normalize(x, p=2.0, dim=-1)
        y = nn.functional.normalize(y, p=2.0, dim=-1)
        loss = nn.functional.mse_loss(x, y.detach(), reduction="none")
        return loss.sum(), loss.numel()
        

    def HandleData(self, x, y):
        loss = 0
        elements = 0

        x = x.view(x.size(0), x.size(1), -1)
        y = y.view(y.size(0), y.size(1), -1)

        l, e = self.CalculateLoss(x, y, 2)

        loss += l 
        elements += e
        return loss, elements

        
    def forward(self, X_VAL, Y_VAL):
        loss = 0
        elements = 0
        
        #Improve, no need to calculate each loop.
        weights = [math.exp(i * 0.1) for i in range(len(X_VAL))]
        total_weight = sum(weights)

        for i in range(len(X_VAL)):
            l, s  =  self.HandleData(X_VAL[i], Y_VAL[i]) 
            
            #Optional weight
            if self.use_weight:
                w = weights[i] / total_weight
                l = l * w
                
            loss += l 
            elements += s
            
        return loss / elements


if __name__ == "__main__":
    device = torch.device("cpu")

    #Select a teacher and tensor transformations before feeding to the teacher.
    teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device).eval()
    norm = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    loss_func = RepAlignLoss(teacher, norm, device, use_weight=False, verbose=True).to(device)

    #Since the teacher is dino V2, we need to feed it tensors [B,3,H,W] tensor with 14X14 patches.
    model_output = torch.randn(1, 3, 140, 140).to(device)
    gt = torch.randn(1, 3, 140, 140).to(device)

    #Feed both tensors to DinoV2 and store each layer data.
    output_data = loss_func.MakeData(model_output)
    with torch.no_grad():
        gt_data = loss_func.MakeData(gt)

    #Calculate loss
    loss = loss_func(output_data, gt_data)
    print(loss.item())
