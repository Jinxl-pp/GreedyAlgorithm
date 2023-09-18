import torch
import torch.nn as nn 


class ShallowNN(nn.Module):
    def __init__(self, 
                 sigma,
                 in_dim,
                 width,
                 out_dim=1,
                 ):
        super(ShallowNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, width, bias = True)
        self.layer2 = nn.Linear(width, out_dim, bias = False)
        self.F = sigma    
        self.num_neurons = width
        
    def update_neurons(self, parameters):
        layer2_weight = parameters[0].requires_grad_(True)
        layer1_weight = parameters[1].requires_grad_(True)
        layer1_bias = parameters[2].requires_grad_(True)
        param_dict = {'layer1.weight': layer1_weight, 
                        'layer1.bias': layer1_bias,
                        'layer2.weight': layer2_weight,
                        }
        self.load_state_dict(param_dict, strict=True)
        
    def forward(self, out):
        out = self.layer1(out)
        out = self.F(out)
        out = self.layer2(out)
        return out