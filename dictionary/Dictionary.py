import torch
import torch.nn as nn 
import numpy as np


dtype = torch.float64
torch.set_default_dtype(dtype)

## =====================================
## general dictionaries
## standard DNN model
## single neuron and activations
## tensor


class ShallowNN(nn.Module):
    def __init__(self, in_dim, width, out_dim, sigma):
        super(ShallowNN, self).__init__()
        self.layer1 = nn.Linear(in_dim,width)
        self.layer2 = nn.Linear(width,out_dim)
        self.F = sigma    
        
    def forward(self, out):
        out = self.layer1(out)
        out = self.F(out)
        out = self.layer2(out)
        return out
    

class BasisFunction(nn.Module):
    def __init__(self, in_dim, activation_function):
        super(BasisFunction, self).__init__()
        self.layer = nn.Linear(in_dim, 1)
        self.activation_function = activation_function
        
    def update_neuron(self, parameters):
        weight = parameters.weight.requires_grad_(True)
        bias = parameters.bias.requires_grad_(True)
        param_dict = {'layer.weight': weight, 'layer.bias': bias}
        self.load_state_dict(param_dict, strict=True)

    def forward(self, p):
        val = self.layer(p)
        val = self.activation_function(val)
        return val