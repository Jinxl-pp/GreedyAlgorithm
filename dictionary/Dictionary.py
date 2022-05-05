import sys
sys.path.append('../')

import torch
import torch.nn as nn 
import numpy as np
from optimization import *

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
    def __init__(self, base_dim, activation_function):
        super(BasisFunction, self).__init__()
        self.layer = nn.Linear(base_dim, 1)
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


class Dictionary(BasisFunction):
    
    def __init__(self, 
                base_dim, 
                activation_function,
                optimizer,
                param_dim, 
                param_mesh,
                param_domain,):
        super().__init__(base_dim, activation_function)

        self.optimizer = optimizer
        self.param_dim = param_dim
        self.param_mesh = param_mesh
        self.param_domain =  param_domain

    def _gather_flat_param():
        pass

    def _basis_train():
        pass

    def select_initial_basis():
        pass

    def find_optimal_basis():
        pass


