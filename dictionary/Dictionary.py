import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F

## =====================================
## standard DNN model
## single neuron and activations
## tensor

class ReluDerivative(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.n = n
        self.m = m
        self.fact = np.math.factorial(n) / np.math.factorial(n-m)
        
    # d^m/dx^m [relu(x)^n], 
    def forward(self, out):
        out = F.relu(out).pow(self.n-self.m) * self.fact
        return out

class Bspline(nn.Module):
    def __init__(self, wei, deg):
        super().__init__()
        self.w = wei
        self.p = deg
    
    def forward(self, x):
        out = 0
        for i in range(len(self.w)):
            out = out + self.w[i]*F.relu(i - x).pow(self.p)
        out = out * (self.p+1)
        return out  

class GAnet(nn.Module):
    def __init__(self, in_dim, width, out_dim, sigma):
        super(GAnet, self).__init__()
        self.layer1 = nn.Linear(in_dim,width)
        self.layer2 = nn.Linear(width,out_dim)
        self.F = sigma
        
    def forward(self, out):
        out = self.layer1(out)
        out = self.F(out)
        out = self.layer2(out)
        return out
    
class GAneuron(nn.Module):
    def __init__(self, in_dim, sigma):
        super(GAneuron, self).__init__()
        self.layer = nn.Linear(in_dim, 1)
        self.F = sigma
        
    def forward(self, out):
        out = self.layer(out)
        out = self.F(out)
        return out