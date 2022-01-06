import torch
import numpy as np
pi = 3.1415926535897932384626

## =====================================
## 1 dimensional exact solution function

class DataCos1d:
    
    def target(out):
        out = torch.cos(pi*out)
        return out
    
    def dtarget(out):
        out = -pi*torch.sin(pi*out)
        return out
    
    def right_hand_side(out):
        out = pi**2 * torch.cos(pi*out) + torch.cos(pi*out)
        return out
    
    
## =====================================
## 2 dimensional exact solution function 

class DataCos2d:

    def target(pts):
        x = pts[:,0:1]
        y = pts[:,1:2]
        out = torch.cos(2*pi*x) * torch.cos(2*pi*y)
        return out
    
    def dx_target(pts):
        x = pts[:,0:1]
        y = pts[:,1:2]
        out = -2*pi * torch.cos(2*pi*y) * torch.sin(2*pi*x)
        return out
    
    def dy_target(pts):
        x = pts[:,0:1]
        y = pts[:,1:2]
        out = -2*pi * torch.cos(2*pi*x) * torch.sin(2*pi*y)
        return out
    
    def right_hand_side(pts):
        x = pts[:,0:1]
        y = pts[:,1:2]
        out = torch.cos(2*pi*x) * torch.cos(2*pi*y) + \
              8*pi**2 * torch.cos(2*pi*x) * torch.cos(2*pi*y)
        return out
    
   
## =====================================
## 3 dimensional exact solution function     
    
class DataCos3d:
    
    def target(pts):
        x = pts[:,0:1]
        y = pts[:,1:2]
        z = pts[:,2:3]
        out = torch.cos(2*pi*x) * torch.cos(2*pi*y) * torch.cos(2*pi*z)
        return out
    
    def dx_target(pts):
        x = pts[:,0:1]
        y = pts[:,1:2]
        z = pts[:,2:3]
        out = -2*pi * torch.cos(2*pi*y) * torch.cos(2*pi*z) * torch.sin(2*pi*x)
        return out
    
    def dy_target(pts):
        x = pts[:,0:1]
        y = pts[:,1:2]
        z = pts[:,2:3]
        out = -2*pi * torch.cos(2*pi*x) * torch.cos(2*pi*z) * torch.sin(2*pi*y)
        return out
    
    def dz_target(pts):
        x = pts[:,0:1]
        y = pts[:,1:2]
        z = pts[:,2:3]
        out = -2*pi * torch.cos(2*pi*x) * torch.cos(2*pi*y) * torch.sin(2*pi*z)
        return out
    
    def right_hand_side(pts):
        x = pts[:,0:1]
        y = pts[:,1:2]
        z = pts[:,2:3]
        out = (12*pi^2 + 1)*torch.cos(2*pi*x)*torch.cos(2*pi*y)*torch.cos(2*pi*z)
        return out
    
    
    