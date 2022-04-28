import torch
import numpy as np

# pi = 3.1415926535897932384626

## =====================================
## 3 dimensional exact solution function     
    
class DataCos1m3d_NeumannBC:
    """ 2nd order elliptic PDE in 3D:
        \Omega:    (-1,1)*(-1,1)*(-1,1)
             m:    1
           eqn:    [(-\Delta)^m] u + u = f, in \Omega
                              du/dn = 0, on \partial \Omega
    """
    def __init__(self):
        pass
    
    def solution(self, p):
        """ The exact solution of the PDE
            INPUT:
                p: tensor object, 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        z = p[..., 2:3]
        val = torch.cos(2*pi*x) * torch.cos(2*pi*y) * torch.cos(2*pi*z)
        return val
    
    def dx_solution(self, p):
        """ The derivative on x-axis of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        z = p[..., 2:3]
        val = -2*pi * torch.cos(2*pi*y) * torch.cos(2*pi*z) * torch.sin(2*pi*x)
        return val
    
    def dy_solution(self, p):
        """ The derivative on y-axis of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        z = p[..., 2:3]
        val = -2*pi * torch.cos(2*pi*x) * torch.cos(2*pi*z) * torch.sin(2*pi*y)
        return val
    
    def dz_solution(self, p):
        """ The derivative on z-axis of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        z = p[..., 2:3]
        val = -2*pi * torch.cos(2*pi*x) * torch.cos(2*pi*y) * torch.sin(2*pi*z)
        return val

    def gradient(self, p):
        """ The gradient of the exact solution 
        """

        pi = np.pi
        val = torch.zeros_like(p)
        val[..., 1] = self.dx_solution(p)
        val[..., 2] = self.dy_solution(p)
        val[..., 3] = self.dz_solution(p)
        return val
    
    def source(self, p):
        """ The right-hand-side term of the PDE
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        z = p[..., 2:3]
        val = (12*pi^2 + 1)*torch.cos(2*pi*x)*torch.cos(2*pi*y)*torch.cos(2*pi*z)
        return val
    
    