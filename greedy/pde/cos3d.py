"""
Created on Tue Feb 07 12:13 2023

@author: Jin Xianlin (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Class of 3d cosine-type functions served as the 
        real solution to test the approximation of 
        neural networks.
        The dimensionality and PDE information are all
        recorded inside the class.
@modifications: to be added
"""

import torch
import numpy as np
from .pde import PDE

# pi = 3.1415926535897932384626

##=====================================================##
#            zero Neumann boundary condition            #
##=====================================================##   
    
class DataCos_2nd_3d_NBC:
    """ 2nd order elliptic PDE in 3D:
        \Omega:    (-1,1)*(-1,1)*(-1,1)
             m:    1
           eqn:    [(-\Delta)^m] u + u = f, in \Omega
                              du/dn = 0, on \partial \Omega
    """
    def __init__(self):
        """
        dim: the dimensionality of data
        order: the order of partial differential equation
        """
        self.dim = 3
        self.order = 2
        
    def dimension(self):
        return self.dim

    def equation_order(self):
        return self.order
    
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
        val[..., 0:1] = self.dx_solution(p)
        val[..., 1:2] = self.dy_solution(p)
        val[..., 2:3] = self.dz_solution(p)
        return val
    
    def source(self, p):
        """ The right-hand-side term of the PDE
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        z = p[..., 2:3]
        val = (12*pi**2 + 1) * torch.cos(2*pi*x) * torch.cos(2*pi*y) * torch.cos(2*pi*z)
        return val
    
    