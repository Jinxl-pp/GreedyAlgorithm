"""
Created on Tue Oct 10 12:27 2023

@author: Jin Xianlins (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Class of 2d tanh-type functions served as the 
        real solution to test the approximation of 
        neural networks.
        The dimensionality and PDE information are all
        recorded inside the class.
@modifications: to be added
"""

import torch
import numpy as np
from .pde import PDE


class Datatanh_2d_DBC(PDE):
    
    """ Allen-Cahn equation in 2D:
        \Omega:    (-1,1)
             t:    (0,1]
           eqn:    ut - l*u_xx + e*(u^3-u) = 0, (-1,1)*(0,1]
                   u(x,t) = gD, {-1,1}*(0,1]
                   u(x,0) = u0(x), (-1,1)
    """
    
    def __init__(self, lam, eps):
        """
        lam: coefficient of -u_xx
        eps: coefficient of nonlinear term
        dim: the dimensionality of data
        order: the highest order of partial differential equation
        """
        self.lam = lam
        self.eps = eps
        self.dim = 2
        self.order = 2
        
    def dimension(self):
        return self.dim

    def equation_order(self):
        return self.order
    
    def solution(self, p):
        """ The exact solution of the PDE
            INPUT:
                p: tensor object
        """

        x = p[..., 0:1]
        t = p[..., 1:2]
        coef1 = np.sqrt(self.eps/self.lam/8.)
        coef2 = self.eps * 0.75
        val = -0.5 + 0.5*torch.tanh(coef1*x - coef2*t)
        return val
    
    def dirichlet(self, p):
        """ The dirichlet boundary value of the solution
            INPUT:
                p: boundary points, tensor object
        """
        pi = np.pi
        val = self.solution(p)
        return val
    
    def source(self, p):
        """ The right-hand-side term of the PDE
        """

        x = p[..., 0:1]
        val = torch.cos(x) * 0.
        return val  
        
    