"""
Created on Tue Sept 27 23:33 2022

@author: Jinpp (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Class of 2d polynomial functions served as the 
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

class DataPoly_4th_2d_NBC(PDE):
    """ 4nd order elliptic PDE in 2D:
        \Omega:    (-1,1)*(-1,1)
             m:    2
           eqn:    [(-\Delta)^m] u + u = f, in \Omega
                               BN^0(u) = 0, on \partial \Omega
                               BN^1(u) = 0, on \partial \Omega
    """

    def __init__(self):
        """
        dim: the dimensionality of data
        order: the order of partial differential equation
        """
        self.dim = 2
        self.order = 4
        
    def dimension(self):
        return self.dim

    def equation_order(self):
        return self.order

    def solution(self, p):
        """ The exact solution of the PDE
            INPUT:
                p: tensor object
        """
        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = (x.pow(2) - 1).pow(4)*(y.pow(2) - 1).pow(4)
        return val
    
    def dx_solution(self, p):
        """ The derivative on x-axis of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = 8*x*(x.pow(2) - 1).pow(3)*(y.pow(2) - 1).pow(4)
        return val
    
    def dy_solution(self, p):
        """ The derivative on y-axis of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = 8*y*(x.pow(2) - 1).pow(4)*(y.pow(2) - 1).pow(3)
        return val

    def gradient(self, p):
        """ The gradient of the exact solution 
        """

        pi = np.pi
        val = torch.zeros_like(p)
        val[..., 0:1] = self.dx_solution(p)
        val[..., 1:2] = self.dy_solution(p)
        return val

    def dxx_solution(self, p):
        """ The 2nd order pure derivative on x-axis
            of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = 8*(x.pow(2) - 1).pow(3)*(y.pow(2) - 1).pow(4) + \
               48*x.pow(2)*(x.pow(2) - 1).pow(2)*(y.pow(2) - 1).pow(4)
        return val
    
    def dxy_solution(self, p):
        """ The 2nd order mixed derivative
            of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = 64*x*y*(x.pow(2) - 1).pow(3)*(y.pow(2) - 1).pow(3)
        return val

    def dyy_solution(self, p):
        """ The 2nd order pure derivative on y-axis
            of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = 8*(x.pow(2) - 1).pow(4)*(y.pow(2) - 1).pow(3) + \
               48*y.pow(2)*(x.pow(2) - 1).pow(4)*(y.pow(2) - 1).pow(2)
        return val

    def hessian(self, p):
        """ All the 2nd derivatives of the exact solution 
        """

        pi = np.pi
        val = torch.zeros_like(p)
        val[..., 0:1] = self.dxx_solution(p)
        val[..., 1:2] = self.dxy_solution(p)
        val[..., 2:3] = self.dyy_solution(p)
        return val

    def source(self, p):
        """ The right-hand-side term of the PDE
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = 144*(x.pow(2) - 1).pow(2)*(y.pow(2) - 1).pow(4) + \
                128*(x.pow(2) - 1).pow(3)*(y.pow(2) - 1).pow(3) + \
                144*(x.pow(2) - 1).pow(4)*(y.pow(2) - 1).pow(2) + \
                (x.pow(2) - 1).pow(4)*(y.pow(2) - 1).pow(4) + \
                384*x.pow(4)*(y.pow(2) - 1).pow(4) + \
                384*y.pow(4)*(x.pow(2) - 1).pow(4) + \
                1152*x.pow(2)*(x.pow(2) - 1)*(y.pow(2) - 1).pow(4) + \
                1152*y.pow(2)*(x.pow(2) - 1).pow(4)*(y.pow(2) - 1) + \
                768*x.pow(2)*(x.pow(2) - 1).pow(2)*(y.pow(2) - 1).pow(3) + \
                768*y.pow(2)*(x.pow(2) - 1).pow(3)*(y.pow(2) - 1).pow(2) + \
                4608*x.pow(2)*y.pow(2)*(x.pow(2) - 1).pow(2)*(y.pow(2) - 1).pow(2)
        return val