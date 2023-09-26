"""
Created on Mon Sept 25 16:06 2023

@author: Jinpp (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Class of 1d polynomial functions served as the 
        real solution to test the approximation of 
        neural networks.
        The dimensionality and PDE information are all
        recorded inside the class.
@modifications: to be added
"""

import torch
import numpy as np
from .pde import PDE


##=====================================================##
#            zero Neumann boundary condition            #
##=====================================================##

class DataPoly_4th_1d_NBC(PDE):
    """ 4nd order elliptic PDE in 1D:
        \Omega:    (-1,1)
             m:    2
           eqn:    [(-\Delta)^m] u + u = f, in \Omega
                                 du/dx = 0, on \partial \Omega
                           [d/dx]^2(u) = 0, on \partial \Omega
    """
    
    def __init__(self):
        """
        dim: the dimensionality of data
        order: the order of partial differential equation
        """
        self.dim = 1
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
        val = (1 - p).pow(4) * (1 + p).pow(4)
        return val
        
    def gradient(self, p):
        """ The gradient of the exact solution 
        """

        val = 4*(p - 1).pow(3)*(p + 1).pow(4) + 4*(p - 1).pow(4)*(p + 1).pow(3)
        return val
    
    def hessian(self, p):
        """ The 2nd derivatives of the exact solution 
        """
        
        val = 8*(p.pow(2) - 1).pow(2) * (7*p.pow(2) - 1)
        return val
    
    def source(self, p):
        """ The right-hand-side term of the PDE
        """
        
        val = p.pow(8) - 4*p.pow(6) + 1686*p.pow(4) - 1444*p.pow(2) + 145.
        return val
