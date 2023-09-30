"""
Created on Tue Sept 27 23:43 2022

@author: Jin Xianlin (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Class of 1d cosine-type functions served as the 
        real solution to test the approximation of 
        neural networks. Dimensionality and PDE information
        are all recorded in the class.
@modifications: to be added
"""

import torch
import numpy as np
from .pde import PDE

# pi = 3.1415926535897932384626


##=======================================================##
#            zero Dirichlet boundary condition            #
##=======================================================##

class DataCos_2nd_1d_DBC(PDE):
    """ 2nd order elliptic PDE in 1D:
        \Omega:    (-1,1)
             m:    1
           eqn:    [(-\Delta)^m] u + u = f, in \Omega
                                     u = 0, on \partial \Omega
    """
    
    def __init__(self):
        """
        dim: the dimensionality of data
        order: the order of partial differential equation
        """
        self.dim = 1
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
        val = torch.cos(pi/2*p)
        return val
    
    def gradient(self, p):
        """ The gradient of the exact solution 
        """

        pi = np.pi
        val = -pi/2*torch.sin(pi/2*p)
        return val
    
    def source(self, p):
        """ The right-hand-side term of the PDE
        """

        pi = np.pi
        val = pi**2/4 * torch.cos(pi/2*p) + torch.cos(pi/2*p)
        return val
    

##=======================================================##
#            zero Dirichlet boundary condition            #
##=======================================================##

class Data_poisson_1d_DBC(PDE):
    """ 2nd order elliptic PDE in 1D:
        \Omega:    (-1,1)
             m:    1
           eqn:    [-\Delta]u = f, in \Omega
                            u = 0, on \partial \Omega
    """
    
    def __init__(self):
        """
        dim: the dimensionality of data
        order: the order of partial differential equation
        """
        self.dim = 1
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
        val = torch.sin(2*pi*p) * torch.cos(pi*p)
        return val
    
    def gradient(self, p):
        """ The gradient of the exact solution 
        """

        pi = np.pi
        val = 2*pi * torch.cos(pi*p) * torch.cos(2*pi*p) - \
                pi * torch.sin(pi*p) * torch.sin(2*pi*p)
        return val
    
    def source(self, p):
        """ The right-hand-side term of the PDE
        """

        pi = np.pi
        val = 5*pi**2 * torch.cos(pi*p) * torch.sin(2*pi*p) + \
                4*pi**2 * torch.cos(2*pi*p) * torch.sin(pi*p)
        return val



##=====================================================##
#            zero Neumann boundary condition            #
##=====================================================##

class DataCos_2nd_1d_NBC(PDE):
    """ 2nd order elliptic PDE in 1D:
        \Omega:    (-1,1)
             m:    1
           eqn:    [(-\Delta)^m] u + u = f, in \Omega
                              du/dn = 0, on \partial \Omega
    """

    def __init__(self):
        """
        dim: the dimensionality of data
        order: the order of partial differential equation
        """
        self.dim = 1
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
        val = torch.cos(pi*p)
        return val
    
    def gradient(self, p):
        """ The gradient of the exact solution 
        """

        pi = np.pi
        val = -pi*torch.sin(pi*p)
        return val
    
    def source(self, p):
        """ The right-hand-side term of the PDE
        """
        pi = np.pi
        val = pi**2 * torch.cos(pi*p) + torch.cos(pi*p)
        return val
    
    def trace(self, p):
        """ The Neumann-trace operator of the PDE
        """
        return 0. * p