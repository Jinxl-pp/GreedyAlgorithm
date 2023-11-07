"""
Created on Tue Sept 27 23:55 2022

@author: Jin Xianlins (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Class of 2d cosine-type functions served as the 
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

    
##=======================================================##
#            zero Dirichlet boundary condition            #
##=======================================================##

class DataCos_2nd_2d_DBC(PDE):

    """ 2nd order elliptic PDE in 2D:
        \Omega:    (-1,1)*(-1,1)
             m:    1
           eqn:    [(-\Delta)^m] u + u = f, in \Omega
                                  u = g, on \partial \Omega
    """

    def __init__(self):
        """
        dim: the dimensionality of data
        order: the order of partial differential equation
        """
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

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = torch.cos(pi/2*x) * torch.cos(pi/2*y)
        return val
    
    def dx_solution(self, p):
        """ The derivative on x-axis of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = -pi/2 * torch.cos(pi/2*y) * torch.sin(pi/2*x)
        return val
    
    def dy_solution(self, p):
        """ The derivative on y-axis of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = -pi/2 * torch.cos(pi/2*x) * torch.sin(pi/2*y)
        return val

    def gradient(self, p):
        """ The gradient of the exact solution 
        """

        pi = np.pi
        val = torch.zeros_like(p)
        val[..., 0:1] = self.dx_solution(p)
        val[..., 1:2] = self.dy_solution(p)
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

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = torch.cos(pi/2*x) * torch.cos(pi/2*y) + \
              pi**2/2 * torch.cos(pi/2*x) * torch.cos(pi/2*y)
        return val    


class Data_Poisson_2d_DBC(PDE):

    """ 2nd order elliptic PDE in 2D:
        \Omega:    (-1,1)*(-1,1)
             m:    1
           eqn:     [(-\Delta)^m] u = f, in \Omega
                                  u = g, on \partial \Omega
    """

    def __init__(self):
        """
        dim: the dimensionality of data
        order: the order of partial differential equation
        """
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

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = torch.cos(pi/2*x) * torch.cos(pi/2*y)
        return val
    
    def dx_solution(self, p):
        """ The derivative on x-axis of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = -pi/2 * torch.cos(pi/2*y) * torch.sin(pi/2*x)
        return val
    
    def dy_solution(self, p):
        """ The derivative on y-axis of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = -pi/2 * torch.cos(pi/2*x) * torch.sin(pi/2*y)
        return val

    def gradient(self, p):
        """ The gradient of the exact solution 
        """

        pi = np.pi
        val = torch.zeros_like(p)
        val[..., 0:1] = self.dx_solution(p)
        val[..., 1:2] = self.dy_solution(p)
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

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = pi**2/2 * torch.cos(pi/2*x) * torch.cos(pi/2*y)
        return val    


##=====================================================##
#            zero Neumann boundary condition            #
##=====================================================##

class DataCos_2nd_2d_NBC(PDE):
    """ 2nd order elliptic PDE in 2D:
        \Omega:    (-1,1)*(-1,1)
             m:    1
           eqn:    [(-\Delta)^m] u + u = f, in \Omega
                              du/dn = 0, on \partial \Omega
    """

    def __init__(self):
        """
        dim: the dimensionality of data
        order: the order of partial differential equation
        """
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
        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = torch.cos(2*pi*x) * torch.cos(2*pi*y)
        return val
    
    def dx_solution(self, p):
        """ The derivative on x-axis of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = -2*pi * torch.cos(2*pi*y) * torch.sin(2*pi*x)
        return val
    
    def dy_solution(self, p):
        """ The derivative on y-axis of the exact solution 
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = -2*pi * torch.cos(2*pi*x) * torch.sin(2*pi*y)
        return val

    def gradient(self, p):
        """ The gradient of the exact solution 
        """

        pi = np.pi
        val = torch.zeros_like(p)
        val[..., 0:1] = self.dx_solution(p)
        val[..., 1:2] = self.dy_solution(p)
        return val

    def source(self, p):
        """ The right-hand-side term of the PDE
        """

        pi = np.pi
        x = p[..., 0:1]
        y = p[..., 1:2]
        val = torch.cos(2*pi*x) * torch.cos(2*pi*y) + \
              8*pi**2 * torch.cos(2*pi*x) * torch.cos(2*pi*y)
        return val