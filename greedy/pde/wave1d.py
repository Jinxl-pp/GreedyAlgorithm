"""
Created on Wed Sep 20 16:18 2023

@author: Jin Xianlin (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Class of 1d sine-type functions served as the 
        real solution to test the approximation of 
        neural networks. Dimensionality and PDE information
        are all recorded in the class.
@modifications: to be added
"""
import torch
import numpy as np
from .pde import PDE

class DataWave_1d(PDE):
    """ Combinations of sine functions with a wide 
        range of frequencies.
        Reference:
        [1] 
            Cai W, Li X, Liu L. A phase shift deep neural network for high 
            frequency approximation and wave problems[J]. SIAM Journal on 
            Scientific Computing, 2020, 42(5): A3285-A3312.
        
    """
    
    def __init__(self):
        """
        dim: the dimensionality of data
        order: the order of partial differential equation
        """
        self.dim = 1
        self.order = 0

    def dimension(self):
        return self.dim
    
    def equation_order(self):
        return self.order

    def solution(self, p):
        """ The exact solution of the equation
            INPUT:
                p: tensor object, 
        """
        
        pi = np.pi
        val = torch.zeros(p.shape)
        val[p >= -pi] = 10 * torch.sin(p[p >= -pi]) + 10 * torch.sin(3 * p[p >= -pi])
        val[p >= 0] = 10 * torch.sin(23 * p[p >= 0])  \
                    + 10 * torch.sin(137 * p[p >= 0]) \
                    + 10 * torch.sin(203 * p[p >= 0])
        val[p <-pi] = 0.0
        val[p > pi] = 0.0
        return val
    
    def gradient(self, p):
        pass 
    
    def source(self, p):
        """ The right-hand-side term of the equation
        """
        return self.solution(p)