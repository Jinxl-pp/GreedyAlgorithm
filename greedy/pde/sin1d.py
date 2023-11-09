"""
Created on Wen Nov 8 16:15 2023

@author: Jin Xianlin (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Class of 1d sine-type functions served as the 
        initial condition of the 1d Burgers' equation.
@modifications: to be added
"""

import torch
import numpy as np
from .pde import PDE

class DataSin_1d_PBC(PDE):
    
    def __init__(self):
        """
        dim: the dimensionality of data
        order: the order of partial differential equation
        """
        self.dim = 1
        self.order = 1
    
    def dimension(self):
        return self.dim
    
    def equation_order(self):
        return self.order
    
    def solution(self):
        pass
    
    def gradient(self):
        pass
    
    def source(self, p):
        pi = np.pi
        val = 0.25 + 0.5*torch.sin(pi*p)
        return val