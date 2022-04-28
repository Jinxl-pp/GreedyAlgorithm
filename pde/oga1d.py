import torch
import numpy as np

pi = np.pi
# pi = 3.1415926535897932384626

## =====================================
## 1 dimensional exact solution function
    
    
class DataCos1m1d_DirichletBC:
    """ 2nd order elliptic PDE in 1D:
        \Omega:    (-1,1)
             m:    1
           eqn:    [(-\Delta)^m] u + u = f, in \Omega
                                     u = 0, on \partial \Omega
    """
    
    def __init__(self):
        pass

    def solution(self, p):
        """ The exact solution of the PDE
            INPUT:
                p: tensor object, 
        """

        val = torch.cos(pi/2*p)
        return val
    
    def gradient(self, p):
        """ The gradient of the exact solution 
        """

        val = -pi/2*torch.sin(pi/2*p)
        return val
    
    def source(self, p):
        """ The right-hand-side term of the PDE
        """

        val = pi**2/4 * torch.cos(pi/2*p) + torch.cos(pi/2*p)
        return val


class DataCos1m1d_NeumannBC:
    """ 2nd order elliptic PDE in 1D:
        \Omega:    (-1,1)
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
        val = torch.cos(pi*p)
        return val
    
    def gradient(self, p):
        """ The gradient of the exact solution 
        """

        val = -pi*torch.sin(pi*p)
        return val
    
    def source(self, p):
        """ The right-hand-side term of the PDE
        """

        val = pi**2 * torch.cos(pi*p) + torch.cos(pi*p)
        return val