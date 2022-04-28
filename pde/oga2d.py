import torch
import numpy as np

# pi = 3.1415926535897932384626

## =====================================
## 2 dimensional exact solution function 
    
class DataCos1m2d_DirichletBC:

    """ 2nd order elliptic PDE in 2D:
        \Omega:    (-1,1)*(-1,1)
             m:    1
           eqn:    [(-\Delta)^m] u + u = f, in \Omega
                                  u = g, on \partial \Omega
    """

    def __init__(self):
        pass

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
        val[..., 1] = self.dx_solution(p)
        val[..., 2] = self.dy_solution(p)
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


class DataCos1m2d_NeumannBC:
    """ 2nd order elliptic PDE in 2D:
        \Omega:    (-1,1)*(-1,1)
             m:    1
           eqn:    [(-\Delta)^m] u + u = f, in \Omega
                              du/dn = 0, on \partial \Omega
    """

    def __init__(self):
        pass

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
        val[..., 1] = self.dx_solution(p)
        val[..., 2] = self.dy_solution(p)
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


class DataPoly2m2d_NeumannBC:
    """ 4nd order elliptic PDE in 2D:
        \Omega:    (-1,1)*(-1,1)
             m:    2
           eqn:    [(-\Delta)^m] u + u = f, in \Omega
                               BN^0(u) = 0, on \partial \Omega
                               BN^1(u) = 0, on \partial \Omega
    """

    def __init__(self):
        pass

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
        val[..., 1] = self.dx_solution(p)
        val[..., 2] = self.dy_solution(p)
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

    def D2_gradient(self, p):
        """ All the 2nd derivatives of the exact solution 
        """

        pi = np.pi
        val = torch.zeros_like(p)
        val[..., 1] = self.dxx_solution(p)
        val[..., 2] = self.dxy_solution(p)
        val[..., 3] = self.dyy_solution(p)
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