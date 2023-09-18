import torch
import numpy as np
import torch.nn.functional as F

## =====================================
## general activations
## tensor

class ActivationFunction():

    class ReluPower():
        """
            ReLU to the power k.
        """

        def __init__(self, degree):
            super().__init__()
            self.degree = degree

        def relu(self, p):
            val = F.relu(p).pow(self.degree)
            return val

        def relu_derivative(self, p, m):
            coefficient = np.math.factorial(self.degree) / np.math.factorial(self.degree-m)
            val = coefficient * F.relu(p).pow(self.degree-m)
            return val

    class Bspline():
        """
            bspline basis function with deg=degree, 
            when degree=1,  bspline is hat.
        """

        def __init__(self, degree):
            super().__init__()
            self.degree = degree
            self.weight = torch.zeros(degree+2)
            for i in range(degree+2):
                weight = 1
                for j in range(degree+2):
                    if j != i:
                        weight *= 1/(i-j)
                self.weight[i] = weight

        def bspline(self, p):
            val = 0 
            for i in range(len(self.weight)):
                val += self.weight[i] * F.relu(i - p).pow(self.degree)
            val *= (self.degree+1)
            return val

        def bspline_derivative(self, p, m):
            val = 0 
            coefficient = np.math.factorial(self.degree) / np.math.factorial(self.degree-m)
            for i in range(len(self.weight)):
                val += self.weight[i] * F.relu(i - p).pow(self.degree-m) * (-1)**m * coefficient
            val *= (self.degree+1)
            return val

    class Sigmoid():
        """
            sigmoidal function and derivatives.
        """
        
        def __init__(self):
            pass

        def sigmoid(self, p):
            val = 1 / (1 + torch.exp(-p))
            return val

        def sigmoid_derivative(self, p, m):

            assert (m >= 1) & (m <= 3) 

            def dsigmoid(p):
                val = torch.exp(-p) / (torch.exp(-p) + 1).pow(2)
                return val

            def d2sigmoid(p):
                val = (2*torch.exp(-2*p)) / (torch.exp(-p) + 1).pow(3) - \
                    torch.exp(-p) / (torch.exp(-p) + 1).pow(2)
                return val

            def d3sigmoid(p):
                val = torch.exp(-p) / (torch.exp(-p) + 1).pow(2) - \
                    (6*torch.exp(-2*p)) / (torch.exp(-p) + 1).pow(3) + \
                    (6*torch.exp(-3*p)) / (torch.exp(-p) + 1).pow(4)
                return val

            val_list = {
                1: dsigmoid,
                2: d2sigmoid,
                3: d3sigmoid
            }
            val = val_list[m](p)
            return val


    def __init__(self, ftype, *args) -> None:
        super().__init__()
        self.ftype = ftype
        self.degree = 0
        if self.ftype != "sigmoid":
            for arg in args:
                self.degree = arg
        
    def activate(self, p):
        val_list = {
            "relu": self.ReluPower(self.degree).relu,
            "bspline": self.Bspline(self.degree).bspline,
            "sigmoid": self.Sigmoid().sigmoid
        }
        val = val_list[self.ftype](p)
        return val

    def dactivate(self, p, m):
        val_list = {
            "relu": self.ReluPower(self.degree).relu_derivative,
            "bspline": self.Bspline(self.degree).bspline_derivative,
            "sigmoid": self.Sigmoid().sigmoid_derivative
        }
        val = val_list[self.ftype](p, m)
        return val