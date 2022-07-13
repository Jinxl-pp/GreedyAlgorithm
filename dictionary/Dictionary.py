import sys
from sympy import Ellipse
sys.path.append('../')

import torch
import torch.nn as nn 
import numpy as np
from optimization import *
from abc import ABC

dtype = torch.float64
torch.set_default_dtype(dtype)

## =====================================
## general dictionaries
## standard DNN model
## single neuron and activations
## tensor


class ShallowNN(nn.Module):
    def __init__(self, in_dim, width, out_dim, sigma):
        super(ShallowNN, self).__init__()
        self.layer1 = nn.Linear(in_dim,width)
        self.layer2 = nn.Linear(width,out_dim)
        self.F = sigma    
        
    def forward(self, out):
        out = self.layer1(out)
        out = self.F(out)
        out = self.layer2(out)
        return out
    

class BasisFunction(nn.Module):
    def __init__(self, base_dim, activation_function):
        super(BasisFunction, self).__init__()
        self.layer = nn.Linear(base_dim, 1)
        self.activation_function = activation_function
        
    def update_neuron(self, parameters):
        weight = parameters.weight.requires_grad_(True)
        bias = parameters.bias.requires_grad_(True)
        param_dict = {'layer.weight': weight, 'layer.bias': bias}
        self.load_state_dict(param_dict, strict=True)

    def forward(self, p):
        val = self.layer(p)
        val = self.activation_function(val)
        return val
    

class AbstractDictionary(ABC):
    
    @abstractmethod
    def argmaxopt(self, Residual_plus_inner_prod, quadrature_points):
        pass
    
    @abstractmethod
    def nonlinearargmaxopt(self, ... )
    
class NNDict(AbstractDictionary):
    
class AbstractFunc(ABC):
    
    @abstractmethod
    def evaluate(point):
        pass

class Dictionary(BasisFunction):
    
    def __init__(self, 
                base_dim, 
                activation_function,
                device,
                optimizer,
                params_domain,
                params_mesh_size):

        """
        The STANDARD general dictionary for shallow neural networks,
                { \sigma(w*x + b): (w, b) \in R^{d+1} }
        INPUT:
            base_dim: the dimension of the background space R^d,
            activation_function: general nonlinear functions,
            device: cpu or cuda
            optimizer: training algorithms for the argmax-subproblem,
            params_domain: torch.tensor object, for b only, 
                            shape = 1-by-2,
            params_mesh_size: a dict object, 
                            len = param_dim.
        """
        super(Dictionary, self).__init__(base_dim, activation_function)

        self.base_dim = base_dim

        self.pi = np.pi
        self.device = device
        self.optimizer = optimizer
        self.params_domain = params_domain
        self.params_mesh_size = params_mesh_size

    def _index_to_sub(self, index, param_shape):
        dim = self.base_dim
        if dim == 1 or dim == 2:
            b = index % param_shape[1]
            a = ((index-b) / param_shape[1]).long()
            sub = torch.cat([a,b], dim=1)
        elif dim == 3:
            c = index % param_shape[2]
            b = ((index-c)/param_shape[2] % param_shape[1]).long()
            a = (((index-c)/param_shape[2]-b)/param_shape[1]).long()
            sub = int(a), int(b), int(c) 
        return sub

    def _gather_vertical_param(self):
        assert self.params_domain.shape[0] == 1

        # the lower and upper bound for b
        min_val = self.params_domain[0][0]
        max_val = self.params_domain[0][1]

        if self.base_dim == 1:
            assert len(self.params_mesh_size) == 1
            N = (max_val-min_val) / self.params_mesh_size['b']
            N = int(N) + 1
            t = torch.tensor([-1.,1.]).to(self.device)
            b = torch.linspace(min_val, max_val, N).to(self.device)
            theta = torch.meshgrid(t, b)
            w = theta[0].reshape(-1,1)
            b = theta[1].reshape(-1,1)
            return theta, (w, b)

        elif self.base_dim == 2:
            assert len(self.params_mesh_size) == 2
            N1 = 2*self.pi / self.params_mesh_size['phi']
            N2 = (max_val-min_val) / self.params_mesh_size['b']
            N1 = int(N1) + 1 
            N2 = int(N2) + 1
            phi = torch.linspace(0, 2*self.pi, N1).to(self.device)
            b = torch.linspace(min_val, max_val, N2).to(self.device)
            theta = torch.meshgrid(phi, b)
            w1 = torch.cos(theta[0]).reshape(-1,1)
            w2 = torch.sin(theta[0]).reshape(-1,1)
            b = theta[1].reshape(-1,1)
            return theta, (w1, w2, b)

        elif self.base_dim == 3:
            assert len(self.params_mesh_size) == 3
            N1 = 2*self.pi / self.params_mesh_size['phi']
            N2 = self.pi / self.params_mesh_size['t']
            N3 = (max_val-min_val) / self.params_mesh_size['b']
            N1 = int(N1) + 1 
            N2 = int(N2) + 1
            N3 = int(N3) + 1
            phi = torch.linspace(0, 2*self.pi, N1).to(self.device)
            t = torch.linspace(0, self.pi, N2).to(self.device)
            b = torch.linspace(min_val, max_val, N3).to(self.device)
            theta = torch.meshgrid(phi, t, b)
            w1 = (torch.cos(theta[1]) * torch.cos(theta[0])).reshape(-1,1)
            w2 = (torch.cos(theta[1]) * torch.sin(theta[0])).reshape(-1,1)
            w3 = torch.sin(theta[1]).reshape(-1,1)
            b = theta[2].reshape(-1,1)
            return theta, (w1, w2, w3, b)

        else:
            raise RuntimeError("Dimension overflows")

    def _basis_train():
        pass

    def select_initial_basis(self, pde_energy):
        dim = self.base_dim
        theta, param = self._gather_vertical_param()
        param_shape = theta[0].shape
        
        # we should modify here 
        # pde_energy should contain assert-language to avoid dimensionlaity inconsistency
        # only index[0] or some top-k indices
        init_loss = pde_energy(param)
        _, index = torch.sort(init_loss, descending=False, dim=0)
        sub = self._index_to_sub(index[0], param_shape)

        if dim==1:
            w = theta[0][sub[:,0:1],sub[:,1:2]]
            b = theta[1][sub[:,0:1],sub[:,1:2]]
            initial_param = torch.cat([w,b], dim=1).to(self.device)
        elif dim==2:
            t = theta[0][sub[:,0:1],sub[:,1:2]]
            b = theta[1][sub[:,0:1],sub[:,1:2]]
            initial_param = torch.cat([t,b], dim=1).to(self.device)
        elif dim==3:
            p = theta[0][sub[:,0:1],sub[:,1:2],sub[:,2:3]]
            t = theta[1][sub[:,0:1],sub[:,1:2],sub[:,2:3]]
            b = theta[2][sub[:,0:1],sub[:,1:2],sub[:,2:3]]
            initial_param = torch.cat([p,t,b], dim=1).to(self.device)


    def find_optimal_basis():
        pass


