import sys
sys.path.append('../')

import torch
import torch.nn as nn 
from torch.nn.parameter import Parameter
import numpy as np

import dictionary as dt
from optimization import generator as gen

dtype = torch.float64
torch.set_default_dtype(dtype)

## =====================================
## general SNN dictionary

class SNNDictionary(dt.AbstractDictionary): # shallow_neural_dict
    
    def __init__(self, 
                geo_dim, 
                activation_function,
                optimizer,
                param_b_domain,
                params_mesh_size,
                device,
                parallel_search=False):

        """
        The STANDARD general dictionary for shallow neural networks,
                { \sigma(w*x + b): (w, b) \in R^{d+1} }
        INPUT:
            geo_dim: the dimension of the background space R^d.
            activation_function: general nonlinear functions.
            optimizer: training algorithms for the argmax-subproblem.
            params_domain: torch.tensor object, for b only.
                            shape = 1-by-2,
            params_mesh_size: a dict object, 
                            len = param_dim.
            device: cpu or cuda.
            parallel_search: option for parallel optimization.
        """
        super(SNNDictionary, self).__init__()
        assert (geo_dim >= 1) and (geo_dim <= 3), "Standard SNN only supports dimenion no more than 3."
        
        self.geo_dim = geo_dim
        self.pi = np.pi
        self.optimizer = optimizer
        self.params_domain = self._get_domain(param_b_domain)
        self.param_mesh_size = param_mesh_size
        
        self.device = device
        self.parallel_search = parallel_search
        
        
    def _get_domain(self, param_b_domain):
        if self.geo_dim == 1:
            params_domain = param_b_domain
        elif self.geo_dim == 2:
            param_t_domain = torch.tensor([[0., 2*self.pi]])
            params_domain = torch.cat([param_t_domain, param_b_domain], dim=0)
        elif self.geo_dim == 3:
            param_t_domain = torch.tensor([[0., 2*self.pi]])
            param_p_domain = torch.tensor([[0., self.pi]])
            params_domain = torch.cat([param_t_domain, param_p_domain, param_b_domain], dim=0)
        return params_domain
    

    def _index_to_sub(self, index, param_shape):
        dim = self.geo_dim
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
    
    
    def _polar_to_cartesian(self, theta):
        
        # coordinate transformation
        if self.geo_dim == 1:
            w = theta[0].reshape(-1,1)
            b = theta[1].reshape(-1,1)
            return (w, b)
        elif self.geo_dim == 2:
            w1 = torch.cos(theta[0]).reshape(-1,1)
            w2 = torch.sin(theta[0]).reshape(-1,1)
            b = theta[1].reshape(-1,1)
            return (w1, w2, b)
        elif self.geo_dim == 3:
            w1 = torch.cos(theta[0]) * torch.sin(theta[1]).reshape(-1,1)
            w2 = torch.sin(theta[0]) * torch.sin(theta[1]).reshape(-1,1)
            w3 = torch.cos(theta[1]).reshape(-1,1)
            b = theta[2].reshape(-1,1)
            return (w1, w2, w3, b)
        else:
            raise RuntimeError("Dimension error.")

    

    def _gather_vertical_param(self):

        # get all paramter samples, methods differ w.r.t. dimensions
        if self.geo_dim == 1:
            
            # get lower and upper bounds for b
            assert self.params_domain.shape[0] == 1
            min_val_b = self.params_domain[0][0]
            max_val_b = self.params_domain[0][1]
            
            # generate param-mesh for (w,b)
            N = (max_val_b - min_val_b) / self.param_mesh_size
            N = int(N) + 1
            t = torch.tensor([-1.,1.]).to(self.device)
            b = torch.linspace(min_val_b, max_val_b, N).to(self.device)
            theta = torch.meshgrid(t, b)

        elif self.geo_dim == 2:
            
            # get lower and upper bounds for phi and b, where phi is the polar coordinate
            assert self.params_domain.shape[0] == 2
            min_val_t = self.params_domain[0][0]
            max_val_t = self.params_domain[0][1]
            min_val_b = self.params_domain[1][0]
            max_val_b = self.params_domain[1][1]
            
            # generate param-mesh for (w1,w2,b), where w1 = cos(t), w2 = sin(phi)
            N1 = (max_val_t - min_val_t) / self.param_mesh_size
            N2 = (max_val_b - min_val_b) / self.param_mesh_size
            N1 = int(N1) + 1 
            N2 = int(N2) + 1
            t = torch.linspace(min_val_t, max_val_t, N1).to(self.device)
            b = torch.linspace(min_val_b, max_val_b, N2).to(self.device)
            theta = torch.meshgrid(t, b)

        elif self.geo_dim == 3:
            
            # get lower and upper bounds for phi, t and b, where phi, t are the polar coordinates
            assert self.params_domain.shape[0] == 3
            min_val_t = self.params_domain[0][0]
            max_val_t = self.params_domain[0][1]
            min_val_p = self.params_domain[1][0]
            max_val_p = self.params_domain[1][1]
            min_val_b = self.params_domain[2][0]
            max_val_b = self.params_domain[2][1]
            
            # generate param-mesh for (w1,w2,w3,b), where w1 = cos(phi)*sin(t), w2 = sin(phi)*sin(t), w3 = cos(t)
            N1 = (max_val_t - min_val_t) / self.param_mesh_size
            N2 = (max_val_p - min_val_p) / self.param_mesh_size
            N3 = (max_val_b - min_val_b) / self.param_mesh_size
            N1 = int(N1) + 1 
            N2 = int(N2) + 1
            N3 = int(N3) + 1
            t = torch.linspace(min_val_t, max_val_t, N1).to(self.device)
            p = torch.linspace(min_val_p, max_val_p, N2).to(self.device)
            b = torch.linspace(min_val_b, max_val_b, N3).to(self.device)
            theta = torch.meshgrid(t, p, b)
        
        else:
            raise RuntimeError("Dimension overflows")
        
        param = self._polar_to_cartesian(theta)
        return theta, param
        
        
    # def _get_bilinear??
            

    def _select_initial_elements(self, pde_energy):
        
        dim = self.geo_dim
        all_theta, all_param = self._gather_vertical_param()
        param_shape = theta[0].shape
        
        # we should modify here 
        # pde_energy should contain assert-language to avoid dimensionlaity inconsistency
        # only index[0] or some top-k indices
        all_init_loss = pde_energy(all_param)
        _, index = torch.sort(all_init_loss, descending=False, dim=0)
        sub = self._index_to_sub(index[0], param_shape)

        if dim==1:
            we = all_theta[0][sub[:,0:1],sub[:,1:2]]
            bi = all_theta[1][sub[:,0:1],sub[:,1:2]]
            initial_param = torch.cat([we,bi], dim=1).to(self.device)
        elif dim==2:
            th = all_theta[0][sub[:,0:1],sub[:,1:2]]
            bi = all_theta[1][sub[:,0:1],sub[:,1:2]]
            initial_param = torch.cat([th,bi], dim=1).to(self.device)
        elif dim==3:
            ph = all_theta[0][sub[:,0:1],sub[:,1:2],sub[:,2:3]]
            th = all_theta[1][sub[:,0:1],sub[:,1:2],sub[:,2:3]]
            bi = all_theta[2][sub[:,0:1],sub[:,1:2],sub[:,2:3]]
            initial_param = torch.cat([ph,th,bi], dim=1).to(self.device)
        return initial_param
    
    
    def _get_optimizer(self, theta, optimizer_type):
        
        # theta contains weights and bias
        bias = Parameter(bias)
        if self.geo_dim == 1:
            b = Parameter(theta[0])
            params_input = [b]
        elif self.geo_dim == 2:
            t = Parameter(theta[0])
            b = Parameter(theta[1])
            params_input = [t,b]
        elif self.geo_dim == 3:
            t = Parameter(theta[0])
            p = Parameter(theta[1])
            b = Parameter(theta[2])
            params_input = [p,t,b]
            
        # use optimizer-generator to get optimizer
        optimizer = gen.Generator(params_input, self.params_domain).get_optimizer(optimizer_type)
        
        return params_input, optimizer
    
    
    def _argmax_optimize_seq(self, pde_energy, theta_list, optimizer_type):
        
        # get the number of parameter sets 
        num_search = theta_list.shape[0]
        evaluate_list = torch.zeros(num_search, 1)
        
        # train parameters of each set and evaluate their ultimate energy
        epochs = 100
        for i in range(num_search): 
            theta = theta_list[i,...]
            param, optimizer = self._get_optimizer(theta, optimizer_type)
            for epoch in range(epochs):
                def closure():
                    optimizer.zero_grad()
                    param = self._polar_to_cartesian(param)
                    loss = pde_energy(param)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                new_loss = closure().detach()
            theta_list[i,0:-1] = weight
            theta_list[i,-1:] = bias
            evaluate_list[i] = new_loss
        
        return theta_list, evaluate_list            
            
    
    def _argmax_optimize_par(self, pde_energy, theta_list, optimizer_type):
        """ 
        Not sure if this is neccessary. Default not calling.
        """
        # import extra packages
        # import random
        # from mpi4py import MPI 
        pass
    
    def _argmax_optimize(self, pde_energy, theta_list, optimizer_type):
        """ 
        Train multiple elements simultaneously by optimizing their parameters (theta_list) 
        INPUT:
            pde_energy: the evaluating function of PDE's energy.
            theta_list: an m-by-k tensor, where 
                        m is the number of parameter set,
                        k is the number of parameters in each set 
                        (k=2, when dim=1 or dim=2, k=3 when dim=3).
            optimizer_type: options for choosing optimizers. Available for
                        "pgd", projected gradient descent (PGD) method,
                        "fista", acceleratd PGD method,
                        "lbfgs", L-BFGS method.
        """
        
        if not self.parallel_search:
            return self._argmax_optimize_seq(pde_energy, theta_list, optimizer_type)
        else:
            return self._argmax_optimize_par(pde_energy, theta_list, optimizer_type)

    def find_optimal_element(self, pde_energy):
        
        # initial guess 
        theta_list = self._select_initial_elements(pde_energy)
