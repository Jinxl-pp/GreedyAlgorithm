import time
import torch
import torch.nn as nn 
from torch.nn.parameter import Parameter
import numpy as np

from . import dictionary as dt
from ..optimization import generator as gen

dtype = torch.float64
torch.set_default_dtype(dtype)

## =====================================
## general SNN dictionary

class NeuronDictionary2D(dt.AbstractDictionary): # shallow_neural_dict with 2D inputs
    
    def __init__(self, 
                activation,
                optimizer,
                param_b_domain,
                params_mesh_size,
                device,
                parallel_search=False):

        """
        The STANDARD general dictionary for shallow neural networks,
                { \sigma(w*x + b): (w, b) \in R^{3} }
        INPUT:
            activation: nonlinear activation functions.
            optimizer: training algorithms for the argmax-subproblem.
                        if self.optimizer = False, then set best_k=1 in 
                        self._select_initial_elements().
            params_domain: torch.tensor object, for b only.
                            shape = 1-by-2,
            params_mesh_size: a dict object, 
                            len = param_dim.
            device: cpu or cuda.
            parallel_search: option for parallel optimization.
        """
        super(NeuronDictionary2D, self).__init__()
        
        self.geo_dim = 2
        self.pi = np.pi
        
        self.activation = activation
        self.optimizer = optimizer
        self.params_domain = self._get_domain(param_b_domain)
        self.params_mesh_size = params_mesh_size
        
        self.device = device
        self.parallel_search = parallel_search
        
        
    def _get_domain(self, param_b_domain):
        
        # 2D dictionary needs a polar coordinate t and the bias b
        param_t_domain = torch.tensor([[0., 2*self.pi]])
        params_domain = torch.cat([param_t_domain, param_b_domain], dim=0)
        return params_domain
    

    def _index_to_sub(self, index, param_shape):
        
        # from vector-indices to tensor-subscripts
        b = index % param_shape[1]
        a = ((index-b) / param_shape[1]).long()
        sub = torch.cat([a,b], dim=1)
        return sub
    
    
    def _polar_to_cartesian(self, theta):
        
        # coordinate transformation
        w1 = torch.cos(theta[0]).reshape(-1,1)
        w2 = torch.sin(theta[0]).reshape(-1,1)
        b = theta[1].reshape(-1,1)
        return (w1, w2, b)

    
    def _gather_vertical_param(self):

        # get lower and upper bounds for phi and b, where phi is the polar coordinate
        assert self.params_domain.shape[0] == 2
        min_val_t = self.params_domain[0][0]
        max_val_t = self.params_domain[0][1]
        min_val_b = self.params_domain[1][0]
        max_val_b = self.params_domain[1][1]
        
        # generate param-mesh for (w1,w2,b), where w1 = cos(t), w2 = sin(phi)
        N1 = (max_val_t - min_val_t) / self.params_mesh_size
        N2 = (max_val_b - min_val_b) / self.params_mesh_size
        N1 = int(N1) + 1 
        N2 = int(N2) + 1
        t = torch.linspace(min_val_t, max_val_t, N1).to(self.device)
        b = torch.linspace(min_val_b, max_val_b, N2).to(self.device)
        theta = torch.meshgrid(t, b, indexing="ij")
        
        param = self._polar_to_cartesian(theta)
        return theta, param
            

    def _select_initial_elements(self, pde_energy, best_k):
            
        # generate parameter-samples from a fine grid, and 
        # select the top ones by evaluate them in the pde_energy
        all_theta, all_param = self._gather_vertical_param()
        param_shape = all_theta[0].shape
        
        # only index[0] or some top-k indices
        all_init_loss = pde_energy(all_param)
        _, index = torch.sort(all_init_loss, descending=False, dim=0)
        sub = self._index_to_sub(index[0:best_k], param_shape)

        # pick by subscripts
        t = all_theta[0][sub[:,0:1],sub[:,1:2]]
        b = all_theta[1][sub[:,0:1],sub[:,1:2]]
        intial_guess = torch.cat([t,b], dim=1).to(self.device)
        
        return intial_guess
    
    
    def _get_optimizer(self, theta, optimizer_type):
        
        # theta contains w1, w2 (weights) and b (bias), but w1, w2 
        # are given by the same parameter in the polar coordinate
        t = Parameter(theta[0])
        b = Parameter(theta[1])
        params_input = [t,b]
            
        # use optimizer-generator to get optimizer
        optimizer = gen.Generator(params_input, self.params_domain).get_optimizer(optimizer_type)
        
        return (t,b), optimizer
    
    
    def _argmax_optimize_seq(self, pde_energy, theta_list, optimizer_type):
        
        # each theta in the list should contain t and b
        assert theta_list.shape[1] == 2
        
        # get the number of parameter sets 
        num_search = theta_list.shape[0]
        theta_updated = torch.zeros(num_search, 2)
        evaluate_list = torch.zeros(num_search, 1)
        
        # train parameters of each set and evaluate their ultimate energy
        epochs = 10
        for i in range(num_search): 
            print(' training the {:.0f}-th candidate'.format(i+1))
            theta, optimizer = self._get_optimizer(theta_list[i,...], optimizer_type)
            for epoch in range(epochs):
                def closure():
                    optimizer.zero_grad()
                    param = self._polar_to_cartesian(theta)
                    loss = pde_energy(param)
                    loss.backward()
                    # print('loss: {:.16e}, epoch = {:}'.format(loss.item(), epoch))
                    return loss
                optimizer.step(closure)
                # print('loss: {:.10e}, epoch = {:}'.format(closure().item(), epoch))
                new_loss = closure().detach()
            theta_updated[i,0] = theta[0]
            theta_updated[i,1] = theta[1]
            evaluate_list[i] = new_loss
        
        return theta_updated, evaluate_list            
            
    
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
            theta_list: an m-by-n tensor, where 
                        m is the number of parameter set,
                        n is the number of parameters in each set 
                        (n=2, when dim=1 or dim=2, n=3 when dim=3).
            optimizer_type: options for choosing optimizers. Available for
                        "pgd", projected gradient descent (PGD) method,
                        "fista", acceleratd PGD method,
                        "lbfgs", L-BFGS method.
        """
        
        if not self.parallel_search:
            return self._argmax_optimize_seq(pde_energy, theta_list, optimizer_type)
        else:
            return self._argmax_optimize_par(pde_energy, theta_list, optimizer_type)


    def find_optimal_element(self, energy):
        
        # initial guesses 
        start_0 = time.time()
        pde_energy = energy.evaluate_large_scale
        theta_init_guess = self._select_initial_elements(pde_energy, best_k=1)
        end_0 = time.time()
        print('\n Initial guess time = {:.4f}s'.format(end_0 - start_0))

        # process the optimization 
        if self.optimizer:    
            # process the optimization 
            start_1 = time.time()
            optimizer_type = self.optimizer
            pde_energy = energy.evaluate
            print('\n Start optimization:')
            theta_list, evaluate_list = self._argmax_optimize(pde_energy, theta_init_guess, optimizer_type)
            end_1 = time.time()
            print(' optimization time = {:.4f}s'.format(end_1 - start_1))
            
            # find the best element via the list of evaluation
            index = evaluate_list.argmin()
            optimal_element = self._polar_to_cartesian(theta_list[index, ...])
        else:
            optimal_element = self._polar_to_cartesian(theta_init_guess[0, ...].reshape(-1,1))
            
        # total time cost
        end_2 = time.time()
        print('\n Total selection time = {:.4f}s'.format(end_2 - start_0))
        
        # return the parameters of the best element
        return optimal_element

