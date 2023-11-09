import torch
import numpy as np
from . import energy 

class FNM_L2fitting_2d(energy.LinearEnergy):
    
    def __init__(self, 
                activation,
                quadrature,
                pde,
                device,
                parallel_evaluation=False): 
        """ 
        The discrete energy functional for the L2-fitting problem:
                        u = f, in Omega of R
                        J(u) = (1/2)*(u-f,u-f)
        INPUT: 
            activation: nonlinear activation functions.
            quadrature: full quadrature information.
            pde: a PDE object used in energy evaluation.
            device: cpu or cuda.
            parallel_evaluation: option for evaluation with a large scale.     
        """
        super(FNM_L2fitting_2d, self).__init__()
        
        self.sigma = activation.activate
        self.dsigma = activation.dactivate
        self.quadpts = quadrature.quadpts
        self.weights = quadrature.weights
        self.area = quadrature.area
        self.source_data = pde.solution(self.quadpts)
        self.pde = pde
        
        self.device = device
        """
            pre_solution: The evaluation of target function at pre_solution 
                          determines the parameters of the next neuron. The 
                          pre_solution is initialized by zero function.
        """ 
        self.pre_solution = self._zero
        self.parallel_evaluation = parallel_evaluation
    
        
    def _get_energy_items(self, obj_func):
        
        # gradient evaluation on quadpts
        obj_val = obj_func(self.quadpts)
        obj_val_data = obj_val.detach()
        
        # reset self.quadpts
        self.quadpts = self.quadpts.detach()
        
        return obj_val_data
    
    
    def _get_error(self, p):
        return self.pde.solution(p) - self.pre_solution(p)
    

    def _partition(self, num_param, n):
        division = len(num_param) / n
        return [0]+[round(division * (i + 1)) for i in range(n)]
    

    def _zero(self, p):
        return 0. * p[...,0:1]
    
    
    def _energy_norm(self, obj_func): 
        
        """
        Get measure in square L2 norm
        """
        
        # get bilinear forms 
        items = self._get_energy_items(obj_func)
        
        # default L2 norm
        l2 = items.pow(2) * self.weights
        l2_norm = l2.sum() * self.area
        
        # assemble energy norm (not semi-norm)
        energy_norm = l2_norm
        
        return energy_norm
    
    
    def energy_error(self):
        """
        Numerical errors of pre_solution in square L2 norm
        """
        return self._energy_norm(self._get_error)
    
    
    def get_stiffmat_and_rhs(self, parameters, core):
        
        """
        Use energy bilinear form to assemble a stiffmatrix and load vector

        Args:
            parameters: a set of parameters (w,b)
            core: a core matrix generated outside this energy class
        """
        
        # core matrix and vectors used for vecterization
        g1 = self.sigma(core)
        g2 = g1 * self.weights.t()
        
        # assemble stiffness matrix
        Gk = torch.mm(g1, g2.t()) * self.area
        
        # assemble load vector
        f = self.source_data * self.weights
        bk = torch.mm(g1, f) * self.area 
        
        return (Gk, bk)
    
    
    def evaluate(self, param):
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_val = items
                
        # get components of theta
        A = torch.cat([param[0], param[1], param[2]], dim=1)
        ones = torch.ones(len(self.quadpts),1).to(self.device)
        B = torch.cat([self.quadpts, ones], dim=1).t()  
        
        # core matrix and vectors used for vecterization
        core = torch.mm(A, B)
        g = self.sigma(core)
        fg = torch.mm(g, self.source_data * self.weights)
        ug = torch.mm(g, u_val * self.weights)
        
        # assemble
        energy_eval = -(1/2)*((ug-fg) * self.area).pow(2)
            
        return energy_eval
    
    
    def evaluate_large_scale(self, param):
        """
        The large scale evaluation of 2D problem, for evaluation
        in the multiple-parameter situation.
        """
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_val = items 
                
        # get components of theta
        A = torch.cat([param[0], param[1], param[2]], dim=1)
        ones = torch.ones(len(self.quadpts),1).to(self.device)
        B = torch.cat([self.quadpts, ones], dim=1).t()  
        
        # core matrix and vectors used for vecterization
        num_param = A.shape[0]
        ug = torch.zeros(num_param,1)
        fg = torch.zeros(num_param,1)     
        
        num_batch = 2
        division = self._partition(range(num_param),num_batch)
        for i in range(num_batch):
            core = torch.mm(A[division[i]:division[i+1],:], B)
            g = self.sigma(core)
            del core
            ug[division[i]:division[i+1]] = torch.mm(g, u_val * self.weights)
            fg[division[i]:division[i+1]] = torch.mm(g, self.source_data * self.weights)
            del g
        
        # assemble
        energy_eval = -(1/2)*((ug-fg) * self.area).pow(2)   
            
        return energy_eval
    
    
    def update_solution(self, pre_solution):
        self.pre_solution = pre_solution