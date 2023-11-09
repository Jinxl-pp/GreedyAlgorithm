import torch
import numpy as np
from . import energy 

class FNM_Elliptic_2nd_1d_NBC(energy.LinearEnergy): 
    
    def __init__(self, 
                activation,
                quadrature,
                pde,
                device,
                parallel_evaluation=False): 
    
        """ 
        The discrete energy functional for the second order elliptic PDE:
                        - u_xx + u = f, in Omega of R
                             du/dx = g, on boundary of Omega
        with g=0 as the homogeneous Neumann's boundary condition.
        Solve the equation by minimizing an energy functional with 
        weak form of this PDE.
            J(u) = (1/2)*(u_x,u_x) + (1/2)*(u,u) - (f,u) - (g,u).
        INPUT: 
            activation: nonlinear activation functions.
            quadrature: full quadrature information.
            pde: a PDE object used in energy evaluation.
            device: cpu or cuda.
            parallel_evaluation: option for evaluation with a large scale.     
        """
        super(FNM_Elliptic_2nd_1d_NBC, self).__init__()
        
        self.sigma = activation.activate
        self.dsigma = activation.dactivate
        self.quadpts = quadrature.quadpts
        self.weights = quadrature.weights
        self.area = quadrature.area
        self.source_data = pde.source(self.quadpts)
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
        quadpts = self.quadpts.requires_grad_()
        obj_val = obj_func(quadpts)
        obj_val_data = obj_val.detach()
        
        # get gradient evaluated on all quadpts
        f = obj_val.sum()
        gradient = torch.autograd.grad(outputs=f, inputs=quadpts) #, create_graph=True)
        obj_grad_data = gradient[0].detach()
        
        # reset self.quadpts
        self.quadpts = self.quadpts.detach()
        
        return (obj_val_data, obj_grad_data)
    
    
    def _get_error(self, p):
        return self.pde.solution(p) - self.pre_solution(p)
    
    
    def _zero(self, p):
        return 0. * p
    
    
    def _energy_norm(self, obj_func): 
        
        """
        Get measure in square L2 and the energy norm a(u,u)
        """
        
        # get bilinear forms 
        items = self._get_energy_items(obj_func)
        
        # default L2 norm
        l2 = items[0].pow(2) * self.weights
        l2_norm = l2.sum() * self.area
        
        # assemble energy norm (not semi-norm)
        energy_norm = 0
        for item in items:
            h = item.pow(2) * self.weights
            energy_norm += h.sum() * self.area
        
        return (l2_norm, energy_norm)
    
    
    def energy_error(self):
        """
        Numerical errors of pre_solution in square L2 and the energy norm a(u,u)
        """
        return self._energy_norm(self._get_error)
        
    
    def get_stiffmat_and_rhs(self, parameters, core):
        
        """
        Use energy bilinear form to assemble a stiffmatrix and load vector

        Args:
            parameters: a set of parameters (w,b)
            core: a core matrix generated outside this energy class
        """
        
        # get components of parameters, in a column
        w = parameters[:,0:1]
        
        # core matrix and vectors used for vecterization
        g1 = self.sigma(core)
        dg1 = self.dsigma(core, 1)
        g2 = g1 * self.weights.t()
        dg2 = dg1 * self.weights.t()
        
        # assemble stiffness matrix
        G = torch.mm(g1, g2.t()) * self.area
        dG = torch.mm(dg1, dg2.t()) * torch.mm(w, w.t()) * self.area
        Gk = dG + G
        
        # assemble load vector
        f = self.source_data * self.weights
        bk = torch.mm(g1, f) * self.area 
        
        return (Gk, bk)
    
    
    def evaluate(self, param):
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_val = items[0]
        u_grad = items[1]
                
        # get components of theta
        w = param[0]
        A = torch.cat([param[0], param[1]], dim=1)
        ones = torch.ones(len(self.quadpts),1).to(self.device)
        B = torch.cat([self.quadpts, ones], dim=1).t()  
        
        # core matrix and vectors used for vecterization
        core = torch.mm(A, B)
        g = self.sigma(core)
        dg = self.dsigma(core, 1)
        fg = torch.mm(g, self.source_data * self.weights)
        ug = torch.mm(g, u_val * self.weights)
        dudg = torch.mm(dg, u_grad * self.weights) * w
        
        # assemble
        energy_eval = -(1/2)*((dudg+ug-fg) * self.area).pow(2)
            
        return energy_eval
    
    
    def evaluate_large_scale(self, param):
        """ 
        The large scale evaluation of 1D problem.
        """
        return self.evaluate(param)
    
    
    def update_solution(self, pre_solution):
        self.pre_solution = pre_solution