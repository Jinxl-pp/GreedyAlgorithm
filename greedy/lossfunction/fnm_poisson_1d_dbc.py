import torch
import numpy as np
from . import energy 

class FNM_Poisson_1d_DBC(energy.LinearEnergy): 
    
    def __init__(self, 
                activation,
                quadrature,
                boundary,
                penalty,
                pde,
                device,
                parallel_evaluation=False): 
        """ 
        The discrete energy functional for the Poisson's equation:
                        - u_xx = f, in Omega of R
                             u = g, on boundary of Omega
        with g=0 as the homogeneous Dirichlet's boundary condition.
        Solve this PDE by minimizing the variational loss function:
        J(u) = 
            (1/2)*(nabla_u, nabla_u) + (1/2*mu)*(u-g,u-g) - (f,u),
        with mu being the penalty parameter for boundary condition. 
        The H^1 minimizer u1 := argmin J(v) satisfies:
                ||u-u1||_{1,mu} <= C * sqrt(mu) * ||u||_2
        INPUT: 
            activation: nonlinear activation functions.
            quadrature: full quadrature information.
            boundary: the boundary nodes of 1D problem.
            penalty: the penalty parameter mu.
            pde: a PDE object used in energy evaluation.
            device: cpu or cuda.
            parallel_evaluation: option for evaluation with a large scale.     
        """
        super(FNM_Poisson_1d_DBC, self).__init__()
        
        self.penalty = penalty
        self.boundary = boundary
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
        
        # Dirichlet's trace evaluated on the boundary, with penalty scaling
        obj_val_bdata = obj_func(self.boundary) * (1/self.penalty)
        
        # function value, gradient evaluation on quadpts
        quadpts = self.quadpts.requires_grad_()
        obj_val = obj_func(quadpts)
        obj_val_data = obj_val.detach()
        
        # get gradient evaluated on all quadpts
        f = obj_val.sum()
        gradient = torch.autograd.grad(outputs=f, inputs=quadpts) #, create_graph=True)
        obj_grad_data = gradient[0].detach()
        
        # reset self.quadpts
        self.quadpts = self.quadpts.detach()
        
        return (obj_val_bdata, obj_val_data, obj_grad_data)
    
    
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
        
        # L2 norm
        l2 = items[1].pow(2) * self.weights
        l2_norm = l2.sum() * self.area
        
        # assemble energy norm (not semi-norm)
        energy_norm = 0
        for item in items[1:]:
            h = item.pow(2) * self.weights
            energy_norm += h.sum() * self.area
            
        energy_norm += items[0].pow(2).sum() * (self.penalty)
        
        return (l2_norm, energy_norm)
    
    
    def energy_error(self):
        """
        Numerical errors of pre_solution in square L2 and the energy norm a(u,u)
        """
        return self._energy_norm(self._get_error)
    
    
    def get_stiffmat_and_rhs(self, parameters, core_in, core_bd):
        
        """
        Use energy bilinear form to assemble a stiffmatrix and load vector

        Args:
            parameters: a set of parameters (w,b)
            core_in: a core matrix generated outside this energy class
            core_bd: a core matrix generated outside this energy class
        """
        
        # get components of parameters, in a column
        w = parameters[:,0:1]
        
        # core matrix and vectors used for vecterization
        g_bd = self.sigma(core_bd)
        g_in = self.sigma(core_in)
        dg1 = self.dsigma(core_in, 1)
        dg2 = dg1 * self.weights.t()
        
        # assemble stiffness matrix
        G_bd = torch.mm(g_bd, g_bd.t()) * (1/self.penalty)
        dG = torch.mm(dg1, dg2.t()) * torch.mm(w, w.t()) * self.area
        Gk = dG + G_bd
        
        # assemble load vector
        f = self.source_data * self.weights
        bk = torch.mm(g_in, f) * self.area 
        
        return (Gk, bk)
    
    
    def evaluate(self, param):
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_val_bd = items[0]
        u_val_in = items[1]
        u_grad_in = items[2]
                
        # get components of theta
        w = param[0]
        A = torch.cat([param[0], param[1]], dim=1)
        ones_in = torch.ones(len(self.quadpts),1).to(self.device)
        B_in = torch.cat([self.quadpts, ones_in], dim=1).t()  
        ones_bd = torch.ones(len(self.boundary),1).to(self.device)
        B_bd = torch.cat([self.boundary, ones_bd], dim=1).t()  

        # core matrix and vectors used for vecterization
        core_in = torch.mm(A, B_in)
        core_bd = torch.mm(A, B_bd)
        g_bd = self.sigma(core_bd)
        g_in = self.sigma(core_in)
        dg_in = self.dsigma(core_in, 1)
        fg_in = torch.mm(g_in, self.source_data * self.weights)
        ug_bd = torch.mm(g_bd, u_val_bd)
        dudg_in = torch.mm(dg_in, u_grad_in * self.weights) * w

        # assemble
        energy_eval = -(1/2)*((dudg_in - fg_in) * self.area + ug_bd).pow(2)
            
        return energy_eval
    
    
    def evaluate_large_scale(self, param):
        """ 
        The large scale evaluation of 1D problem.
        """
        return self.evaluate(param)
    
    
    def update_solution(self, pre_solution):
        self.pre_solution = pre_solution