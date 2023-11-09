import torch
import numpy as np
from . import energy 

class PINN_Poisson_1d_DBC(energy.LinearEnergy): 
    
    def __init__(self, 
                activation,
                quadrature,
                boundary,
                pde,
                device,
                penalty=1,
                parallel_evaluation=False): 
        
        """ 
        The discrete energy functional for the second order elliptic PDE:
                        - u_xx = f, in Omega of R
                             u = g, on boundary of Omega
        with g=0 as the homogeneous Neumann's boundary condition.
        Solve the equation by minimizing an energy functional with 
        strong form of this PDE.
            J(u) = ||u_xx + f||^2 + mu*||u-g||^2,
        with mu being the penalty parameter for the boundary condition. 
        INPUT: 
            activation: nonlinear activation functions.
            quadrature: full quadrature information.
            boundary: the boundary nodes of 1D problem.
            pde: a PDE object used in energy evaluation.
            device: cpu or cuda.
            penalty: the parameter in front of boundary terms.
            parallel_evaluation: option for evaluation with a large scale.     
        """
        super(PINN_Poisson_1d_DBC, self).__init__()
        
        self.sigma = activation.activate
        self.dsigma = activation.dactivate
        self.quadpts = quadrature.quadpts
        self.weights = quadrature.weights
        self.area = quadrature.area
        self.boundary = boundary
        self.source_data = pde.source(self.quadpts)
        self.trace = pde.solution(self.boundary)
        self.pde = pde
        self.penalty = penalty
        
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
        gradient = torch.autograd.grad(outputs=f, inputs=quadpts, create_graph=True)
        obj_grad_data = gradient[0].detach()

        # get second-order derivatives evaluated on all quadpts
        df = gradient[0].sum()
        hessian = torch.autograd.grad(outputs=df, inputs=quadpts)
        obj_hess_data = hessian[0].detach()
        
        # gradient evaluation on boundary nodes
        obj_val_bd = obj_func(self.boundary).detach()
        
        # clear autograd graph
        self.quadpts = self.quadpts.detach()
        self.boundary = self.boundary.detach()
        
        return (obj_val_bd, obj_val_data, obj_grad_data, obj_hess_data)
    
    
    def _get_error(self, p):
        return self.pde.solution(p) - self.pre_solution(p)
    
    
    def _energy_norm(self, obj_func): 
        
        """
        Get measure in square L2 and H^2 norm
        """
        
        # get bilinear forms 
        items = self._get_energy_items(obj_func)
        
        # default L2 norm
        l2 = items[1].pow(2) * self.weights
        l2_norm = l2.sum() * self.area
        
        # assemble energy norm (not semi-norm)
        energy_norm = 0
        for item in items[1:]:
            h = item.pow(2) * self.weights
            energy_norm += h.sum() * self.area
        
        return (l2_norm, energy_norm)
    
    
    def _zero(self, p):
        return 0. * torch.sin(p)
    
    
    def energy_error(self):
        """
        Numerical errors of pre_solution in square L2 and H^2 norm
        """
        return self._energy_norm(self._get_error)
    
    
    def loss(self):
        """
        Loss function evaluated at the current solution
        """
        
        # get bilinear forms
        items = self._get_energy_items(self.pre_solution)
        
        # inner residual
        u_hess = items[3]
        residual_in = (u_hess + self.source_data).pow(2) * self.weights
        residual_in = residual_in.sum() * self.area
        
        # boundary residual
        u_trace = items[0]
        residual_bd = (u_trace - self.trace).pow(2).sum() * self.penalty
        
        return residual_in + residual_in
    
    
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
        v = w * w
        
        # core matrix and vectors used for vecterization
        d2g1 = self.dsigma(core_in, 2)
        d2g2 = d2g1 * self.weights.t()
        g_bd = self.sigma(core_bd)
        
        # assemble stiffness matrix
        d2G = torch.mm(d2g1, d2g2.t()) * torch.mm(v, v.t()) * self.area
        G_bd = torch.mm(g_bd, g_bd.t()) * self.penalty
        Gk = d2G + G_bd
        
        # assemble load vector
        gD = self.trace
        f = self.source_data * self.weights
        bk = - torch.mm(d2g1, f) * v * self.area + \
            torch.mm(g_bd, gD) * self.penalty
         
        return (Gk, bk)
        
        
    def evaluate(self, param):
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_trace = items[0]
        u_hess = items[3]
         
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
        g_bd= self.sigma(core_bd)
        d2g_in = self.dsigma(core_in, 2)
        residual_in = torch.mm(d2g_in, (u_hess+self.source_data) * self.weights) * w * w
        residual_bd = torch.mm(g_bd, u_trace - self.trace) * self.penalty
        
        # assemble
        energy_eval = -(1/2)*(residual_in * self.area + residual_bd).pow(2)
            
        return energy_eval
    
    
    def evaluate_large_scale(self, param):
        """ 
        The large scale evaluation of 1D problem.
        """
        return self.evaluate(param)
    
    
    def update_solution(self, pre_solution):
        self.pre_solution = pre_solution
        
        
        
        
        
    
    