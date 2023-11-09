import torch
import numpy as np
from . import energy 

class PINN_Elliptic_2nd_1d_NBC(energy.LinearEnergy): 
    
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
                        - u_xx + u = f, in Omega of R
                             du/dx = g, on boundary of Omega
        with g=0 as the homogeneous Neumann's boundary condition.
        Solve the equation by minimizing an energy functional with 
        strong form of this PDE.
            J(u) = ||-u_xx + u - f||^2 + mu*||du/dx-g||^2,
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
        super(PINN_Elliptic_2nd_1d_NBC, self).__init__()
        
        self.sigma = activation.activate
        self.dsigma = activation.dactivate
        self.quadpts = quadrature.quadpts
        self.weights = quadrature.weights
        self.area = quadrature.area
        self.boundary = boundary
        self.source_data = pde.source(self.quadpts)
        self.trace = pde.trace(self.boundary)
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
        boundary = self.boundary.requires_grad_()
        obj_val_bd = obj_func(boundary)
        f_bd = obj_val_bd.sum()
        n_trace = torch.autograd.grad(outputs=f_bd, inputs=boundary)
        obj_grad_bdata = n_trace[0].detach()
        
        # clear autograd graph
        self.quadpts = self.quadpts.detach()
        self.boundary = self.boundary.detach()
        
        return (obj_grad_bdata, obj_val_data, obj_grad_data, obj_hess_data)
    
    
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
        val = items[1]
        hess = items[3]
        residual_in = (-hess + val - self.source_data).pow(2) * self.weights
        residual_in = residual_in.sum() * self.area
        
        # boundary residual
        trace = items[0]
        residual_bd = (trace - self.trace).pow(2).sum() * self.penalty
        
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
        I = torch.ones(w.shape)
        v = w * w
        
        # core matrix and vectors used for vecterization
        g1 = self.sigma(core_in)
        d2g1 = self.dsigma(core_in, 2)
        g2 = g1 * self.weights.t()
        d2g2 = d2g1 * self.weights.t()
        dg_bd = self.dsigma(core_bd, 1)
        
        # assemble stiffness matrix
        G1 = torch.mm(g1, g2.t()) * self.area
        G2 = torch.mm(g1, d2g2.t()) * torch.mm(I, v.t()) * self.area
        G3 = torch.mm(d2g1, g2.t()) * torch.mm(v, I.t()) * self.area
        G4 = torch.mm(d2g1, d2g2.t()) * torch.mm(v, v.t()) * self.area
        G5 = torch.mm(dg_bd, dg_bd.t()) * torch.mm(w, w.t()) * self.penalty
        Gk = G1 - G2 - G3 + G4 + G5
        
        # assemble load vector
        gN = self.trace
        f = self.source_data * self.weights
        bk = torch.mm(-d2g1, f) * v * self.area + \
            torch.mm(g1, f) * self.area + torch.mm(dg_bd, gN) * self.penalty
         
        return (Gk, bk)
        
        
    def evaluate(self, param):
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_trace = items[0]
        u_val = items[1]
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
        g_in = self.sigma(core_in)
        dg_bd = self.dsigma(core_bd, 1)
        d2g_in = self.dsigma(core_in, 2)
        fg = torch.mm(g_in, self.source_data * self.weights) - \
                torch.mm(d2g_in, self.source_data * self.weights) * w * w
        ug = torch.mm(g_in, (-u_hess + u_val) * self.weights) - \
             torch.mm(d2g_in, (-u_hess + u_val) * self.weights) * w * w
        ug_bd = torch.mm(dg_bd, u_trace) * w * self.penalty
        
        # assemble
        energy_eval = -(1/2)*((ug-fg) * self.area + ug_bd).pow(2)
            
        return energy_eval
    
    
    def evaluate_large_scale(self, param):
        """ 
        The large scale evaluation of 1D problem.
        """
        return self.evaluate(param)
    
    
    def update_solution(self, pre_solution):
        self.pre_solution = pre_solution
        
        
        
        
        
    
    