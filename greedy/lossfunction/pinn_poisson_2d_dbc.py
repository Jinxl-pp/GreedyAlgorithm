import torch
import numpy as np
from . import energy 

class PINN_Poisson_2d_DBC(energy.LinearEnergy): 
    
    def __init__(self, 
                activation,
                quadrature_in,
                quadrature_bd,
                pde,
                device,
                penalty=1,
                parallel_evaluation=False): 
        
        """ 
        The discrete energy functional for the second order elliptic PDE:
                        -Lap(u) = f, in Omega of R
                              u = gD, on boundary of Omega
        with gD=0 as the homogeneous Neumann's boundary condition.
        Solve the equation by minimizing an energy functional with 
        strong form of this PDE.
            J(u) = ||Lap(u) + f||^2 + mu*||u-gD||^2,
        with mu being the penalty parameter for the boundary condition. 
        INPUT: 
            activation: nonlinear activation functions.
            quadrature_in: full quadrature information of interior domain.
            quadrature_bd: full quadrature information of boundary.
            pde: a PDE object used in energy evaluation.
            device: cpu or cuda.
            penalty: the parameter in front of boundary terms.
            parallel_evaluation: option for evaluation with a large scale.     
        """
        super(PINN_Poisson_2d_DBC, self).__init__()
        
        self.sigma = activation.activate
        self.dsigma = activation.dactivate
        self.quadpts_in = quadrature_in.quadpts
        self.quadpts_bd = quadrature_bd.quadpts
        self.weights_in = quadrature_in.weights
        self.weights_bd = quadrature_bd.weights
        self.area_in = quadrature_in.area
        self.area_bd = quadrature_bd.area
        self.source_data = pde.source(self.quadpts_in)
        self.trace = pde.solution(self.quadpts_bd)
        self.pde = pde
        self.penalty = penalty
        
        self.device = device
        self.qtype = quadrature_in.qtype
        """
            pre_solution: The evaluation of target function at pre_solution 
                          determines the parameters of the next neuron. The 
                          pre_solution is initialized by zero function.
        """ 
        self.pre_solution = self._zero
        self.parallel_evaluation = parallel_evaluation
        
        
    def _get_energy_items(self, obj_func):
        
        # initialized enery-item
        items = []
        
        # evaluation on boundary
        obj_val_bdata = obj_func(self.quadpts_bd)
        items.append(obj_val_bdata.detach())
        
        # evaluation on domain quadpts
        xpts = self.quadpts_in[...,0:1].requires_grad_()
        ypts = self.quadpts_in[...,1:2].requires_grad_()
        quadpts = torch.cat([xpts,ypts],dim=1)
        obj_val = obj_func(quadpts)
        items.append(obj_val.detach())
        
        # get gradient evaluated on domain quadpts
        f = obj_val.sum()
        gradient_x = torch.autograd.grad(outputs=f, inputs=xpts, create_graph=True)
        gradient_y = torch.autograd.grad(outputs=f, inputs=ypts, create_graph=True)
        
        # get Laplacian evaluated on domain quadpts
        dxf = gradient_x[0].sum()
        dyf = gradient_y[0].sum()
        hessian_xx = torch.autograd.grad(outputs=dxf, inputs=xpts, create_graph=True)
        hessian_yy = torch.autograd.grad(outputs=dyf, inputs=ypts)
        items.append(hessian_xx[0].detach())
        items.append(hessian_yy[0].detach())
        
        # clear autograd graph
        self.quadpts_in = self.quadpts_in.detach()
        self.quadpts_bd = self.quadpts_bd.detach()
        
        return items
    

    def _get_error(self, p):
        return self.pde.solution(p) - self.pre_solution(p)
    
    
    def _partition(self, num_param, n):
        division = len(num_param) / n
        return [0]+[round(division * (i + 1)) for i in range(n)]
    
    
    def _energy_norm(self, obj_func): 
        
        """
        Get measure in square L2 and H^2 norm
        """
        
        # get bilinear forms 
        items = self._get_energy_items(obj_func)
        
        # default L2 norm
        l2 = items[1].pow(2) * self.weights_in
        l2_norm = l2.sum() * self.area_in
        
        # assemble energy norm (not semi-norm)
        energy_norm = 0
        for item in items[1:]:
            h = item.pow(2) * self.weights_in
            energy_norm += h.sum() * self.area_in
        
        return (l2_norm, energy_norm)
    
    
    def _zero(self, p):
        x = p[...,0:1]
        y = p[...,1:2]
        return 0. * torch.sin(x) * torch.sin(y)
    
    
    def energy_error(self):
        """
        Numerical errors of pre_solution in square L2 and incomplete H^2 norm
        """
        return self._energy_norm(self._get_error)
    
    
    def loss(self):
        """
        Loss function evaluated at the current solution
        """
        
        # get bilinear forms
        items = self._get_energy_items(self.pre_solution)
        
        # interior residual
        u_lap = items[2] + items[3]
        residual_in = (u_lap + self.source_data).pow(2) * self.weights_in
        residual_in = residual_in.sum() * self.area_in
        
        # boundary residual
        u_trace = items[0]
        residual_bd = (u_trace - self.trace).pow(2) * self.weights_bd
        residual_bd = residual_bd.sum() * self.area_bd * self.penalty
        
        return residual_in + residual_bd
    
    
    def get_stiffmat_and_rhs(self, parameters, core_in, core_bd):
        
        """
        Use energy bilinear form to assemble a stiffmatrix and load vector

        Args:
            parameters: a set of parameters (w,b)
            core: a core matrix generated outside this energy class
        """
        
        # get components of parameters, in a column
        w1 = parameters[:,0:1]
        w2 = parameters[:,1:2]
        v11 = w1 * w1
        v22 = w2 * w2
        
        # core matrix and vectors used for vecterization
        g1_bd = self.sigma(core_bd)
        g2_bd = g1_bd * self.weights_bd.t()
        d2g1_in = self.dsigma(core_in, 2)
        d2g2_in = d2g1_in * self.weights_in.t()
        
        # assemble stiffness matrix
        coef = torch.mm(v11, v11.t()) + torch.mm(v22, v22.t()) + \
                torch.mm(v11, v22.t()) + torch.mm(v22, v11.t())
        d2G = torch.mm(d2g1_in, d2g2_in.t()) * coef * self.area_in
        G_bd = torch.mm(g1_bd, g2_bd.t()) * self.area_bd * self.penalty
        Gk = d2G + G_bd 
        
        # assemble load vector
        gD = self.trace * self.weights_bd 
        f = self.source_data * self.weights_in
        bk = -torch.mm(d2g1_in, f) * (v11+v22) * self.area_in + \
            torch.mm(g1_bd, gD) * self.area_bd * self.penalty
        
        return (Gk, bk)
    
    
    def evaluate(self, param):
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_trace = items[0]
        u_lap = items[2] + items[3]
         
        # get components of theta
        w1 = param[0]
        w2 = param[1]
        A = torch.cat([param[0], param[1], param[2]], dim=1)
        ones_in = torch.ones(len(self.quadpts_in),1).to(self.device)
        B_in = torch.cat([self.quadpts_in, ones_in], dim=1).t()  
        ones_bd = torch.ones(len(self.quadpts_bd),1).to(self.device)
        B_bd = torch.cat([self.quadpts_bd, ones_bd], dim=1).t()  
        
        # core matrix and vectors used for vecterization
        core_in = torch.mm(A, B_in) 
        core_bd = torch.mm(A, B_bd)
        g_bd = self.sigma(core_bd)
        d2g_in = self.dsigma(core_in, 2)
        residual_in = torch.mm(d2g_in, (u_lap+self.source_data)*self.weights_in) * (w1*w1 + w2*w2)
        residual_bd = torch.mm(g_bd, (u_trace-self.trace)*self.weights_bd) * self.penalty
        
        # assemble
        energy_eval = -(1/2)*(residual_in*self.area_in + residual_bd*self.area_bd).pow(2)
            
        return energy_eval
    
    
    def evaluate_large_scale(self, param):
        
        if self.qtype == "MC":
            return self.evaluate(param)

        else:
            # get items of the energy bilinear form, denote pre_solution := u
            items = self._get_energy_items(self.pre_solution)
            u_trace = items[0]
            u_lap = items[2] + items[3]
            
            # get components of theta
            w1 = param[0]
            w2 = param[1]
            v11 = w1 * w1
            v22 = w2 * w2
            A = torch.cat([param[0], param[1]], dim=1)
            ones_in = torch.ones(len(self.quadpts_in),1).to(self.device)
            B_in = torch.cat([self.quadpts_in, ones_in], dim=1).t()  
            ones_bd = torch.ones(len(self.quadpts_bd),1).to(self.device)
            B_bd = torch.cat([self.quadpts_bd, ones_bd], dim=1).t()  
            
            # core matrix and vectors used for vecterization
            num_param = A.shape[0]
            residual_in = torch.zeros(num_param,1)
            residual_bd = torch.zeros(num_param,1)
            
            num_batch = 2
            division = self._partition(range(num_param),num_batch)
            for i in range(num_batch):
                core_in = torch.mm(A[division[i]:division[i+1],:], B_in)
                core_bd = torch.mm(A[division[i]:division[i+1],:], B_bd)
                g_bd = self.sigma(core_bd)
                d2g_in = self.dsigma(core_in, 2)
                residual_in[division[i]:division[i+1]] = torch.mm(d2g_in, (u_lap+self.source_data)*self.weights_in) * \
                        (v11[division[i]:division[i+1]] + v22[division[i]:division[i+1]])
                residual_bd[division[i]:division[i+1]] = torch.mm(g_bd, (u_trace-self.trace)*self.weights_bd) * self.penalty
            
            # assemble
            energy_eval = -(1/2)*(residual_in*self.area_in + residual_bd*self.area_bd).pow(2)
                
            return energy_eval
    
    
    def update_solution(self, pre_solution):
        self.pre_solution = pre_solution