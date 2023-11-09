import torch
import numpy as np
from . import energy 

class PINN_AllenCahn_2d_DBC(energy.NonlinearEnergy): 
    
    def __init__(self, 
                activation,
                quadrature_in,
                quadrature_bd,
                quadrature_init,
                pde,
                device,
                penalty_bd=1,
                penalty_init=1,
                parallel_evaluation=False): 
        
        """ 
        The discrete energy functional for the Allen-Cahn's equation:
                    ut - l*u_xx + e*(u^3-u) = 0, Omega*(0,T]
                    u(x,t) = gD, dOmega*(0,T]
                    u(x,0) = u0(x), Omega
        with the Dirichlet type of boundary condition and an initial 
        condition. Solve the equation by minimizing an energy functional  
        with strong form of this PDE.
            J(u) = || ut - l*u_xx + e*(u^3-u) ||^2 
                    + mu1*||u-gD||^2
                    + mu2*||u-u0||^2
        with mu1 and mu2 being the penalty parameters for the boundary 
        and initial conditions. 
        INPUT: 
            activation: nonlinear activation functions.
            quadrature_in: full quadrature information of interior domain.
            quadrature_bd: full quadrature information of spacial boundary.
            quadrature_init: full quadrature information of time boundary.
            pde: a PDE object used in energy evaluation.
            device: cpu or cuda.
            penalty_bd: the parameter in front of boundary condition.
            penalty_init: the parameter in front of temporal condition.
            parallel_evaluation: option for evaluation with a large scale.     
        """
        
        super(PINN_AllenCahn_2d_DBC, self).__init__()
        
        self.sigma = activation.activate
        self.dsigma = activation.dactivate
        
        """
            All training datasets:
        """
        self.quadpts_in = quadrature_in.quadpts
        self.quadpts_bd = quadrature_bd.quadpts
        self.quadpts_init = quadrature_init.quadpts
        self.weights_in = quadrature_in.weights
        self.weights_bd = quadrature_bd.weights
        self.weights_init = quadrature_init.weights
        self.area_in = quadrature_in.area
        self.area_bd = quadrature_bd.area
        self.area_init = quadrature_init.area
        
        self.source_data = pde.source(self.quadpts_in)
        self.trace_bd = pde.dirichlet(self.quadpts_bd)
        self.trace_init = pde.dirichlet(self.quadpts_init)
        self.lam = pde.lam
        self.eps = pde.eps
        self.penalty_bd = penalty_bd
        self.penalty_init = penalty_init
        
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
        
        # evaluation on spacial boundary
        obj_val_bdata = obj_func(self.quadpts_bd)
        items.append(obj_val_bdata.detach())
        
        # evaluation on temporal boundary
        obj_val_idata = obj_func(self.quadpts_init)
        items.append(obj_val_idata.detach())
        
        # evaluation on domain quadpts
        xpts = self.quadpts_in[...,0:1].requires_grad_()
        tpts = self.quadpts_in[...,1:2].requires_grad_()
        quadpts = torch.cat([xpts,tpts],dim=1)
        obj_val = obj_func(quadpts)
        items.append(obj_val.detach())
        
        # get gradient evaluated on domain quadpts
        f = obj_val.sum()
        gradient_x = torch.autograd.grad(outputs=f, inputs=xpts, create_graph=True)
        gradient_t = torch.autograd.grad(outputs=f, inputs=tpts, create_graph=True)
        items.append(gradient_t[0].detach())
        
        # get u_xx evaluated on domain quadpts
        dxf = gradient_x[0].sum()
        hessian_xx = torch.autograd.grad(outputs=dxf, inputs=xpts)
        items.append(hessian_xx[0].detach())
        
        # clear autograd graph
        self.quadpts_in = self.quadpts_in.detach()
        self.quadpts_bd = self.quadpts_bd.detach()
        self.quadpts_init = self.quadpts_init.detach()
        
        return items
    
    
    def _get_error(self, p):
        return self.pde.solution(p) - self.pre_solution(p)
    
    
    def _partition(self, num_param, n):
        division = len(num_param) / n
        return [0]+[round(division * (i + 1)) for i in range(n)]
    
    
    def _energy_norm(self, obj_func): 
        
        """
        Get measure in square L2 norm
        """
        
        # get bilinear forms 
        items = self._get_energy_items(obj_func)
        
        # default L2 norm
        l2 = items[1].pow(2) * self.weights_in
        l2_norm = l2.sum() * self.area_in
        
        return l2_norm
    
    
    def _zero(self, p):
        x = p[...,0:1]
        t = p[...,1:2]
        return 0. * torch.sin(x) * torch.sin(t)
    
    
    def energy_error(self):
        """
        Numerical errors of pre_solution in square L2 norm
        """
        return self._energy_norm(self._get_error)
    

    def loss(self):
        """
        Loss function evaluated at the current solution
        """
        
        # get bilinear forms
        items = self._get_energy_items(self.pre_solution)
        
        # interior residual
        u_val = items[2]
        u_t_val = items[3]
        u_xx_val = items[4]
        residual_in = u_t_val - self.lam*u_xx_val + self.eps*u_val.pow(3) - self.eps*u_val
        residual_in = residual_in.pow(2) * self.weights_in
        residual_in = residual_in.sum() * self.area_in
        
        # spacial boundary residual
        u_trace_bd = items[0]
        residual_bd = (u_trace_bd - self.trace_bd).pow(2) * self.weights_bd
        residual_bd = residual_bd.sum() * self.area_bd * self.penalty_bd
        
        # temporal boundary residual
        u_trace_init = items[1]
        residual_init = (u_trace_init - self.trace_init).pow(2) * self.weights_init
        residual_init = residual_init.sum() * self.area_init * self.penalty_init
        
        return residual_in + residual_bd + residual_init    


    def get_nonlinear_system(self, cur_solution, core_in, core_bd, core_init):
        """
        Construct the nonlinear operator and its Jacobian

        Args:
            cur_solution: the current solution u_n, in Newton's iteration
            core_in: a core matrix generated outside this energy class
            core_bd: a core matrix generated outside this energy class
            core_init: a core matrix generated outside this energy class
        """
        
        # evaluation of cur_solution
        items = self._get_energy_items(cur_solution)
        cur_val_bd = items[0]
        cur_val_init = items[1]
        cur_val = items[2]
        cur_t_val = items[3]
        cur_xx_val = items[4]
        
        # get components of parameters, in a column
        wx = cur_solution.layer1.weights[:,0:1]
        wt = cur_solution.layer1.weights[:,1:2]
        
        # core matrix and vectors used for vecterization
        g_in = self.sigma(core_in)
        dg_in = self.dsigma(core_in,1)
        d2g_in = self.dsigma(core_in,2)
        g_bd = self.sigma(core_bd)
        g_init = self.sigma(core_init)
        
        # assemble vertical vector: the nonlinear operator
        gD = self.trace_bd * self.weights_bd
        u0 = self.trace_init * self.weights_init 
        linear = (cur_t_val - self.lam*cur_xx_val - self.eps*cur_val) * self.weights_in
        nlinear_1 = self.eps * cur_val.pow(3) * self.weights_in
        nlinear_2 = self.eps * cur_val.pow(2) * 3
        G = (torch.mm(dg_in, linear+nlinear_1) * wt - \
            self.lam * torch.mm(d2g_in, linear+nlinear_1) * wx * wx - \
            self.eps * torch.mm(g_in, linear+nlinear_1)) * self.area_in
        G += torch.mm(g_in, nlinear_2*(linear+nlinear_1)) * self.area_in
        G += torch.mm(g_bd, cur_val_bd - gD) * self.area_bd * self.penalty_bd
        G += torch.mm(g_init, cur_val_init - u0) * self.area_init * self.penalty_init

        # assemble matrix: the Jacobian 
        g_linear = dg_in*wt - self.lam*d2g_in*wx*wx - self.eps*g_in
        g_nlinear_1 = 3 * self.eps * g_in * cur_val.pow(2).t()
        g_nlinear_2 = 6 * self.eps**2 * g_in * cur_val.pow(4).t()
        J = torch.mm(g_linear, g_linear.t()*self.weights_in) * self.area_in
        J += torch.mm(g_nlinear_1, g_linear.t()*self.weights_in) * self.area_in
        J += torch.mm(g_nlinear_1, g_nlinear_1.t()*self.weights_in) * self.area_in
        J += torch.mm(g_nlinear_2, g_in.t()*self.weights_in) * self.area_in
        J += torch.mm(g_bd, g_bd.t()*self.weights_bd) * self.area_bd
        J += torch.mm(g_init, g_init.t()*self.weights_init) * self.area_init
        
        return (J, G.t())
    
    
    def evaluate(self, param):
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_trace_bd = items[0]
        u_trace_init = items[1]
        u_val = items[2]
        u_t_val = items[3]
        u_xx_val = items[4]
        
        # get components of theta
        wx = param[0]
        wt = param[1]
        A = torch.cat([param[0], param[1], param[2]], dim=1)
        ones_in = torch.ones(len(self.quadpts_in),1).to(self.device)
        B_in = torch.cat([self.quadpts_in, ones_in], dim=1).t()  
        ones_bd = torch.ones(len(self.quadpts_bd),1).to(self.device)
        B_bd = torch.cat([self.quadpts_bd, ones_bd], dim=1).t()  
        ones_init = torch.ones(len(self.quadpts_init),1).to(self.device)
        B_init = torch.cat([self.quadpts_init, ones_init], dim=1).t()
        
        # core matrix and vectors used for vecterization
        core_in = torch.mm(A, B_in) 
        core_bd = torch.mm(A, B_bd)  
        core_init = torch.mm(A, B_init)
        g_in = self.sigma(core_in)
        dg_in = self.dsigma(core_in, 1)
        d2g_in = self.dsigma(core_in, 2)
        g_bd = self.sigma(core_bd)
        g_init = self.sigma(core_init)
        
        # residuals
        u_linear = u_t_val - self.lam*u_xx_val - self.eps*u_val
        u_nlinear = self.eps * u_val.pow(3)
        g_linear = dg_in*wt - self.lam*d2g_in*wx*wx - self.eps*g_in
        g_nlinear = self.eps * g_in * u_val.pow(2).t() * 3
        residual_in = torch.mm(g_linear+g_nlinear, (u_linear+u_nlinear)*self.weights_in)
        residual_bd = torch.mm(g_bd, (u_trace_bd-self.trace_bd)*self.weights_bd) * self.penalty_bd
        residual_init = torch.mm(g_init, (u_trace_init-self.trace_init)*self.weights_init) * self.penalty_init
        
        # assemble
        energy_eval = -(1/2)*(residual_in*self.area_in + residual_bd*self.area_bd + \
                            residual_init*self.area_init).pow(2)
        
        return energy_eval
    
    
    def update_solution(self, solution):
        self.pre_solution = solution