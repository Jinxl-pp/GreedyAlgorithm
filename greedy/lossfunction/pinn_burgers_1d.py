import torch
import numpy as np
from . import energy 

class PINN_Burgers_1d_PBC(energy.NonlinearEnergy): 
    
    def __init__(self, 
            activation,
            quadrature,
            boundary,
            time_step,
            pde,
            device,
            penalty_bd=1,
            parallel_evaluation=False): 
        
        """ 
        The discrete energy functional for the Burgers' equation:
                    ut + u*u_x = 0, Omega*(0,T]
                    u(lb,t) = u(rb,t), (0,T]
                    u(x,0) = u0(x), Omega
        with the periodical type of boundary condition and an initial 
        condition. Solve the equation by discretizing time variable 
        first and minimizing a loss function of the strong form of the 
        PDE on each time level:
            J(u) = || u + k*u*u_x - f ||^2 + mu*||u(lb)-u(rb)||^2
        with mu being the penalty parameter for the boundary condition
        and k being the time-step.
        INPUT: 
            activation: nonlinear activation functions.
            quadrature: full quadrature information of interior domain.
            boundary: lb and rb.
            time_step: length of time step.
            pde: a PDE object of the solution of last time step.
            device: cpu or cuda.
            penalty_bd: the parameter in front of boundary condition.
            parallel_evaluation: option for evaluation with a large scale.     
        """
        
        super(PINN_Burgers_1d_PBC, self).__init__()
        
        self.device = device
        self.sigma = activation.activate
        self.dsigma = activation.dactivate
        self.k = time_step
        self.penalty = penalty_bd
        
        self.boundary = boundary
        self.quadpts = quadrature.quadpts
        self.weights = quadrature.weights
        self.area = quadrature.area
        self.source_data = pde.source(self.quadpts)
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
        obj_val_bd = obj_func(self.boundary)
        obj_val_lb = obj_val_bd[0][0]
        obj_val_rb = obj_val_bd[1][0]
        items.append(obj_val_lb.detach())
        items.append(obj_val_rb.detach())
        
        # evaluation on domain quadpts
        quadpts = self.quadpts.requires_grad_()
        obj_val = obj_func(quadpts)
        items.append(obj_val.detach())
        
        # get gradient evaluated on all quadpts
        f = obj_val.sum()
        gradient = torch.autograd.grad(outputs=f, inputs=quadpts) #, create_graph=True)
        obj_grad_data = gradient[0].detach()
        items.append(obj_grad_data.detach())

        # reset self.quadpts
        self.quadpts = self.quadpts.detach()
        
        return items
    

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
        return 0. * p
    
    
    def energy_error(self):
        pass
    
    
    def loss(self):
        """
        Loss function on each time level
        """
        
        # get enery items
        items = self._get_energy_items(self.pre_solution)
        
        # interior residual
        u_val = items[2]
        u_x_val = items[3]
        residual_in = u_val + self.k*u_val*u_x_val - self.source_data
        residual_in = residual_in.pow(2) * self.weights
        residual_in = residual_in.sum() * self.area
        
        # boundary residual, periodical condition
        u_val_lb = items[0]
        u_val_rb = items[1]
        residual_bd = self.penalty*(u_val_lb-u_val_rb).pow(2)
        
        return residual_in + residual_bd
    
    
    def get_nonlinear_system(self, cur_solution, core_in, core_bd):
        """
        Construct the nonlinear operator and its Jacobian

        Args:
            cur_solution: the current solution u_n, in Newton's iteration
            core_in: a core matrix generated outside this energy class
            core_bd: a core matrix generated outside this energy class
        """
        
        # evaluation of cur_solution
        items = self._get_energy_items(cur_solution.forward)
        u_lb = items[0]
        u_rb = items[1]
        u = items[2]
        du = items[3]
        u2 = u.pow(2)
        du2 = du.pow(2)
        udu = u*du
        u2du = u.pow(2)*du
        udu2 = u*du.pow(2)
        f = self.source_data
        uf = u*f
        duf = du*f
        
        # core matrix and vectors used for vecterization
        g = self.sigma(core_in)
        dg = self.dsigma(core_in,1)
        gbd = self.sigma(core_bd)
        diff_bd = gbd[...,0:1] - gbd[...,1:2] 
        
        # get components of parameters, in a column
        num_neuron = g.shape[0]
        w = cur_solution.layer1.weight[0:num_neuron,0:1]
        I = torch.ones_like(w)
        
        # assemble vertical vector: the nonlinear operator
        G_blinear = torch.mm(g, u*self.weights)
        G_linear = self.k * torch.mm(g, udu*self.weights) + \
                    self.k * torch.mm(dg, (u2+self.k*u2du-uf)*self.weights) * w + \
                    self.k * torch.mm(g, (udu+self.k*udu2-duf)*self.weights)
        G_bd = self.penalty * diff_bd * (u_lb-u_rb)
        G_f = torch.mm(g, f*self.weights)*self.area
        G = (G_blinear + G_linear)*self.area + G_bd - G_f
        
        # assemble matrix: the Jacobian      
        J = torch.mm(g, (g*self.weights.t()).t()) * self.area  
        J += self.k * torch.mm(g, (g*du.t()*self.weights.t()).t()) * self.area
        J += self.k * torch.mm(g, (dg*u.t()*self.weights.t()).t()) * torch.mm(I, w.t()) * self.area
        J += self.k * torch.mm(dg, (g*(u+self.k*udu).t()*self.weights.t()).t()) * torch.mm(w, I.t()) * self.area
        J += self.k * torch.mm(dg, (dg*(self.k*u).t()*self.weights.t()).t()) * torch.mm(w, w.t()) * self.area
        J += self.k * torch.mm(g, (g*(du+self.k*du2).t()*self.weights.t()).t()) * self.area
        J += self.k * torch.mm(g, (dg*(self.k*udu).t()*self.weights.t()).t()) * torch.mm(I, w.t()) * self.area
        J += self.k * torch.mm(dg, (g*(u+self.k*udu-f).t()*self.weights.t()).t()) * torch.mm(w, I.t()) * self.area
        J += self.k * torch.mm(g, (dg*(u+self.k*udu-f).t()*self.weights.t()).t()) * torch.mm(I, w.t()) * self.area
        J += self.penalty * torch.mm(diff_bd, diff_bd.t())
        
        return (G, J)
    
    
    def evaluate(self, param):
        """
        Evaluation of the gradient of loss function
        """
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_lb = items[0]
        u_rb = items[1]
        u = items[2]
        du = items[3]
        u2 = u.pow(2)
        du2 = du.pow(2)
        udu = u*du
        u2du = u.pow(2)*du
        udu2 = u*du.pow(2)
        f = self.source_data
        uf = u*f
        duf = du*f
        
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
        g = self.sigma(core_in)
        dg = self.dsigma(core_in,1)
        gbd = self.sigma(core_bd)
        ug = torch.mm(g, u*self.weights) + \
                self.k * torch.mm(g, udu*self.weights) + \
                self.k * torch.mm(dg, (u2+self.k*u2du)*self.weights) * w + \
                self.k * torch.mm(g, (udu+self.k*udu2)*self.weights)
        ugbd = self.penalty * (gbd[...,0:1]-gbd[...,1:2]) * (u_lb-u_rb)
        fg = self.k * torch.mm(dg, uf*self.weights) * w + \
                self.k * torch.mm(g, duf*self.weights) + \
                torch.mm(g, f*self.weights)
                
        # assemble
        energy_eval = -(1/2)*((ug-fg)*self.area + ugbd).pow(2)
        
        return energy_eval
    
    
    def evaluate_large_scale(self, param):
        """ 
        The large scale evaluation of 1D problem.
        """
        return self.evaluate(param)        
        
        
    def update_solution(self, solution):
        self.pre_solution = solution