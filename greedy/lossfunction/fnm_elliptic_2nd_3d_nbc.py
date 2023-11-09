import torch
import numpy as np
from . import energy 

class FNM_Elliptic_2nd_3d_NBC(energy.LinearEnergy):
    
    def __init__(self, 
                activation,
                quadrature,
                pde,
                device,
                parallel_evaluation=False): 
        """ 
        The discrete energy functional for the second order elliptic PDE:
                        - Lap(u) + u = f, in Omega of R
                             du/dx = g, on boundary of Omega
        with g=0 as the homogeneous Neumann's boundary condition.
        Solve the equation by minimizing an energy functional with 
        weak form of this PDE.
            J(u) = (1/2)*(nabla_u,nabla_u) + (1/2)*(u,u) - (f,u) - (g,u).
        INPUT: 
            activation: nonlinear activation functions.
            quadrature: full quadrature information.
            pde: a PDE object used in energy evaluation.
            device: cpu or cuda.
            parallel_evaluation: option for evaluation with a large scale.     
        """
        super(FNM_Elliptic_2nd_3d_NBC, self).__init__()
        
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
        
        # initialized enery-item
        items = []
        
        # gradient evaluation on quadpts
        xpts = self.quadpts[...,0:1].requires_grad_()
        ypts = self.quadpts[...,1:2].requires_grad_()
        zpts = self.quadpts[...,2:3].requires_grad_()
        quadpts = torch.cat([xpts,ypts,zpts],dim=1)
        obj_val = obj_func(quadpts)
        items.append(obj_val.detach())
        
        # get gradient evaluated on all quadpts
        f = obj_val.sum()
        gradient_x = torch.autograd.grad(outputs=f, inputs=xpts, create_graph=True)
        gradient_y = torch.autograd.grad(outputs=f, inputs=ypts, create_graph=True)
        gradient_z = torch.autograd.grad(outputs=f, inputs=zpts)
        items.append(gradient_x[0].detach())
        items.append(gradient_y[0].detach())
        items.append(gradient_z[0].detach())
        
        # reset self.quadpts
        self.quadpts = self.quadpts.detach()
        
        return items
    
    
    def _get_error(self, p):
        return self.pde.solution(p) - self.pre_solution(p)
    
    
    def _partition(self, num_param, n):
        division = len(num_param) / n
        return [0]+[round(division * (i + 1)) for i in range(n)]
    
    
    def _zero(self, p):
        return 0. * p[...,0:1]
    
    
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
        w1 = parameters[:,0:1]
        w2 = parameters[:,1:2]
        w3 = parameters[:,2:3]
        
        # core matrix and vectors used for vecterization
        g1 = self.sigma(core)
        dg1 = self.dsigma(core, 1)
        g2 = g1 * self.weights.t()
        dg2 = dg1 * self.weights.t()
        
        # assemble stiffness matrix
        G = torch.mm(g1, g2.t()) * self.area
        dxG = torch.mm(dg1, dg2.t()) * torch.mm(w1, w1.t()) * self.area
        dyG = torch.mm(dg1, dg2.t()) * torch.mm(w2, w2.t()) * self.area
        dzG = torch.mm(dg1, dg2.t()) * torch.mm(w3, w3.t()) * self.area
        Gk = dxG + dyG + dzG + G
        
        # assemble load vector
        f = self.source_data * self.weights
        bk = torch.mm(g1, f) * self.area 
        
        return (Gk, bk)
    

    def evaluate(self, param):
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_val = items[0]
        u_grad_x = items[1]
        u_grad_y = items[2]
        u_grad_z = items[3]
                
        # get components of theta
        w1 = param[0]
        w2 = param[1]
        w3 = param[2]
        A = torch.cat([param[0], param[1], param[2]], dim=1)
        ones = torch.ones(len(self.quadpts),1).to(self.device)
        B = torch.cat([self.quadpts, ones], dim=1).t()  
        
        # core matrix and vectors used for vecterization
        core = torch.mm(A, B)
        g = self.sigma(core)
        dg = self.dsigma(core, 1)
        fg = torch.mm(g, self.source_data * self.weights)
        ug = torch.mm(g, u_val * self.weights)
        dxudg = torch.mm(dg, u_grad_x * self.weights) * w1
        dyudg = torch.mm(dg, u_grad_y * self.weights) * w2
        dzudg = torch.mm(dg, u_grad_z * self.weights) * w3
        dudg = dxudg + dyudg + dzudg
        
        # assemble
        energy_eval = -(1/2)*((dudg+ug-fg) * self.area).pow(2)
            
        return energy_eval
    
    
    def evaluate_large_scale(self, param):
        """
        The large scale evaluation of 2D problem, for evaluation
        in the multiple-parameter situation.
        """
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_val = items[0]
        u_grad_x = items[1]
        u_grad_y = items[2]
        u_grad_z = items[3]  
                
        # get components of theta
        w1 = param[0]
        w2 = param[1]
        w3 = param[2]
        A = torch.cat([param[0], param[1], param[2], param[3]], dim=1)
        ones = torch.ones(len(self.quadpts),1).to(self.device)
        B = torch.cat([self.quadpts, ones], dim=1).t()  
        print(A.shape)
        print(B.shape)

        # core matrix and vectors used for vecterization
        num_param = A.shape[0]
        ug = torch.zeros(num_param,1)
        fg = torch.zeros(num_param,1)
        dxudg = torch.zeros(num_param,1) 
        dyudg = torch.zeros(num_param,1)        
        dzudg = torch.zeros(num_param,1)        
        
        num_batch = 2
        division = self._partition(range(num_param),num_batch)
        for i in range(num_batch):
            core = torch.mm(A[division[i]:division[i+1],:], B)
            g = self.sigma(core)
            dg = self.dsigma(core, 1)
            del core
            ug[division[i]:division[i+1]] = torch.mm(g, u_val * self.weights)
            fg[division[i]:division[i+1]] = torch.mm(g, self.source_data * self.weights)
            dxudg[division[i]:division[i+1]] = torch.mm(dg, u_grad_x * self.weights) * w1[division[i]:division[i+1],:]
            dyudg[division[i]:division[i+1]] = torch.mm(dg, u_grad_y * self.weights) * w2[division[i]:division[i+1],:]
            dzudg[division[i]:division[i+1]] = torch.mm(dg, u_grad_z * self.weights) * w3[division[i]:division[i+1],:]
            del g, dg
        dudg = dxudg + dyudg + dzudg
        
        # assemble
        energy_eval = -(1/2)*((dudg+ug-fg) * self.area).pow(2)   
            
        return energy_eval
    
    
    def update_solution(self, pre_solution):
        self.pre_solution = pre_solution