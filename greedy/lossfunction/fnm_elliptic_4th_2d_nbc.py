import torch
from torch.autograd import Variable
import numpy as np
from . import energy 

class FNM_Elliptic_4th_2d_NBC(energy.LinearEnergy): 
    
    def __init__(self, 
                activation,
                quadrature,
                pde,
                device,
                parallel_evaluation=False): 
        
        """ 
        The discrete energy functional for the second order elliptic PDE:
                        [-Lap]^2(u) + u = f, in Omega of R
                    [d\dn](Lap(u)+[d/ds]^2(u)) 
                       - [d\ds](k*du/ds) = g_1, on boundary of Omega
                             [d/dn]^2(u) = g_2, on boundary of Omega
        with g_1=g_2=0 as the homogeneous Neumann's boundary condition.
        The first Neumann-type boundary condition consists of the tangential 
        direction [d/ds] and the curvature of boundary k. One can refer to 
        [1] 
            Chien, W. CH. (1980). Variational methods and finite elements.
            
        for more details. Solve the equation by minimizing an energy functional 
        with weak form of this PDE.
            J(u) = (1/2)*(hessian(u),hessian(u)) + (1/2)*(u,u) - (f,u) - (g_1+g_2,u).
        INPUT: 
            activation: nonlinear activation functions.
            quadrature: full quadrature information.
            pde: a PDE object used in energy evaluation.
            device: cpu or cuda.
            parallel_evaluation: option for evaluation with a large scale.     
        """
        super(FNM_Elliptic_4th_2d_NBC, self).__init__()
        
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
        quadpts = torch.cat([xpts,ypts],dim=1)
        obj_val = obj_func(quadpts)
        items.append(obj_val.detach())
        
        # get gradient evaluated on all quadpts
        f = obj_val.sum()
        gradient_x = torch.autograd.grad(outputs=f, inputs=xpts, create_graph=True)
        gradient_y = torch.autograd.grad(outputs=f, inputs=ypts, create_graph=True)
        items.append(gradient_x[0].detach())
        items.append(gradient_y[0].detach())

        # get second-order derivatives evaluated on all quadpts
        dxf = gradient_x[0].sum()
        dyf = gradient_y[0].sum()
        hessian_xx = torch.autograd.grad(outputs=dxf, inputs=xpts, create_graph=True)
        hessian_xy = torch.autograd.grad(outputs=dxf, inputs=ypts, create_graph=True)
        hessian_yx = torch.autograd.grad(outputs=dyf, inputs=xpts, create_graph=True)
        hessian_yy = torch.autograd.grad(outputs=dyf, inputs=ypts)
        items.append(hessian_xx[0].detach())
        items.append(hessian_xy[0].detach())
        items.append(hessian_yx[0].detach())
        items.append(hessian_yy[0].detach())
        
        # reset self.quadpts
        self.quadpts = self.quadpts.detach()
        
        return items
    
    
    def _get_error(self, p):
        return self.pde.solution(p) - self.pre_solution(p)
    
    
    def _partition(self, num_param, n):
        division = len(num_param) / n
        return [0]+[round(division * (i + 1)) for i in range(n)]
    
    
    def _zero(self, p):
        x = p[...,0:1]
        y = p[...,1:2]
        return 0. * torch.sin(x) * torch.sin(y)
    
    
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
        v11 = w1 * w1
        v12 = w1 * w2
        v21 = w2 * w1
        v22 = w2 * w2
        
        # core matrix and vectors used for vecterization
        g1 = self.sigma(core)
        d2g1 = self.dsigma(core, 2)
        g2 = g1 * self.weights.t()
        d2g2 = d2g1 * self.weights.t()
        
        # assemble stiffness matrix
        G = torch.mm(g1, g2.t()) * self.area
        coef = torch.mm(v11, v11.t()) + torch.mm(v12, v12.t()) + \
                torch.mm(v21, v21.t()) + torch.mm(v22, v22.t())
        d2G = torch.mm(d2g1, d2g2.t()) * coef * self.area
        Gk = d2G + G
        
        # assemble load vector
        f = self.source_data * self.weights
        bk = torch.mm(g1, f) * self.area 
        
        return (Gk, bk)
    
    
    def evaluate(self, param):
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_val = items[0]
        u_hess_xx = item[3]
        u_hess_xy = item[4]
        u_hess_yx = item[5]
        u_hess_yy = item[6]
                
        # get components of theta
        w1 = param[0]
        w2 = param[1]
        A = torch.cat([param[0], param[1], param[2]], dim=1)
        ones = torch.ones(len(self.quadpts),1).to(self.device)
        B = torch.cat([self.quadpts, ones], dim=1).t()  
        
        # core matrix and vectors used for vecterization
        core = torch.mm(A, B)
        g = self.sigma(core)
        d2g = self.dsigma(core, 2)
        fg = torch.mm(g, self.source_data * self.weights)
        ug = torch.mm(g, u_val * self.weights)
        dxxudxxg = torch.mm(d2g, u_hess_xx * self.weights) * w1 * w1
        dxyudxyg = torch.mm(d2g, u_hess_xy * self.weights) * w1 * w2
        dyyudyyg = torch.mm(d2g, u_hess_yy * self.weights) * w2 * w2
        d2ud2g = dxxudxxg + 2*dxyudxyg + dyyudyyg
        
        # assemble
        energy_eval = -(1/2)*((d2ud2g+ug-fg) * self.area).pow(2)
            
        return energy_eval
    
    
    def evaluate_large_scale(self, param):
        """
        The large scale evaluation of 2D problem, for evaluation
        in the multiple-parameter situation.
        """
        
        # get items of the energy bilinear form, denote pre_solution := u
        items = self._get_energy_items(self.pre_solution)
        u_val = items[0]
        u_hess_xx = items[3]
        u_hess_xy = items[4]
        u_hess_yx = items[5]
        u_hess_yy = items[6]
                
        # get components of theta
        w1 = param[0]
        w2 = param[1]
        A = torch.cat([param[0], param[1], param[2]], dim=1)
        ones = torch.ones(len(self.quadpts),1).to(self.device)
        B = torch.cat([self.quadpts, ones], dim=1).t()  
        
        # core matrix and vectors used for vecterization
        num_param = A.shape[0]
        ug = torch.zeros(num_param,1)
        fg = torch.zeros(num_param,1)
        dxxudxxg = torch.zeros(num_param,1) 
        dxyudxyg = torch.zeros(num_param,1)    
        dyyudyyg = torch.zeros(num_param,1)
        
        num_batch = 2
        division = self._partition(range(num_param),num_batch)
        for i in range(num_batch):
            core = torch.mm(A[division[i]:division[i+1],:], B)
            g = self.sigma(core)
            d2g = self.dsigma(core, 2)
            del core
            ug[division[i]:division[i+1]] = torch.mm(g, u_val * self.weights)
            fg[division[i]:division[i+1]] = torch.mm(g, self.source_data * self.weights)
            dxxudxxg[division[i]:division[i+1]] = torch.mm(d2g, u_hess_xx * self.weights) * \
                                        w1[division[i]:division[i+1],:] * w1[division[i]:division[i+1],:]
            dxyudxyg[division[i]:division[i+1]] = torch.mm(d2g, u_hess_xy * self.weights) * \
                                        w1[division[i]:division[i+1],:] * w2[division[i]:division[i+1],:]
            dyyudyyg[division[i]:division[i+1]] = torch.mm(d2g, u_hess_yy * self.weights) * \
                                        w2[division[i]:division[i+1],:] * w2[division[i]:division[i+1],:]
            del g, d2g
        d2ud2g = dxxudxxg + 2*dxyudxyg + dyyudyyg
        
        # assemble
        energy_eval = -(1/2)*((d2ud2g+ug-fg) * self.area).pow(2)
            
        return energy_eval
    
    
    def update_solution(self, pre_solution):
        self.pre_solution = pre_solution
    