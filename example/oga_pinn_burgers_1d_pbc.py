"""
Created on Wen Nov 8 14:30 2023

@author: Jinpp (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Training shallow neural network using the residual 
        loss (i.e., PINN method) and the orthogonal greedy 
        algorithm, to solve the following Burgers' equation
        in 1D * 1D:
                    ut + u*u_x = 0, (-1,1)*(0,T]
                    u(lb,t) = u(rb,t), (0,T]
                    u(x,0) = u0(x), (-1,1)
        with the periodical type of boundary condition and an
        initial condition. The training data and the testing 
        data are produced by piecewise Gauss-Legendre quadrature 
        rule. For dictionary settings:
        (1) activation available for relu, bspline and sigmoid,
        (2) optimizer available for pgd, fista and False.
@modifications: to be added
"""

import sys
sys.path.append('../')

import time
import torch
import numpy as np

from greedy.pde import sin1d
from greedy.model import shallownet
from greedy.model import activation_function as af 
from greedy.model import neuron_dictionary_1d as ndict
from greedy.tools import show_solution as show
from greedy.lossfunction import pinn_burgers_1d as loss
from greedy.quadrature import gauss_legendre_quadrature as gq

# precision settings
torch.set_printoptions(precision=25)
data_type = torch.float64
torch.set_default_dtype(data_type)

# device settings
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(use_gpu)


class NewtonLinearWeights():
    
    def __init__(self, 
        energy,
        shallow_nn,
        core_in,
        core_bd,
        newton_eta,
        newton_min,
        newton_max,
        newton_tol,
        back_track_number
        ): 
        
        self.eta_initial = newton_eta
        self.eta = newton_eta
        self.min = newton_min
        self.max = newton_max
        self.tol = newton_tol
        self.btmax = back_track_number
    
        self.snn = shallow_nn
        self.core_in = core_in
        self.core_bd = core_bd
        self.num_neuron = core_in.shape[0]
        self.nsystem = energy.get_nonlinear_system
        
    def _set_eta(self, factor):
        self.eta = factor * self.eta    
        
    def _reset_eta(self):
        self.eta = self.eta_initial
    
    def _forward(self, x, eta ,dx):
        x += eta*dx
        return x
        
    def step(self):
        
        # get coefficient
        w2 = self.snn.layer2.weight.detach()
        w1 = self.snn.layer1.weight.detach()
        b1 = self.snn.layer1.bias.detach()
        coef = w2[...,0:self.num_neuron]
        
        # starting step
        nsystem = self.nsystem(self.snn, self.core_in, self.core_bd)
        G, J = nsystem[0], nsystem[1]
        residual = torch.norm(G)
        # print(' Newton residual: {:.6e}'.format(residual.item()))
        pre_residual = 10
        
        # optimizing
        newton_counter = 0
        while (newton_counter < self.min or residual > self.tol ) and newton_counter < self.max:
            
            d = -1. * torch.linalg.solve(J, G)
            coef = self._forward(coef, self.eta, d.t())
            w2[...,0:self.num_neuron] = coef
            parameters = (w2, w1, b1)
            self.snn.update_neurons(parameters)
            
            nsystem = self.nsystem(self.snn, self.core_in, self.core_bd)
            G = nsystem[0]
            residual = torch.norm(G)
            bt_counter = 0
            bt_factor = 0.9
            
            # print(' Newton residual: {:.6e}'.format(residual.item()))
            while residual > pre_residual and bt_counter < self.btmax:
                
                coef = self._forward(coef, -self.eta, d.t())
                self._set_eta(bt_factor)
                coef = self._forward(coef, self.eta, d.t())
                w2[...,0:self.num_neuron] = coef
                
                self.snn.update_neurons(parameters)
                
                nsystem = self.nsystem(self.snn, self.core_in, self.core_bd)
                G = nsystem[0]
                residual = torch.norm(G)
                bt_counter += 1
                
            if residual > pre_residual:
                self._forward(coef, -self.eta, d.t())
                w2[...,0:self.num_neuron] = coef
                parameters = (w2, w1, b1)
                self.snn.update_neurons(parameters)
                
            if bt_counter >= self.btmax:
                newton_counter = self.max
                
            pre_residual = residual
            newton_counter += 1
            self._reset_eta()
            
        return self.snn



# training framework 
def orthogonal_greedy(dictionary, energy, snn, loss_tol):
    
    # iteration settings
    num_epochs = snn.num_neurons
    loss_record = torch.zeros(num_epochs, 1).to(device)
    
    # iteration values
    dim = dictionary.geo_dim
    num_quadpts = energy.quadpts.shape[0]
    num_bdnodes = energy.boundary.shape[0]
    core_mat_in = torch.zeros(num_epochs, num_quadpts).to(device)
    core_mat_bd = torch.zeros(num_epochs, num_bdnodes).to(device)
    inner_param = torch.zeros(num_epochs, dim+1).to(device) 
    outer_param = torch.zeros(1, num_epochs).to(device)   
    
    # iteration
    for k in range(num_epochs):
        
        # numerical loss
        loss = energy.loss()
        loss_record[k] = loss   
        
        if (k+1)%10 == 0:
            print("\n")
            print("-----------------------------")
            print('----the N = {:.0f}-th neuron----'.format(k+1))
            print("-----------------------------")
            
            # display numerical errors
            print("\n Current loss function and numerical errors:")
            print(' Loss: {:.6e}'.format(loss.item()))
        
        # stopping criterion for OGA
        if loss < loss_tol:
            break
        
        # find the currently best direction to reduce the energy
        optimal_element = dictionary.find_optimal_element(energy)
        
        # update parameter list
        for d in range(dim+1):
            inner_param[k][d] = optimal_element[d] 

        # current solution and nonlinear operator
        start = time.time()
        Ak = inner_param[k,:].reshape(1,-1) 
        ones = torch.ones(num_quadpts,1).to(device)
        Bk_in = torch.cat([energy.quadpts, ones], dim=1) 
        Ck_in = torch.mm(Ak, Bk_in.t()) 
        ones = torch.ones(num_bdnodes,1).to(device)
        Bk_bd = torch.cat([energy.boundary, ones], dim=1) 
        Ck_bd = torch.mm(Ak, Bk_bd.t()) 
        core_mat_in[k:k+1, :] = Ck_in
        core_mat_bd[k:k+1, :] = Ck_bd
        core_in = core_mat_in[0:k+1, :]
        core_bd = core_mat_bd[0:k+1, :]
        
        # clear 
        del Ak, Bk_in, Bk_bd, Ck_in, Ck_bd
        
        # initialize the shallow network 
        w1 = inner_param[:,0:dim]
        b1 = inner_param[:,dim:dim+1].flatten()
        w2 = outer_param.clone()
        parameters = (w2, w1, b1)
        snn.update_neurons(parameters)
        
        # Newton's iteration, update the shallow network 
        optimizer = NewtonLinearWeights(energy, snn, core_in, core_bd,
                                        newton_eta=1,
                                        newton_min=3,
                                        newton_max=30,
                                        newton_tol=1e-14,
                                        back_track_number=100)
        snn = optimizer.step()
        outer_param = snn.layer2.weight.detach()
        
        # update the previous solution
        energy.update_solution(snn.forward)
    
    # return numerical results
    return loss_record, snn



if __name__ == "__main__":
    
    # pde's exact solution
    pde = sin1d.DataSin_1d_PBC()
    
    # neuron dictionary settings
    ftype = "relu" 
    degree = 2
    activation = af.ActivationFunction(ftype, degree)
    optimizer = False 
    param_b_domain = torch.tensor([[-1., 1.]])
    param_mesh_size = 1/1000
    dictionary = ndict.NeuronDictionary1D(activation,
                                        optimizer,
                                        param_b_domain,
                                        param_mesh_size,
                                        device)
    
    # G-L training data settings
    nquadpts = 2
    index = nquadpts - 1
    h = np.array([1/1000])
    interval = np.array([[-1.,1.]])
    boundary = torch.from_numpy(interval).to(device).reshape(-1,1)
    gl_quad = gq.GaussLegendreDomain(index, device)
    quadrature = gl_quad.interval_quadpts(interval, h)
    
    
    # oga training process
    penalty = 1
    time_step = 1e-3
    max_time = 2
    nTime = int(max_time/time_step) + 1
    num_neurons = 300
    obj_func = pde.source
    energy=1
    time_slice = [0,100,200,300,400,500,600,636,700,800,900,1000]
        
    
    for i in range(nTime):
       
        # update pde object 
        pde.source = obj_func
       
        # enery loss function settings
        energy = loss.PINN_Burgers_1d_PBC(dictionary.activation,
                                    quadrature,
                                    boundary,
                                    time_step,
                                    pde,
                                    device,
                                    penalty)
        
        # snn for the current time step
        snn = shallownet.ShallowNN(sigma=activation.activate,
                                in_dim=1,
                                width=num_neurons
                                )        
        
        # training snn
        loss_tol = 1e-10
        loss_val, snn = orthogonal_greedy(dictionary, energy, snn, loss_tol)
        obj_func = snn.forward
        
        # save and plot model snn
        if i in time_slice:
            t = (i+1) * time_step
            time_label = 't={:.4f}s'.format(t)
            show.show_solution_1d(snn.forward, interval, label=time_label)
        
        