"""
Created on Sat Aug 5 11:14 2023

@author: Jinpp (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Training shallow neural network using the variational
        loss and the orthogonal greedy algorithm, to solve the
        following second order elliptic equation in 1D:
                    - u_xx + u = f, in Omega of R
                    du/dx = g, on boundary of Omega
        with g=0 as the homogeneous Neumann's boundary condition.
        The training data and the testing data are produced by
        piecewise Gauss-Legendre quadrature rule.
@modifications: to be added
"""

import sys
sys.path.append('../')

import time
import torch
import numpy as np

from greedy.pde import cos1d
from greedy.tools import show_rate
from greedy.model import shallownet
from greedy.model import activation_function as af 
from greedy.model import neuron_dictionary_1d as ndict
from greedy.lossfunction import elliptic_2ord_1d_neumann_bc as loss
from greedy.quadrature import gauss_legendre_quadrature as gq

# precision settings
torch.set_printoptions(precision=25)
data_type = torch.float64
torch.set_default_dtype(data_type)

# device settings
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(use_gpu)


# training framework 
def orthogonal_greedy(dictionary, energy, snn):
    
    # iteration settings
    num_epochs = snn.num_neurons
    errl2_record = torch.zeros(num_epochs, 1).to(device)
    erra_record = torch.zeros(num_epochs, 1).to(device)
    
    # iteration values
    dim = dictionary.geo_dim
    num_quadpts = energy.quadpts.shape[0]
    core_mat = torch.zeros(num_epochs, num_quadpts).to(device)
    inner_param = torch.zeros(num_epochs, dim+1).to(device) # inner parameters
    outer_param = torch.zeros(1, num_epochs).to(device)   # outer parameters
    
    # iteration
    for k in range(num_epochs):
        
        print("\n")
        print("-----------------------------")
        print('----the N = {:.0f}-th neuron----'.format(k+1))
        print("-----------------------------")
        
        # display numerical errors in each step
        errors = energy.energy_error()
        errl2, erra = torch.sqrt(errors[0]), torch.sqrt(errors[1])
        errl2_record[k] = errl2
        erra_record[k] = erra        
        print("\n Current numerical errors:")
        print(' L2-error: {:.6e}'.format(errl2.item()))
        print(' Hm-error: {:.6e}'.format(erra.item()))
        
        # find the currently best direction to reduce the energy
        optimal_element = dictionary.find_optimal_element(energy)
        
        # update parameter list
        inner_param[k][0] = optimal_element[0] # w
        inner_param[k][1] = optimal_element[1] # b
        
        # stiffness matrix and load vector
        start = time.time()
        Ak = inner_param[k,:].reshape(1,-1) # (w, b)^T
        ones = torch.ones(num_quadpts,1).to(device)
        Bk = torch.cat([energy.quadpts, ones], dim=1) # (x,1)
        Ck = torch.mm(Ak, Bk.t()) # (w, b)^T * (x, 1)^T
        core_mat[k:k+1, :] = Ck
        core = core_mat[0:k+1, :]
        system = energy.get_stiffmat_and_rhs(inner_param[0:k+1,...], core)
        
        # Galerkin orthogonal projection
        Gk, bk = system[0], system[1]
        coef = torch.linalg.solve(Gk, bk)
        outer_param[:, 0:k+1] = coef.reshape(1,-1).to(device)
        
        # clear 
        del system, core, ones
        del Ak, Bk, Ck, Gk, bk
        
        # update the shallow network 
        w1 = inner_param[:,0:1]
        b1 = inner_param[:,1:2].flatten()
        w2 = outer_param.clone()
        parameters = (w2, w1, b1)
        snn.update_neurons(parameters)
        
        # update the previous solution
        energy.update_solution(snn.forward)
    
    # return numerical results
    return errl2_record, erra_record, snn
 



if __name__ == "__main__":
    
    # pde's exact solution
    pde = cos1d.DataCos1m1dNeumannBC()
    
    # neuron dictionary settings
    ftype = "relu" # "relu" # "bspline" # "sigmoid"
    degree = 2
    activation = af.ActivationFunction(ftype, degree)
    optimizer = False #False # "pgd" # "fista" 
    param_b_domain = torch.tensor([[-2., 2.]])
    param_mesh_size = 1/1000
    dictionary = ndict.NeuronDictionary1D(activation,
                                        optimizer,
                                        param_b_domain,
                                        param_mesh_size,
                                        device)
    
    # enery loss function settings
    index = 10
    h = np.array([1/1000])
    interval = np.array([[-1.,1.]])
    gl_quad = gq.GaussLegendreDomain(index, device)
    # quadrature = gl_quad.interval_quadpts(interval, h)
    # energy = loss.Elliptic_2ord_1d_NBC(dictionary.activation,
    #                                 quadrature,
    #                                 pde,
    #                                 device)
   
    # # oga training process
    # num_neurons = 128
    # snn = shallownet.ShallowNN(sigma=activation.activate,
    #                            in_dim=1,
    #                            width=num_neurons
    #                            )
    # start = time.time()
    # l2_err, a_err, snn = orthogonal_greedy(dictionary, energy, snn)
    # end = time.time()
    
    # # show error
    # atype = 'OGA'
    # total_time = end - start
    # show_rate.finite_neuron_method(num_neurons, l2_err, a_err, atype, ftype, degree, total_time)
    
    # show solution
    