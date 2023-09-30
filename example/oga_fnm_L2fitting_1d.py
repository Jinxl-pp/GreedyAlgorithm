"""
Created on Wed Sep 20 16:16 2023

@author: Jinpp (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Training shallow neural network using the variational
        loss (i.e., finite neuron method) and the orthogonal 
        greedy algorithm, to solve the following L2-fitting 
        problem in 1D:
                    u = f, in Omega of R. 
        The training data and the testing data are produced by
        piecewise Gauss-Legendre quadrature rule. For dictionary
        settings:
        (1) activation available for relu, bspline and sigmoid,
        (2) optimizer available for pgd, fista and False.
@modifications: to be added
"""

import sys
sys.path.append('../')

import time
import torch
import numpy as np

from greedy.pde import wave1d,cos1d
from greedy.tools import show_rate, show_solution
from greedy.model import shallownet
from greedy.model import activation_function as af 
from greedy.model import neuron_dictionary_1d as ndict
from greedy.lossfunction import fnm_L2fitting_1d as loss
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
        error = energy.energy_error()
        errl2 = torch.sqrt(error)
        errl2_record[k] = errl2      
        print("\n Current numerical errors:")
        print(' L2-error: {:.6e}'.format(errl2.item()))
        
        # find the currently best direction to reduce the energy
        optimal_element = dictionary.find_optimal_element(energy)
        
        # update parameter list
        for d in range(dim+1):
            inner_param[k][d] = optimal_element[d]

        # stiffness matrix and load vector
        start = time.time()
        Ak = inner_param[k,:].reshape(1,-1) 
        ones = torch.ones(num_quadpts,1).to(device)
        Bk = torch.cat([energy.quadpts, ones], dim=1) 
        Ck = torch.mm(Ak, Bk.t()) 
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
        w1 = inner_param[:,0:dim]
        b1 = inner_param[:,dim:dim+1].flatten()
        w2 = outer_param.clone()
        parameters = (w2, w1, b1)
        snn.update_neurons(parameters)
        
        # update the previous solution
        energy.update_solution(snn.forward)
    
    # return numerical results
    return errl2_record, snn


if __name__ == "__main__":
    
    # pde's exact solution
    pde = wave1d.DataWave_1d()
    
    # neuron dictionary settings
    ftype = "relu"
    degree = 1
    activation = af.ActivationFunction(ftype, degree)
    optimizer = False 
    param_b_domain = torch.tensor([[-np.pi-1, np.pi+1]])
    param_mesh_size = 1/2000
    dictionary = ndict.NeuronDictionary1D(activation,
                                        optimizer,
                                        param_b_domain,
                                        param_mesh_size,
                                        device)
    
    # training data settings
    nquadpts = 2
    index = nquadpts - 1
    h = np.array([1/2000])
    interval = np.array([[-np.pi, np.pi]])
    gl_quad = gq.GaussLegendreDomain(index, device)
    quadrature = gl_quad.interval_quadpts(interval, h)
    
    # enery loss function settings
    energy = loss.FNM_L2fitting_1d(dictionary.activation,
                                quadrature,
                                pde,
                                device)
   
    # oga training process
    num_neurons = 1024
    snn = shallownet.ShallowNN(sigma=activation.activate,
                               in_dim=1,
                               width=num_neurons
                               )
    start = time.time()
    l2_err, snn = orthogonal_greedy(dictionary, energy, snn)
    end = time.time()
    
    # show solution 
    show_solution.show_solution_1d(pde.solution, snn.forward, interval)
    
    # show error
    atype = 'OGA'
    total_time = end - start
    show_rate.finite_neuron_method_fitting(num_neurons, l2_err, atype, ftype, degree, total_time)