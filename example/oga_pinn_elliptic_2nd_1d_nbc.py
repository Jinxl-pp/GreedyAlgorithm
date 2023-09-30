"""
Created on Sat Aug 5 11:14 2023

@author: Jinpp (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Training shallow neural network using the residual 
        loss (i.e., PINN method) and the orthogonal greedy 
        algorithm, to solve the following second-order 
        elliptic equation in 1D:
                    - u_xx + u = f, in Omega of R
                    du/dx = g, on boundary of Omega
        with g=0 as the homogeneous Neumann's boundary condition.
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

from greedy.pde import cos1d
from greedy.tools import show_rate
from greedy.model import shallownet
from greedy.model import activation_function as af 
from greedy.model import neuron_dictionary_1d as ndict
from greedy.lossfunction import pinn_elliptic_2nd_1d_nbc as loss
from greedy.quadrature import gauss_legendre_quadrature as gq
from greedy.quadrature import monte_carlo_quadrature as mc

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
    loss_record = torch.zeros(num_epochs, 1).to(device)
    errl2_record = torch.zeros(num_epochs, 1).to(device)
    errh2_record = torch.zeros(num_epochs, 1).to(device)
    
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
        
        print("\n")
        print("-----------------------------")
        print('----the N = {:.0f}-th neuron----'.format(k+1))
        print("-----------------------------")
        
        # display numerical errors in each step
        loss = energy.loss()
        errors = energy.energy_error()
        errl2, errh2 = torch.sqrt(errors[0]), torch.sqrt(errors[1])
        loss_record[k] = loss
        errl2_record[k] = errl2
        errh2_record[k] = errh2        
        print("\n Current loss function and numerical errors:")
        print(' Loss: {:.6e}'.format(loss.item()))
        print(' L2-error: {:.6e}'.format(errl2.item()))
        print(' H2-error: {:.6e}'.format(errh2.item()))
        
        # find the currently best direction to reduce the energy
        optimal_element = dictionary.find_optimal_element(energy)
        
        # update parameter list
        for d in range(dim+1):
            inner_param[k][d] = optimal_element[d] 

        # stiffness matrix and load vector
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
        system = energy.get_stiffmat_and_rhs(inner_param[0:k+1,...], core_in, core_bd)
        
        # Galerkin orthogonal projection
        Gk, bk = system[0], system[1]
        coef = torch.linalg.solve(Gk, bk)
        outer_param[:, 0:k+1] = coef.reshape(1,-1).to(device)
        
        # clear 
        del system, core_in, core_bd, ones
        del Ak, Bk_in, Bk_bd, Ck_in, Ck_bd, Gk, bk
        
        # update the shallow network 
        w1 = inner_param[:,0:dim]
        b1 = inner_param[:,dim:dim+1].flatten()
        w2 = outer_param.clone()
        parameters = (w2, w1, b1)
        snn.update_neurons(parameters)
        
        # update the previous solution
        energy.update_solution(snn.forward)
    
    # return numerical results
    return loss_record, errl2_record, errh2_record, snn


if __name__ == "__main__":
    
    # pde's exact solution
    pde = cos1d.DataCos_2nd_1d_NBC()
    
    # neuron dictionary settings
    ftype = "relu" 
    degree = 3
    activation = af.ActivationFunction(ftype, degree)
    optimizer = False 
    param_b_domain = torch.tensor([[-2., 2.]])
    param_mesh_size = 1/1000
    dictionary = ndict.NeuronDictionary1D(activation,
                                        optimizer,
                                        param_b_domain,
                                        param_mesh_size,
                                        device)
    
    # # G-L training data settings
    # nquadpts = 2
    # index = nquadpts - 1
    # h = np.array([1/1000])
    # interval = np.array([[-1.,1.]])
    # boundary = torch.from_numpy(interval).to(device).reshape(-1,1)
    # gl_quad = gq.GaussLegendreDomain(index, device)
    # quadrature = gl_quad.interval_quadpts(interval, h)
    
    # M-C training data settings
    mc_quad = mc.MonteCarloDomain(device)
    nsamples = int(1e+4)
    interval = np.array([[-1.,1.]])
    boundary = torch.from_numpy(interval).to(device).reshape(-1,1)
    quadrature = mc_quad.interval_samples(interval, nsamples)
    
    # enery loss function settings
    penalty = 1
    energy = loss.PINN_Elliptic_2nd_1d_NBC(dictionary.activation,
                                    quadrature,
                                    boundary,
                                    pde,
                                    device,
                                    penalty)
    
    # oga training process
    num_neurons = 128
    snn = shallownet.ShallowNN(sigma=activation.activate,
                               in_dim=1,
                               width=num_neurons
                               )
    start = time.time()
    loss, l2_err, h2_err, snn = orthogonal_greedy(dictionary, energy, snn)
    end = time.time()
    
    
    # show error
    atype = 'OGA'
    total_time = end - start
    show_rate.pinn_method(num_neurons, loss, l2_err, h2_err, atype, ftype, degree, total_time)
    
    
    # example settings, part one:
    # training data: guass-legendre 
    # h = np.array([1/1000])     
    # param_mesh_size = 1/1000
    # 
    # theoretical convergence rates:
    # O(n^-4) for loss decreasing
    # O(n^-4) in L2, O(n^-2) in H2
    # final results:
    # +------------------------------------------------------------------------------+
    # |               OGA-PINN, relu_power = 3, total time = 56.5397s                |
    # +-----+-----------+-----------+-----------+---------+------------+-------------+
    # |   N |    loss   | loss_rate |   l2_err  | l2_rate | energy_err | energy_rate |
    # +-----+-----------+-----------+-----------+---------+------------+-------------+
    # |   2 | 2.328e+02 |     -     | 1.092e+00 |    -    | 1.038e+01  |      -      |
    # |   4 | 1.886e+02 |    0.30   | 3.450e+00 |  -1.66  | 9.818e+00  |     0.08    |
    # |   8 | 1.549e+00 |    6.93   | 1.549e-02 |   7.80  | 8.741e-01  |     3.49    |
    # |  16 | 1.292e-02 |    6.91   | 2.192e-04 |   6.14  | 8.035e-02  |     3.44    |
    # |  32 | 6.008e-04 |    4.43   | 6.655e-06 |   5.04  | 1.733e-02  |     2.21    |
    # |  64 | 3.510e-05 |    4.10   | 3.238e-07 |   4.36  | 4.189e-03  |     2.05    |
    # | 128 | 2.067e-06 |    4.09   | 1.893e-08 |   4.10  | 1.017e-03  |     2.04    |
    # +-----+-----------+-----------+-----------+---------+------------+-------------+
    
    
    # example settings, part two:
    # training data: monte-carlo 
    # number of samples: 1e+4
    # final results:
    # +------------------------------------------------------------------------------+
    # |               OGA-PINN, relu_power = 3, total time = 132.9729s               |
    # +-----+-----------+-----------+-----------+---------+------------+-------------+
    # |   N |    loss   | loss_rate |   l2_err  | l2_rate | energy_err | energy_rate |
    # +-----+-----------+-----------+-----------+---------+------------+-------------+
    # |   2 | 2.355e+02 |     -     | 1.089e+00 |    -    | 1.043e+01  |      -      |
    # |   4 | 1.899e+02 |    0.31   | 3.458e+00 |  -1.67  | 9.889e+00  |     0.08    |
    # |   8 | 1.452e+00 |    7.03   | 1.553e-02 |   7.80  | 8.465e-01  |     3.55    |
    # |  16 | 1.217e-02 |    6.90   | 3.433e-04 |   5.50  | 7.795e-02  |     3.44    |
    # |  32 | 6.221e-04 |    4.29   | 6.214e-05 |   2.47  | 1.764e-02  |     2.14    |
    # |  64 | 3.522e-05 |    4.14   | 3.616e-05 |   0.78  | 4.197e-03  |     2.07    |
    # | 128 | 2.147e-06 |    4.04   | 2.548e-05 |   0.51  | 1.037e-03  |     2.02    |
    # +-----+-----------+-----------+-----------+---------+------------+-------------+