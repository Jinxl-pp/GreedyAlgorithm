"""
Created on Wen Oct 4 14:20 2023

@author: Jinpp (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Training shallow neural network using the variational
        loss (i.e., finite neuron method) and the orthogonal 
        greedy algorithm, to solve the following second-order 
        elliptic equation in 2D:
                    - Lap(u) = f, in Omega of R
                           u = g, on boundary of Omega
        with g=0 as the homogeneous Dirichlet's boundary condition.
        Solve this PDE by minimizing the variational loss function:
        J(v) = 
            (1/2)*(nabla v, nabla v) + (1/2*mu)*(v-g,v-g)_{boundary} - (f,v),
        with mu being the penalty parameter for boundary condition. 
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

from greedy.pde import cos2d
from greedy.tools import show_rate
from greedy.model import shallownet
from greedy.model import activation_function as af 
from greedy.model import neuron_dictionary_2d as ndict
from greedy.lossfunction import pinn_poisson_2d_dbc as loss
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
    num_quadpts = energy.quadpts_in.shape[0]
    num_bdnodes = energy.quadpts_bd.shape[0]
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
        Bk_in = torch.cat([energy.quadpts_in, ones], dim=1) 
        Ck_in = torch.mm(Ak, Bk_in.t()) 
        ones = torch.ones(num_bdnodes,1).to(device)
        Bk_bd = torch.cat([energy.quadpts_bd, ones], dim=1) 
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
    pde = cos2d.Data_Poisson_2d_DBC()
    
    # neuron dictionary settings
    ftype = "relu" 
    degree = 3
    activation = af.ActivationFunction(ftype, degree)
    optimizer = False 
    param_b_domain = torch.tensor([[-2., 2.]])
    param_mesh_size = 1/30
    dictionary = ndict.NeuronDictionary2D(activation,
                                        optimizer,
                                        param_b_domain,
                                        param_mesh_size,
                                        device)
    
    # M-C training data settings
    mc_quad = mc.MonteCarloQuadrature(device)
    nsamples_in = int(1e+4)
    nsamples_bd = int(1e+3)
    rectangle = np.array([[-1.,1.],[-1.,1.]])
    quadrature_in = mc_quad.rectangle_samples(rectangle, nsamples_in)
    quadrature_bd = mc_quad.rectangle_boundary_samples(rectangle, nsamples_bd)
    
    # enery loss function settings
    penalty = 1
    energy = loss.PINN_Poisson_2d_DBC(dictionary.activation,
                                    quadrature_in,
                                    quadrature_bd,
                                    pde,
                                    device,
                                    penalty)
    
    # oga training process
    num_neurons = 128
    snn = shallownet.ShallowNN(sigma=activation.activate,
                               in_dim=2,
                               width=num_neurons
                               )
    start = time.time()
    loss_val, l2_err, h2_err, snn = orthogonal_greedy(dictionary, energy, snn)
    end = time.time()
    
    
    # show error
    atype = 'OGA'
    total_time = end - start
    show_rate.pinn_method(num_neurons, loss_val, l2_err, h2_err, atype, ftype, degree, total_time)
    
    
    # example settings, part two:
    # training data: monte-carlo 
    # param_mesh_size = 1/30
    # number of interior samples: 1e+4
    # number of boundary samples: 1e+3 * 4
    # theoretical convergence rates:
    # O(n^-2.5) for loss decreasing
    # O(n^-1.25) in H2 (energy norm)
    # final results:
    # +------------------------------------------------------------------------------+
    # |               OGA-PINN, relu_power = 3, total time = 385.2297s               |
    # +-----+-----------+-----------+-----------+---------+------------+-------------+
    # |   N |    loss   | loss_rate |   l2_err  | l2_rate | energy_err | energy_rate |
    # +-----+-----------+-----------+-----------+---------+------------+-------------+
    # |   2 | 2.101e+01 |     -     | 1.654e+00 |    -    | 3.465e+00  |      -      |
    # |   4 | 1.803e+01 |    0.22   | 1.866e+00 |  -0.17  | 3.267e+00  |     0.08    |
    # |   8 | 7.433e+00 |    1.28   | 3.720e-01 |   2.33  | 2.308e+00  |     0.50    |
    # |  16 | 2.368e-01 |    4.97   | 7.886e-02 |   2.24  | 7.238e-01  |     1.67    |
    # |  32 | 2.148e-02 |    3.46   | 9.785e-03 |   3.01  | 1.799e-01  |     2.01    |
    # |  64 | 1.516e-03 |    3.82   | 1.240e-03 |   2.98  | 4.487e-02  |     2.00    |
    # | 128 | 1.933e-04 |    2.97   | 3.848e-04 |   1.69  | 2.226e-02  |     1.01    |
    # +-----+-----------+-----------+-----------+---------+------------+-------------+