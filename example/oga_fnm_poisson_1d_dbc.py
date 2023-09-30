"""
Created on Sun Sep 24 14:01 2023

@author: Jinpp (xianlincn@pku.edu.cn)
@version: 1.0
@brief: Training shallow neural network using the variational
        loss (i.e., finite neuron method) and the orthogonal 
        greedy algorithm, to solve the following second-order 
        elliptic equation in 1D:
                    - u_xx = f, in Omega of R
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

from greedy.pde import cos1d
from greedy.tools import show_rate
from greedy.model import shallownet
from greedy.model import activation_function as af 
from greedy.model import neuron_dictionary_1d as ndict
from greedy.lossfunction import fnm_poisson_1d_dbc as loss
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
    
    # iteration values
    dim = dictionary.geo_dim
    num_quadpts = energy.quadpts.shape[0]
    num_bdnodes = energy.boundary.shape[0]
    core_mat_in = torch.zeros(num_epochs, num_quadpts).to(device)
    core_mat_bd = torch.zeros(num_epochs, num_bdnodes).to(device)
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
        print("\n Current numerical errors:")
        print(' L2-error: {:.6e}'.format(errl2.item()))
        print(' Hm-error: {:.6e}'.format(erra.item()))
        
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
    return errl2, erra, snn



if __name__ == "__main__":
    
    # pde's exact solution
    pde = cos1d.Data_poisson_1d_DBC()
    
    # neuron dictionary settings
    ftype = "relu" 
    degree = 2
    activation = af.ActivationFunction(ftype, degree)
    optimizer = False 
    param_b_domain = torch.tensor([[-2., 2.]])
    param_mesh_size = 1/1000
    dictionary = ndict.NeuronDictionary1D(activation,
                                        optimizer,
                                        param_b_domain,
                                        param_mesh_size,
                                        device)
    
    # training data settings
    nquadpts = 2
    index = nquadpts - 1
    h = np.array([1/1000])
    interval = np.array([[-1.,1.]])
    gl_quad = gq.GaussLegendreDomain(index, device)
    quadrature = gl_quad.interval_quadpts(interval, h)
    
    # test order 
    max_it = 7
    l2_err = torch.zeros(max_it,1)
    a_err = torch.zeros(max_it,1)
    start = time.time()
    for i in range(max_it):
        # enery loss function settings
        num_neurons = 2**(i+1)
        penalty = 1e-1 * num_neurons**(-2)
        boundary = torch.from_numpy(interval).to(device).reshape(-1,1)
        energy = loss.FNM_Poisson_1d_DBC(dictionary.activation,
                                        quadrature,
                                        boundary,
                                        penalty,
                                        pde,
                                        device)
        
        # oga training process
        snn = shallownet.ShallowNN(sigma=activation.activate,
                                in_dim=1,
                                width=num_neurons
                                )
        l2, a, _ = orthogonal_greedy(dictionary, energy, snn)
        l2_err[i][0] = l2
        a_err[i][0] = a.detach()
        
    end = time.time()

    # show error
    atype = 'OGA'
    total_time = end - start
    show_rate.finite_neuron_method(num_neurons, l2_err, a_err, atype, ftype, degree, total_time)
    
    # example settings:
    # h = np.array([1/1000])     
    # param_mesh_size = 1/1000
    # 
    # final results:
    # +------------------------------------------------------+
    # |   OGA-FNM, relu_power = 2, total time = 101.1392s    |
    # +-----+-----------+---------+------------+-------------+
    # |   N |   l2_err  | l2_rate | energy_err | energy_rate |
    # +-----+-----------+---------+------------+-------------+
    # |   2 | 6.780e-01 |    -    | 5.089e+00  |      -      |
    # |   4 | 8.189e-01 |  -0.27  | 4.962e+00  |     0.04    |
    # |   8 | 1.594e-01 |   2.36  | 1.323e+00  |     1.91    |
    # |  16 | 1.177e-02 |   3.76  | 3.986e-01  |     1.73    |
    # |  32 | 1.122e-03 |   3.39  | 1.124e-01  |     1.83    |
    # |  64 | 1.616e-04 |   2.80  | 4.679e-02  |     1.26    |
    # | 128 | 3.333e-05 |   2.28  | 2.231e-02  |     1.07    |
    # +-----+-----------+---------+------------+-------------+