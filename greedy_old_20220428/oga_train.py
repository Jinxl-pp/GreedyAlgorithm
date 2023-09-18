import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.parameter import Parameter

import gc
import time
from prettytable import PrettyTable

import oga_pde as op
import oga_tool as ot

np.set_printoptions(precision=16) 
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=6)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(use_gpu)


def quad_pts(cube, dim, order, h):

    quad_set = {
        1: ot.QuadGauss(order).quadpts_1d,
        2: ot.QuadGauss(order).quadpts_2d,
        3: ot.QuadGauss(order).quadpts_3d
        }

    GQ = quad_set.get(dim)
    quad_info = GQ(cube, h)
    quad_info.pts = torch.from_numpy(quad_info.pts).type(torch.float64).to(device)
    quad_info.wei = torch.from_numpy(quad_info.wei).type(torch.float64).to(device)
    quad_info.h = torch.from_numpy(quad_info.h).type(torch.float64).to(device)
    
    return quad_info
    
def oga_print(num_neuron, errl2_record, errhm_record, title_string):
    
    N = int(np.log2(num_neuron))
    order = [2**(i+1) - 1 for i in range(N)]
    
    errl2_record = errl2_record[order].cpu()
    errhm_record = errhm_record[order].cpu()
    ratel2_record = np.log2(errl2_record[:-1] / errl2_record[1:])
    ratehm_record = np.log2(errhm_record[:-1] / errhm_record[1:])
    ratel2_record = np.concatenate((np.array([[0.]]),ratel2_record), axis = 0)
    ratehm_record = np.concatenate((np.array([[0.]]),ratehm_record), axis = 0)
    
    table = PrettyTable(['N','err_l2','rate_l2','err_energy','rate_energy'])
    table.align['N'] = 'r'
    table.align['err_l2'] = 'c'
    table.align['rate_l2'] = 'c'
    table.align['err_energy'] = 'c'
    table.align['rate_energy'] = 'c'
    
    for i in range(N):
        if i == 0:
            rate1 = '-'
        else:
            rate1 = '{:.2f}'.format(ratel2_record[i].item())
        if i == 0:
            rate2 = '-'
        else:
            rate2 = '{:.2f}'.format(ratehm_record[i].item())
        table.add_row([order[i]+1, '{:.6e}'.format(errl2_record[i].item()), rate1, \
                                   '{:.6e}'.format(errhm_record[i].item()), rate2])
    print(table.get_string(title = title_string))



def param_mesh(domain_b, mesh_size, dim):
    d1 = domain_b[0][0]
    d2 = domain_b[1][0]
    N = int((d2-d1)/mesh_size) + 1
    if dim == 1:
        w = torch.tensor([-1.,1.]).to(device)
        b = torch.linspace(d1,d2,N).to(device)
        theta = w, b
    elif dim == 2:
        t = torch.linspace(0,2*torch.pi,N).to(device)
        b = torch.linspace(d1,d2,N).to(device)
        theta = t, b
    elif dim == 3:
        p = torch.linspace(0,2*torch.pi,N).to(device)
        t = torch.linspace(0,torch.pi,N).to(device)
        b = torch.linspace(d1,d2,N).to(device)
        theta = p, t, b
    return theta
    

def ind2sub_tensor(idx, dim):
    if len(dim) == 2:
        b = idx % dim[1]
        a = ((idx-b) / dim[1]).long()
        sub = torch.cat([a,b], dim=1)
    elif len(dim) == 3:
        c = idx % dim[2]
        b = ((idx-c)/dim[2] % dim[1]).long()
        a = (((idx-c)/dim[2]-b)/dim[1]).long()
        sub = int(a), int(b), int(c) 
    return sub


def initial_guess(theta_pre, energy, dim, best_k):
    _, idx = torch.sort(energy, descending=False, dim=0)
    if dim == 1:
        w = theta_pre[0]
        b = theta_pre[1]
        theta = torch.meshgrid(w,b)
        tensor_dim = theta[0].shape
        sub = ind2sub_tensor(idx[0:best_k], tensor_dim)
        we = theta[0][sub[:,0:1],sub[:,1:2]]
        bi = theta[1][sub[:,0:1],sub[:,1:2]]
        theta_set = torch.cat([we,bi], dim=1).to(device)
    elif dim == 2:
        t = theta_pre[0]
        b = theta_pre[1]
        theta = torch.meshgrid(t,b)
        tensor_dim = theta[0].shape
        sub = ind2sub_tensor(idx[0:best_k], tensor_dim)
        th = theta[0][sub[:,0:1],sub[:,1:2]]
        bi = theta[1][sub[:,0:1],sub[:,1:2]]
        theta_set = torch.cat([th,bi], dim=1).to(device)
    elif dim == 3:
        p = theta_pre[0]
        t = theta_pre[1]
        b = theta_pre[2]
        theta = torch.meshgrid(p,t,b)
        tensor_dim = theta[0].shape
        sub = ind2sub_tensor(idx[0:best_k], tensor_dim)
        ph = theta[0][sub[:,0:1],sub[:,1:2],sub[:,2:3]]
        th = theta[1][sub[:,0:1],sub[:,1:2],sub[:,2:3]]
        bi = theta[2][sub[:,0:1],sub[:,1:2],sub[:,2:3]]
        theta_set = torch.cat([ph,th,bi], dim=1).to(device)
    return theta_set



def error_norm(energy_func, quad_info, solution, dnn):
    
    def err(pts):
        return solution.target(pts) - dnn(pts)
    
    err_info = energy_func.bilinear_form(err)
    errl2 = torch.sqrt(err_info.vall2)
    errhm = torch.sqrt(err_info.valhm)
    return errl2, errhm


def zero_func(x):
    y = 0*x.sum(1)
    return y.reshape(-1,1) 


def get_hessian(loss, param):
    grad_loss = torch.autograd.grad(loss, param,retain_graph=True, create_graph=True)
    grad = grad_loss[0].sum()
    Hessian = torch.autograd.grad(grad, param)
    return grad_loss[0], Hessian[0]



def gd_backward(energy, dnn_info, theta, rhs, dim):
    
    a = theta[:,0:-1]
    b = theta[:,-1:]
    
    epochs = 200
    learning_rate = 1e-6
    if dim == 1:
        b = Parameter(b)
        optimizer = torch.optim.SGD([b], lr = learning_rate)
    else:
        a = Parameter(a)
        b = Parameter(b)
        optimizer = torch.optim.SGD([a,b], lr = learning_rate)
    for epoch in range(epochs):
        theta_set = torch.cat([a, b], dim=1)
        loss_set = energy(dnn_info, theta_set, rhs)
        loss = loss_set.sum()
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()  
        # if epoch % 50 == 0:
            # print('loss: {:.16e}'.format(loss.item()))
    return theta_set.detach(), loss_set.detach()


def newton_backward(energy, dnn_info, theta_set, rhs, dim):
    
    a = theta_set[:,0:-1]
    b = theta_set[:,-1:]
    
    if dim == 1:
        epochs = 3
        for epoch in range(epochs):
            b = Parameter(b)
            theta = torch.cat([a, b], dim=1)
            loss_set = energy(dnn_info, theta, rhs)
            loss = loss_set.sum()
            grad, hess = get_hessian(loss, b) 
            hess[hess<=1e-4] = 1e+10
            b = b - hess.pow(-1) * grad
            torch.cuda.empty_cache()
            # if epoch % 1 == 0:
                # print('loss: {:.16e}'.format(loss.item()))
        theta_set = torch.cat([a, b], dim=1)
    else:
        epochs = 3
        for epoch in range(epochs):
            theta = Parameter(theta)
            loss = energy(dnn_info, theta, rhs)
            grad, hess = get_hessian(loss, theta) 
            theta = theta - torch.linalg.inv(hess) @ grad
            torch.cuda.empty_cache()
            if epoch % 1 == 0:
                print('loss: {:.16e}'.format(loss.item()))
    return theta_set.detach(), loss_set.detach()   


def oga_train(eqn_order, num_neuron, quad_info, quad_info_test, activation, pde, dim, h, trainer):
    
    energy_set = {
        1: {
            2: {'train': ot.Energy2m1d(device, quad_info, activation), 'test': ot.Energy2m1d(device, quad_info_test, activation)},
            4: {'train': ot.Energy4m1d(device, quad_info, activation), 'test': ot.Energy4m1d(device, quad_info_test, activation)},
            },
        2: {
            0: {'train': ot.Energy0m2d(device, quad_info, activation), 'test': ot.Energy0m2d(device, quad_info_test, activation)},
            2: ot.Energy2m2d(quad_info),
            4: ot.Energy4m2d},
        3: {
            2: ot.Energy2m3d
            }
        }
    energy_dim = energy_set.get(dim)
    energy_func = energy_dim.get(eqn_order).get('train')
    energy_func_test = energy_dim.get(eqn_order).get('test')
    
    sol_set = {
        'cos1d': op.DataCos1d,
        'cos2d': op.DataCos2d,
        'cos3d': op.DataCos3d
        }
    solution = sol_set.get(pde)
    
    trainer_set = {
        'gd': gd_backward,
        'newton': newton_backward
        }
    trainer = trainer_set.get(trainer)
    
    pts = quad_info.pts
    pts.requires_grad = False
    sigma = activation.get(0)
    if eqn_order == 0:
        rhs = solution.target(pts)
    else:
        rhs = solution.right_hand_side(pts)

    best_k = 20
    domain_b = torch.tensor([[-2.],[2.]])
    theta_pre = param_mesh(domain_b, h, dim)
    
    core_mat = torch.zeros(num_neuron, len(pts)).to(device)
    parameters = torch.zeros(num_neuron, dim+1).to(device)
    coefficients = torch.zeros(1, num_neuron).to(device)
    errl2_record = torch.zeros(num_neuron, 1).to(device)
    errhm_record = torch.zeros(num_neuron, 1).to(device)
    
    for k in range(num_neuron):
        
        if k == 0:
            dnn = zero_func
        else:
            dnn = ot.GAnet(dim, k, 1, sigma)
            dnn.layer1.weight = Parameter(parameters[0:k, 0:dim])
            dnn.layer1.bias = Parameter(parameters[0:k, dim])
            dnn.layer2.weight = Parameter(coefficients[:,0:k])
            dnn.layer2.bias = Parameter(torch.zeros(1).to(device))
        
        errl2, errhm = error_norm(energy_func_test, quad_info_test, solution, dnn)
        errl2_record[k] = errl2
        errhm_record[k] = errhm
        print('----the N = {:.0f}-th neuron----'.format(k+1))
        print('L2-error: {:.6e}'.format(errl2.item()))
        print('Hm-error: {:.6e}'.format(errhm.item()))
        
        dnn_info = energy_func.bilinear_form(dnn)
        dnn_info_test = energy_func_test.bilinear_form(dnn)
        start = time.time()
    
        energy_val = energy_func.energy(dnn_info, theta_pre, rhs)
        theta_set = initial_guess(theta_pre, energy_val, dim, best_k)
        end = time.time()
        print('Initialguess-time = {:.4f}s'.format(end - start))
        
        theta_set, loss_set = trainer(energy_func.energy1, dnn_info, theta_set, rhs, dim)
        # Bool = (theta_set[:,1:2]<=2) * (theta_set[:,1:2]>=-2)
        # loss_set = loss_set[Bool] 
        # Bool = torch.cat([Bool, Bool], dim=1)
        # theta_set = theta_set[Bool].reshape(len(loss_set), 2)
        
        start = time.time()
        idx = loss_set.argmin()
        theta = theta_set[idx]
        end = time.time()
        print('Training-time = {:.8f}s'.format(end - start))

        
        start = time.time()
        parameter = energy_func.theta2param(theta).to(device)
        parameters[k,:] = parameter
        Ak = parameter.reshape(1,-1)
        ones = torch.ones(len(pts),1).to(device)
        Bk = torch.cat([pts, ones], dim=1)
        Ck = torch.mm(Ak, Bk.t())
        core_mat[k:k+1, :] = Ck
        core = core_mat[0:k+1, :]
        Gk, bk = energy_func.stiff_mat(parameters[0:k+1,:], core, rhs)
        coef = torch.solve(bk, Gk).solution # torch.linalg.solve(Gk, bk)
        coefficients[:, 0:k+1] = coef.reshape(1,-1).to(device)
        
        del energy_val, dnn_info
        del parameter, core, ones
        del Ak, Bk, Ck, Gk, bk
        torch.cuda.empty_cache()
        
        end = time.time()
        print('Projection-time = {:.4f}s'.format(end - start))
        
    dnn = ot.GAnet(dim, num_neuron, 1, sigma)
    dnn.layer1.weight = Parameter(parameters[:, 0:dim])
    dnn.layer1.bias = Parameter(parameters[:, dim])
    dnn.layer2.weight = Parameter(coefficients)
    dnn.layer2.bias = Parameter(torch.zeros(1))
    
    return dnn, errl2_record.data, errhm_record.data
    

if __name__ == "__main__":
    
    ## OGA implementation
    import argparse
    parser = argparse.ArgumentParser(description='greedy algorithm')
    parser.add_argument('--dim',type=int)
    parser.add_argument('--k',type=int, help='power of ReLU, the activation function', default='2')
    parser.add_argument('--num_neuron', type=int, default='64')
    parser.add_argument('--eqn_order',type=int, help='order of pde, use with 0, 2 or 4', default='2')
    parser.add_argument('--quad_order',type=int, help='order of quadrature rule', default='1')
    parser.add_argument('--h1',type=str, help='meshsize for quadrature samples of training', default='1/100')
    parser.add_argument('--h3',type=str, help='meshsize for quadrature samples of testing', default='1/1000')
    parser.add_argument('--h2',type=str, help='meshsize for paramter samples', default='1/100')
    parser.add_argument('--trainer',type=str, help='uese with gd or newton', default='gd')
    parser.add_argument('--pde', type=str, help='name of exact solution, use with cos1d, cos2d, cos3d, etc.')
    
    args = parser.parse_args()
    args.dim = 1
    args.k = 2
    args.num_neuron = 64
    args.eqn_order = 2
    args.h1 = '1/1000'
    args.h2 = '1/1000'
    args.h3 = '1/1000'
    args.pde = 'cos1d'
    args.h1 = eval(args.h1)
    args.h2 = eval(args.h2)
    args.h3 = eval(args.h3)
    
    # quadrature points    
    h_set = {
        1: {'train': np.array([args.h1]), 'test': np.array([args.h3])},
        2: {'train': np.array([args.h1, args.h1]), 'test': np.array([args.h3, args.h3])},
        3: {'train': np.array([args.h1, args.h1, args.h1]), 'test': np.array([args.h3, args.h3, args.h3])}
        }
    h = h_set.get(args.dim).get('train')
    h_test = h_set.get(args.dim).get('test')
    cube = np.array([[-1,1],[-1,1]])
    quad_info = quad_pts(cube, args.dim, args.quad_order, h)
    quad_info_test = quad_pts(cube, args.dim, args.quad_order, h_test)
    
    print(quad_info.pts.shape)
    
    # # begin OGA training
    activation = {
        0: ot.ReluDerivative(args.k,0),
        1: ot.ReluDerivative(args.k,1),
        2: ot.ReluDerivative(args.k,2)
        }
    start = time.time()
    dnn, errl2_record, errhm_record = \
        oga_train(args.eqn_order, args.num_neuron, quad_info, quad_info_test, activation, args.pde, args.dim, args.h2, args.trainer)
    end = time.time()
    print('Total-time = {:.4f}s'.format(end - start))
    
    # # numerical results
    title_string = 'OGA, ' + 'pde = ' + args.pde + ', relu_power = ' + '%d'%args.k
    oga_print(args.num_neuron, errl2_record, errhm_record, title_string)
  
    
  
    
## =====================================    
# Description of exact solutions

# 1. 1D functions 
#    1.1. u = cos(pix)
#         pde = 'cos1d'
#    1.2. u = ...
   
# 2. 2D functions
#    2.1. u = cos(2pix)cos(2piy)
#         pde = 'cos2d'
#    2.2. u = ...
   
# 3. 3D functions
#    3.1. u = cos(2pix)cos(2piy)cos(2piz)
#         pde = 'cos3d'
#    3.2. u = ...
        