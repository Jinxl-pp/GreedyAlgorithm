import sys
sys.path.append('../')

import time
import torch
import numpy as np

from greedy.quadrature import monte_carlo_quadrature as mc
from greedy.quadrature import gauss_legendre_quadrature as gl
from greedy.quadrature import quasi_monte_carlo_quadrature as qmc

# precision settings
torch.set_printoptions(precision=6)
data_type = torch.float64
torch.set_default_dtype(data_type)

# device settings
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print("GPU =", use_gpu)

# pi
pi = torch.pi


def rerror(a,b):
    return np.abs(a-b) / np.abs(b)

def func_1d(x):
    x = x[:,0].reshape(-1,1)
    return torch.cos(3.5*pi*x)

def func_2d(x):
    x1 = x[:,0].reshape(-1,1)
    x2 = x[:,1].reshape(-1,1)
    return torch.cos(3.5*pi*x1) * torch.cos(3.5*pi*x2)

def func_3d(x):
    x1 = x[:,0].reshape(-1,1)
    x2 = x[:,1].reshape(-1,1)
    x3 = x[:,2].reshape(-1,1)
    return torch.cos(3.5*pi*x1) * torch.cos(3.5*pi*x2) * torch.cos(3.5*pi*x3)

def func_one(x):
    pass

def real_integral(case):
    
    integral = {
        1: {'func': func_1d, 'value': -0.181891363533595, 'domain': np.array([[-1.,1.]])},
        2: {'func': func_2d, 'value':  0.033084468128110, 'domain': np.array([[-1.,1.],[-1.,1.]])},
        3: {'func': func_3d, 'value': -0.006017779019606, 'domain': np.array([[-1.,1.],[-1.,1.],[-1.,1.]])},
        # 4: {'func': func_one, 'value':  3.141592653589793, 'domain': np.array([0.,0.,1.])},
        # 5: {'func': func_one, 'value':  4.188790204786391, 'domain': np.array([0.,0.,0.,1.])}
        }
    return integral.get(case)

def show_error_GL(sampling, h, case):
    
    integral = real_integral(case)
    func = integral.get('func')
    value = integral.get('value')
    domain = integral.get('domain')
    
    start_0 = time.time()
    data = sampling(domain, h)
    end_0 = time.time()
    
    start_1 = time.time()
    funeval = func(data.quadpts)
    print(data.quadpts.shape[0])
    end_1 = time.time()
    
    start_2 = time.time()
    value_num = torch.sum(funeval * data.weights) * data.area
    end_2 = time.time()
    
    print('{:d}D G-L quadrature error = {:.6e}'.format(case, rerror(value_num, value)))
    print('generation time = {:.6f}s'.format(end_0 - start_0))
    print('evaluation time = {:.6f}s'.format(end_1 - start_1))
    print('quadrature time = {:.6f}s'.format(end_2 - start_2))
    print('number of samples = {:d}'.format(data.quadpts.shape[0]))
    

def show_error_MC(sampling, nsamples, case):
    
    integral = real_integral(case)
    func = integral.get('func')
    value = integral.get('value')
    domain = integral.get('domain')
    
    start_0 = time.time()
    data = sampling(domain, nsamples)
    end_0 = time.time()
    
    start_1 = time.time()
    funeval = func(data.quadpts)
    end_1 = time.time()
    
    start_2 = time.time()
    value_num = torch.sum(funeval * data.weights) * data.area
    end_2 = time.time()
    
    print('{:d}D M-C quadrature error = {:.6e}'.format(case, rerror(value_num, value)))
    print('generation time = {:.6f}s'.format(end_0 - start_0))
    print('evaluation time = {:.6f}s'.format(end_1 - start_1))
    print('qudrature time = {:.6f}s'.format(end_2 - start_2))
    print('number of samples = {:d}'.format(data.quadpts.shape[0]))
    
    
def show_error_QMC(sampling, nsamples, case):
    
    integral = real_integral(case)
    func = integral.get('func')
    value = integral.get('value')
    domain = integral.get('domain')
    
    start_0 = time.time()
    data = sampling(domain, nsamples)
    end_0 = time.time()
    
    start_1 = time.time()
    funeval = func(data.quadpts)
    end_1 = time.time()
    
    start_2 = time.time()
    value_num = torch.sum(funeval * data.weights) * data.area
    end_2 = time.time()
    
    print('{:d}D Q-M-C quadrature error = {:.6e}'.format(case, rerror(value_num, value)))
    print('generation time = {:.6f}s'.format(end_0 - start_0))
    print('evaluation time = {:.6f}s'.format(end_1 - start_1))
    print('qudrature time = {:.6f}s'.format(end_2 - start_2))
    print('number of samples = {:d}'.format(data.quadpts.shape[0]))



if __name__ == '__main__':
    
    index = 1
    mc_quad = mc.MonteCarloDomain(device)
    gl_quad = gl.GaussLegendreDomain(index, device)
    qmc_quad = qmc.QuasiMonteCarloQuadrature(device)
    
    # # 1D G-L quadrature test
    # case = 1
    # N = 1000
    # h = np.array([1/N])
    # sampling = gl_quad.interval_quadpts
    # show_error_GL(sampling, h, case)
    
    # # 2D G-L quadrature test
    # case = 2
    # N = 400
    # h = np.array([1/N, 1/N])
    # sampling = gl_quad.rectangle_quadpts
    # show_error_GL(sampling, h, case)
    
    # # 3D G-L quadrature test
    # case = 3
    # N = 100
    # h = np.array([1/N, 1/N, 1/N])
    # sampling = gl_quad.cuboid_quadpts
    # show_error_GL(sampling, h, case)
    
    # # 1D M-C quadrature test
    # case = 1
    # nsamples = int(1e+8)
    # sampling = mc_quad.interval_samples
    # show_error_MC(sampling, nsamples, case)
    
    # # 2D M-C quadrature test
    # case = 2
    # nsamples = int(1e+8)
    # sampling = mc_quad.rectangle_samples
    # show_error_MC(sampling, nsamples, case)

    # # 3D M-C quadrature test
    # case = 3
    # nsamples = int(1e+8)
    # sampling = mc_quad.cuboid_samples
    # show_error_MC(sampling, nsamples, case)
    
    # # 1D Q-M-C quadrature test
    # case = 1
    # nsamples = int(1e+6)
    # sampling = qmc_quad.n_rectangle_samples
    # show_error_QMC(sampling, nsamples, case)
    
    # # 2D Q-M-C quadrature test
    # case = 2
    # nsamples = int(1e+6)
    # sampling = qmc_quad.n_rectangle_samples
    # show_error_QMC(sampling, nsamples, case)
    
    # # 3D Q-M-C quadrature test
    # case = 3
    # nsamples = int(1e+6)
    # sampling = qmc_quad.n_rectangle_samples
    # show_error_QMC(sampling, nsamples, case)
    
    # 2D M-C quadrature test on circle
    
    # 3D M-C quadrature test on sphere


## ============= end ============== ##


## ============= TEST LOG (3D quadrature, h = 1/100):
# generation of quadrature, large time consumption, 3.9081s;
# evaluation at quadrature points, large time consumption, 0.8002s;
# quadrature rule, small time consumption, 0.0946s.
# 
# SUGGESTIONS:
# generate quadrature points only once;
# try to limitate the number of evaluation;
# optimize the evaluation process as best as you can;


## ============= TEST LOG (3D qmc quadrature):
# 2e5 qmc samples can out-perform 1e8 mc samples;
# qmc generates slower than mc (1e8 pts, 63.7431s to 9.5248s);
# qmc gets much smaller error than mc (1e8 pts, 7.73e-05 to 2.79e-02)
# 
# SUGGESTIONS:
# use qmc in higher dimension;
# generate qmc only once;
# store qmc samples if possible;