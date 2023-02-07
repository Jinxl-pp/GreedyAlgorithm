import sys
sys.path.append('../')

import time
import torch
import numpy as np
from quadrature import monte_carlo_quadrature as mc
from quadrature import gauss_legendre_quadrature as gl

# precision settings
torch.set_printoptions(precision=6)
data_type = torch.float64
torch.set_default_dtype(data_type)

# device settings
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(use_gpu)

# pi
pi = torch.pi


def rerror(a,b):
    return np.abs(a-b) / np.abs(b)

def func_1d(x):
    x = x[:,0]
    return torch.cos(3.5*pi*x)

def func_2d(x):
    x1 = x[:,0]#.reshape(-1,1)
    x2 = x[:,1]#.reshape(-1,1)
    return torch.cos(3.5*pi*x1) * torch.cos(3.5*pi*x2)

def func_3d(x):
    x1 = x[:,0]#.reshape(-1,1)
    x2 = x[:,1]#.reshape(-1,1)
    x3 = x[:,2]#.reshape(-1,1)
    return torch.cos(3.5*pi*x1) * torch.cos(3.5*pi*x2) * torch.cos(3.5*pi*x3)

def func_one(x):
    pass

def real_integral(case):
    
    integral = {
        1: {'func': func_1d, 'value': -0.181891363533595, 'domain': np.array([[-1.,1.]])},
        2: {'func': func_2d, 'value':  0.033084468128110, 'domain': np.array([[-1.,1.],[-1.,1.]])},
        3: {'func': func_3d, 'value': -0.006017779019606, 'domain': np.array([[-1.,1.],[-1.,1.],[-1.,1.]])},
        4: {'func': func_one, 'value':  3.141592653589793, 'domain': np.array([0.,0.,1.])},
        5: {'func': func_one, 'value':  4.188790204786391, 'domain': np.array([0.,0.,0.,1.])}
        }
    return integral.get(case)



if __name__ == '__main__':
    
    index = 1
    gl_quad = gl.GaussLegendreDomain(index, device)
    mc_quad = mc.MonteCarloDomain(device)
    
    # 1D G-L quadrature test
    nsamples = 1000
    h = 1 / nsamples
    integral = real_integral(1)
    func = integral.get('func')
    value = integral.get('value')
    domain = integral.get('domain')
    
    start_0 = time.time()
    data = gl_quad.interval_quadpts(domain, h)
    end_0 = time.time()
    
    start_1 = time.time()
    funeval = func(data.quadpts)
    end_1 = time.time()
    
    start_2 = time.time()
    value_num = torch.dot(funeval, data.weights) * torch.prod(data.h)
    end_2 = time.time()
    
    print(rerror(value_num, value))
    print('generation time = {:.6f}s'.format(end_0 - start_0))
    print('evaluation time = {:.6f}s'.format(end_1 - start_1))
    print('qudrature time = {:.6f}s'.format(end_2 - start_2))
    
    # 2D G-L quadrature test
    nsamples = 100
    h = 1 / nsamples
    h = np.array([h,h])
    integral = real_integral(2)
    func = integral.get('func')
    value = integral.get('value')
    domain = integral.get('domain')
    
    start_0 = time.time()
    data = gl_quad.rectangle_quadpts(domain, h)
    end_0 = time.time()
    
    start_1 = time.time()
    funeval = func(data.quadpts)
    end_1 = time.time()
    
    start_2 = time.time()
    value_num = torch.dot(funeval, data.weights) * torch.prod(data.h)
    end_2 = time.time()
    
    print(rerror(value_num, value))
    print('generation time = {:.6f}s'.format(end_0 - start_0))
    print('evaluation time = {:.6f}s'.format(end_1 - start_1))
    print('qudrature time = {:.6f}s'.format(end_2 - start_2))
    
    # 3D G-L quadrature test
    nsamples = 100
    h = 1 / nsamples
    h = np.array([h,h,h])
    integral = real_integral(3)
    func = integral.get('func')
    value = integral.get('value')
    domain = integral.get('domain')
    
    start_0 = time.time()
    data = gl_quad.cuboid_quadpts(domain, h)
    end_0 = time.time()
    
    start_1 = time.time()
    funeval = func(data.quadpts)
    end_1 = time.time()
    
    start_2 = time.time()
    value_num = torch.dot(funeval, data.weights) * torch.prod(data.h)
    end_2 = time.time()
    
    print(rerror(value_num, value))
    print('generation time = {:.6f}s'.format(end_0 - start_0))
    print('evaluation time = {:.6f}s'.format(end_1 - start_1))
    print('qudrature time = {:.6f}s'.format(end_2 - start_2))
    
    # 1D M-C quadrature test
    
    # 2D M-C quadrature test
    
    # 3D M-C quadrature test
    
    # 2D M-C quadrature test on circle
    
    # 3D M-C quadrature test on sphere


# test log (3D quadrature, h = 1/100):
# generation of quadrature, large time consumption, 3.9081s;
# evaluation at quadrature points, large time consumption, 0.8002s;
# quadrature rule, small time consumption, 0.0946s.
# suggestions:
# generate quadrature points only once;
# try to limitate the number of evaluation;
# optimize the evaluation process as best as you can;