import sys
sys.path.append('../')

import torch
import numpy as np

from greedy.quadrature import gauss_legendre_quadrature as gl
from greedy.pde import cos1d, cos2d, poly2d, cos3d

# device settings
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(use_gpu)

if __name__ == '__main__':
    
    index = 1
    gl_quad = gl.GaussLegendreDomain(index, device)
    
    domain_1d = np.array([[-1.,1.]])
    domain_2d = np.array([[-1.,1.],[-1.,1.]])
    domain_3d = np.array([[-1.,1.],[-1.,1.],[-1.,1.]])
    
    nsamples = 10
    h = 1 / nsamples
    h1 = np.array([h])
    h2 = np.array([h,h])
    h3 = np.array([h,h,h])
    
    # 1D pde
    data_1 = gl_quad.interval_quadpts(domain_1d, h1)
    pde = cos1d.DataCos1m1dDirichletBC()
    solu = pde.solution(data_1.quadpts)
    grad = pde.gradient(data_1.quadpts)
    rhs = pde.source(data_1.quadpts)
    print(solu.shape)
    print(grad.shape)
    print(rhs.shape)
    
    # 2D pde
    data_2 = gl_quad.rectangle_quadpts(domain_2d, h2)
    pde = poly2d.DataPoly2m2dNeumannBC()
    solu = pde.solution(data_2.quadpts)
    grad = pde.gradient(data_2.quadpts)
    hess = pde.hessian(data_2.quadpts)
    rhs = pde.source(data_2.quadpts)
    print(solu.shape)
    print(grad.shape)
    print(hess.shape)
    print(rhs.shape)
    
    # 3D pde
    data_3 = gl_quad.cuboid_quadpts(domain_3d, h3)
    pde = cos3d.DataCos1m3d_NeumannBC()
    solu = pde.solution(data_3.quadpts)
    grad = pde.gradient(data_3.quadpts)
    rhs = pde.source(data_3.quadpts)
    print(solu.shape)
    print(grad.shape)
    print(rhs.shape)
    