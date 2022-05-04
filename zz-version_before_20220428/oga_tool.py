import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# import gc

np.set_printoptions(precision=16) 
torch.set_default_dtype(torch.float64)

## =====================================
## standard DNN model
## single neuron and activations
## tensor

class ReluDerivative(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.n = n
        self.m = m
        self.fact = np.math.factorial(n) / np.math.factorial(n-m)
        
    # d^m/dx^m [relu(x)^n], 
    def forward(self, out):
        out = F.relu(out).pow(self.n-self.m) * self.fact
        return out

class Bspline(nn.Module):
    def __init__(self, wei, deg):
        super().__init__()
        self.w = wei
        self.p = deg
    
    def forward(self, x):
        out = 0;
        for i in range(len(self.w)):
            out = out + self.w[i]*F.relu(i - x).pow(self.p)
        out = out * (self.p+1)
        return out  

class GAnet(nn.Module):
    def __init__(self, in_dim, width, out_dim, sigma):
        super(GAnet, self).__init__()
        self.layer1 = nn.Linear(in_dim,width)
        self.layer2 = nn.Linear(width,out_dim)
        self.F = sigma
        
    def forward(self, out):
        out = self.layer1(out)
        out = self.F(out)
        out = self.layer2(out)
        return out
    
class GAneuron(nn.Module):
    def __init__(self, in_dim, sigma):
        super(GAneuron, self).__init__()
        self.layer = nn.Linear(in_dim, 1)
        self.F = sigma
        
    def forward(self, out):
        out = self.layer(out)
        out = self.F(out)
        return out
        
## =====================================
## exact solution of (-Δ)^(m)u + u = f, 1d
## corresponding energy functional
## tensor
    
class Energy2m1d:
            
    def __init__(self, device, quad_info, activation):

        self.h = quad_info.h.to(device)
        self.wei = quad_info.wei.to(device)
        self.pts = quad_info.pts.to(device).requires_grad_()
        self.sigma = activation.get(0)
        self.dsigma = activation.get(1)
        self.device = device
        
    class Struct(object):
        def __init__(self, fun, dfun, vall2, valhm):
            self.fun = fun
            self.dfun = dfun
            self.vall2 = vall2
            self.valhm = valhm
        
    def theta2param(self, theta):
        ww, bb = torch.meshgrid(theta[0],theta[1])
        w = ww.reshape(-1,1)
        b = bb.reshape(-1,1)
        param = torch.cat([w,b], dim=1)
        return param
        
    def bilinear_form(self, func):
        pts = self.pts.clone()
        pts = pts.requires_grad_()
        fun = func(pts)
        y = fun.sum()
        grad_pts = torch.autograd.grad(outputs=y, inputs=pts) #, create_graph=True)
        fun = fun.data
        dfun = grad_pts[0]
        
        vall2 = fun.pow(2) * self.wei
        vall2 = vall2.sum() * (self.h/2)
        valhm = dfun.pow(2) * self.wei
        valhm = valhm.sum() * (self.h/2)
        return self.Struct(fun, dfun, vall2, valhm)
    
    def energy(self, func_info, theta, rhs):
        
        fun = func_info.fun
        dfun = func_info.dfun
    
        ww, bb = torch.meshgrid(theta[0],theta[1])
        w = ww.reshape(-1,1)
        b = bb.reshape(-1,1)
        A = torch.cat([w,b], dim=1)
        ones = torch.ones(len(self.pts),1).to(self.device)  
        B = torch.cat([self.pts, ones], dim=1).t()
        Core = torch.mm(A, B)
        g = self.sigma(Core)
        dg = self.dsigma(Core)
        fg = torch.mm(g, rhs*self.wei)
        ug = torch.mm(g, fun*self.wei)
        dudg = torch.mm(dg, dfun*self.wei) * w
        torch.cuda.empty_cache()
        energy_val = -(1/2)*((dudg+ug-fg)*self.h/2).pow(2)
        return energy_val
    
    def energy1(self, func_info, A, rhs):
        
        fun = func_info.fun
        dfun = func_info.dfun
    
        # A is theta, column shape
        w = A[:,0:1]
        ones = torch.ones(len(self.pts),1).to(self.device)  
        B = torch.cat([self.pts, ones], dim=1).t()
        Core = torch.mm(A, B)
        g = self.sigma(Core)
        dg = self.dsigma(Core)
        fg = torch.mm(g, rhs*self.wei)
        ug = torch.mm(g, fun*self.wei)
        dudg = torch.mm(dg, dfun*self.wei) * w
        torch.cuda.empty_cache()
        energy_val = -(1/2)*((dudg+ug-fg)*self.h/2).pow(2)
        return energy_val
    
    def stiff_mat(self, parameters, core, rhs):
        
        w = parameters[:,0:1]
        # b = parameters[:,1:2]
            
        g = self.sigma(core)
        dg = self.dsigma(core)
        g1 = g * self.wei.t()
        dg1 = dg * self.wei.t()
        G = torch.mm(g, g1.t()) * self.h / 2
        dG = torch.mm(dg, dg1.t()) * torch.mm(w, w.t()) * self.h / 2
        Gk = dG + G
        
        f = rhs * self.wei
        bk = torch.mm(g, f) * self.h / 2
        return Gk, bk
        
    
class Energy4m1d:
    def __init__(self, device, quad_info, activation):
        pass 
    
    def energy():
        pass
    
    def stiff_mat():
        pass
    
    
    
## =====================================
## exact solution of (-Δ)^(m)u + u = f, 2d
## corresponding energy functional
## tensor


class Energy0m2d:
    
    def __init__(self, device, quad_info, activation):

        self.h = quad_info.h.to(device)
        self.wei = quad_info.wei.to(device)
        self.pts = quad_info.pts.to(device).requires_grad_()
        self.sigma = activation.get(0)
        self.dsigma = activation.get(1)
        self.device = device

    class Struct(object):
        def __init__(self, fun, dfun, vall2, valhm):
            self.fun = fun
            self.dfun = dfun
            self.vall2 = vall2
            self.valhm = valhm
            
    def theta2param(self, theta):
        ww, bb = torch.meshgrid(theta[0],theta[1])
        w1 = torch.cos(ww).reshape(-1,1)
        w2 = torch.sin(ww).reshape(-1,1)
        b = bb.reshape(-1,1)
        param = torch.cat([w1,w2,b], dim=1)
        return param

    def bilinear_form(self, func):
        pts = self.pts.clone()
        pts = pts.requires_grad_()
        fun = func(pts).data
        dfun = torch.zeros_like(fun)
        vall2 = fun.pow(2) * self.wei
        vall2 = vall2.sum() * (self.h[0]/2) * (self.h[1]/2)
        valhm = torch.zeros_like(vall2)
        return self.Struct(fun, dfun, vall2, valhm)
            
    def energy(self, func_info, theta, rhs):
        fun = func_info.fun
        ww, bb = torch.meshgrid(theta[0],theta[1])
        w1 = torch.cos(ww).reshape(-1,1)
        w2 = torch.sin(ww).reshape(-1,1)
        b = bb.reshape(-1,1)
        A = torch.cat([w1,w2,b], dim=1)
        ones = torch.ones(len(self.pts),1).to(self.device)  
        B = torch.cat([self.pts, ones], dim=1).t()
        
        ug = torch.zeros(len(A),1)
        fg = torch.zeros(len(A),1)
        fun = fun*self.wei
        rhs = rhs*self.wei
        
        size = 2
        num = int(len(A)/size)
        for i in range(num):
            core = torch.mm(A[i*size:(i+1)*size,:], B)
            core = self.sigma(core)
            ug[i*size:(i+1)*size] = torch.mm(core, fun)
            fg[i*size:(i+1)*size] = torch.mm(core, rhs)
            torch.cuda.empty_cache()
            
        energy_val = -(1/2)*((ug-fg)*(self.h[0]/2)*(self.h[1]/2)).pow(2)
        return energy_val
    
    
    def energy1(self, func_info, theta, rhs):
        fun = func_info.fun
        ww, bb = torch.meshgrid(theta[0],theta[1])
        w1 = torch.cos(ww).reshape(-1,1)
        w2 = torch.sin(ww).reshape(-1,1)
        b = bb.reshape(-1,1)
        A = torch.cat([w1,w2,b], dim=1)
        ones = torch.ones(len(self.pts),1).to(self.device)  
        B = torch.cat([self.pts, ones], dim=1).t()
        Core = torch.mm(A, B)
        g = self.sigma(Core)
        fg = torch.mm(g, rhs*self.wei)
        ug = torch.mm(g, fun*self.wei)
        torch.cuda.empty_cache()
        energy_val = -(1/2)*((ug-fg)*(self.h[0]/2)*(self.h[1]/2)).pow(2)
        return energy_val

    def stiff_mat(self, parameters, core, rhs):
        # w1 = parameters[:,0:1]
        # w2 = parameters[:,1:2]
        # b = parameters[:,1:2]
        g = self.sigma(core)
        f = rhs * self.wei
        g1 = g * self.wei.t()
        Gk = torch.mm(g, g1.t()) * (self.h[0]/2) * (self.h[1]/2)
        bk = torch.mm(g, f) * (self.h[0]/2) * (self.h[1]/2)
        return Gk, bk

    
class Energy2m2d:
    
    class Struct(object):
        def __init__(self, dx_func, dy_func, vall2, valhm):
            self.dx_func = dx_func
            self.dy_func = dy_func
            self.vall2 = vall2
            self.valhm = valhm
    
    def __init__(self, quad_info):
        self.pts = quad_info.pts.requires_grad_()
        self.wei = quad_info.wei
        self.h = quad_info.h
    
    def bilinear_form(self, func):
        y = func(self.pts).sum()
        grad_pts = torch.autograd.grad(outputs=y, inputs=self.pts, create_graph=True)
        dx_func = grad_pts[0][:, 0:1]
        dy_func = grad_pts[0][:, 1:2]
        vall2 = func(self.pts).pow(2) * self.wei
        vall2 = vall2.sum() * (self.h[0]/2) * (self.h[1]/2)
        valhm = (dx_func.pow(2)+dy_func.pow(2)) * self.wei
        valhm = valhm.sum() * (self.h[0]/2) * (self.h[1]/2)
        return self.Struct(dx_func, dy_func, vall2, valhm)
    
    def energy():
        pass
    
    def stiff_mat():
        pass
    
class Energy4m2d:
    def __init__(self):
        pass 
    
    def energy():
        pass
    
    def stiff_mat():
        pass
    
    
## =====================================
## exact solution of (-Δ)^(m)u + u = f, 3d
## tensor
    
class Energy2m3d:
    def __init__(self):
        pass 
    
    def energy():
        pass
    
    def stiff_mat():
        pass
        
    
## =====================================
## quadrature samples
## numpy    

class QuadGauss:
    
    class Struct(object):
        def __init__(self, pts, wei, h):
            self.pts = pts
            self.wei = wei
            self.h = h
    
    def __init__(self, order):
        if order == 0:
            self.pts = 0
            self.wei = 2
        else:
            h1 = np.linspace(0,order,order+1)
            h2 = np.linspace(0,order,order+1) * 2
            J = 2*(h1[1:order+1]**2) / (h2[0:order]+2) * \
                np.sqrt(1/(h2[0:order]+1)/(h2[0:order]+3))
            J = np.diag(J,1) + np.diag(J,-1)
            D, V = np.linalg.eig(J)
            self.pts = D
            self.wei = 2*V[0,:]**2
            self.pts = self.pts.reshape(D.shape[0],1)
            self.wei = self.wei.reshape(D.shape[0],1)
            
    def quadpts_1d(self, cube, h):
        N = int((cube[0][1] - cube[0][0])/h) + 1
        x = np.linspace(cube[0][0], cube[0][1], N)
        xpt = (self.pts*h + x[0:-1] + x[1:]) / 2
        wei = np.tile(self.wei, xpt.shape[1])
        pts = xpt.flatten().reshape(-1,1)
        wei = wei.flatten().reshape(-1,1)
        return self.Struct(pts, wei, h)
    
    def quadpts_2d(self, cube, h):
        
        Nx = int((cube[0][1] - cube[0][0])/h[0]) + 1
        Ny = int((cube[1][1] - cube[1][0])/h[1]) + 1
        x = np.linspace(cube[0][0], cube[0][1], Nx)
        y = np.linspace(cube[1][0], cube[1][1], Ny)
        xp = (self.pts*h[0] + x[0:-1] + x[1:]) / 2
        yp = (self.pts*h[1] + y[0:-1] + y[1:]) / 2
        xpt, ypt = np.meshgrid(xp.flatten(), yp.flatten())
        xpt = xpt.flatten().reshape(-1,1)
        ypt = ypt.flatten().reshape(-1,1)
        pts = np.concatenate((xpt,ypt), axis=1)
        
        wei_x = np.tile(self.wei, xp.shape[1])
        wei_y = np.tile(self.wei, yp.shape[1])
        wei_x, wei_y = np.meshgrid(wei_x.flatten(), wei_y.flatten())
        wei_x = wei_x.flatten().reshape(-1,1)
        wei_y = wei_y.flatten().reshape(-1,1)
        wei = wei_x * wei_y

        return self.Struct(pts, wei, h)
    
    def quadpts_3d(self, cube, h):
        
        Nx = int((cube[0][1] - cube[0][0])/h[0]) + 1
        Ny = int((cube[1][1] - cube[1][0])/h[1]) + 1
        Nz = int((cube[2][1] - cube[2][0])/h[2]) + 1
        x = np.linspace(cube[0][0], cube[0][1], Nx)
        y = np.linspace(cube[1][0], cube[1][1], Ny)
        z = np.linspace(cube[2][0], cube[2][1], Nz)
        xp = (self.pts*h[0] + x[0:-1] + x[1:]) / 2
        yp = (self.pts*h[1] + y[0:-1] + y[1:]) / 2
        zp = (self.pts*h[2] + z[0:-1] + z[1:]) / 2
        xpt, ypt, zpt = np.meshgrid(xp.flatten(), yp.flatten(), zp.flatten())
        xpt = xpt.flatten().reshape(-1,1)
        ypt = ypt.flatten().reshape(-1,1)
        zpt = zpt.flatten().reshape(-1,1)
        pts = np.concatenate((xpt,ypt,zpt), axis=1)
                       
        wei_x = np.tile(self.wei, xp.shape[1])
        wei_y = np.tile(self.wei, yp.shape[1])
        wei_z = np.tile(self.wei, zp.shape[1])
        wei_x, wei_y, wei_z = np.meshgrid(wei_x.flatten(), wei_y.flatten(), wei_z.flatten())
        wei_x = wei_x.flatten().reshape(-1,1)
        wei_y = wei_y.flatten().reshape(-1,1)
        wei_z = wei_z.flatten().reshape(-1,1)
        wei = wei_x * wei_y * wei_z
        
        return self.Struct(pts, wei, h)
        
        
            
    
    













    