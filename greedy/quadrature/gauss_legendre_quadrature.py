import numpy as np
from .quadrature import Quadrature

## =====================================
## quadrature points
## numpy    

class GaussLegendreDomain():
    
    def __init__(self, index, device):
        """ The cartesian Gauss-Lengendre quadrature information on the reference 1d interval [-1,1].
            INPUT:
                index:  the number of quadrature points -1,
                        where (index + 1) points gets a 
                        (2*index + 1) algebraic precision.
                device: cuda/cpu
        """
        
        self.device = device
        self.qtype = "GQ"
        
        if index == 0:
            self.quadpts = np.array([[0.]], dtype=np.float64)
            self.weights = np.array([[2.]], dtype=np.float64)
            
        else:
            h1 = np.linspace(0,index,index+1).astype(np.float64)
            h2 = np.linspace(0,index,index+1).astype(np.float64) * 2
            
            J = 2*(h1[1:index+1]**2) / (h2[0:index]+2) * \
                np.sqrt(1/(h2[0:index]+1)/(h2[0:index]+3))
            J = np.diag(J,1) + np.diag(J,-1)
            D, V = np.linalg.eig(J)
            
            self.quadpts = D
            self.weights = 2*V[0,:]**2
            self.quadpts = self.quadpts.reshape(D.shape[0],1)
            self.weights = self.weights.reshape(D.shape[0],1) / 2 
            
    def point_quadpts(self, point):
        """Returns the point itself.
            INPUT:
                point: np.array object
            
            OUTPUT:
                quadpts: point
                weights: 
        
        """
        quadpts = point.astype(np.float64)
        weights = np.ones_like(quadpts).astype(np.float64)
        h = np.array([1.], dtype=np.float64)
        return Quadrature(self.qtype, self.device, quadpts, weights, h)
        
            
    def interval_quadpts(self, interval, h):
        """ The Gauss-Lengendre quadrature information on a discretized mesh of 1d interval [a,b].
            Usually the mesh is uniform.
            INPUT:
                interval: np.array object
                       h: np.array object, mesh size 
            OUTPUT:
                 quadpts: npts-by-1
                 weights: npts-by-1
                       h:  shape=[1] 
            Examples
            -------
            interval = np.array([[0, 1]], dtype=np.float64)
            h = np.array([0.01], dtype=np.float64)
        """

        N = int((interval[0][1] - interval[0][0])/h) + 1
        xp = np.linspace(interval[0][0], interval[0][1], N)
        quadpts = (self.quadpts*h + xp[0:-1] + xp[1:]) / 2
        weights = np.tile(self.weights, quadpts.shape[1])
        quadpts = quadpts.flatten().reshape(-1,1)
        weights = weights.flatten().reshape(-1,1)
        return Quadrature(self.qtype, self.device, quadpts, weights, h)
    
    def interval_quadpts_exact(self, x, h):
        """ The Gauss-Lengendre quadrature information on a discretized mesh of 1d interval [a,b], 
            used for piecewise-polynomial integrand. Usually the mesh is non-uniform.
            INPUT:
                interval: np.array object
                       h: np.array object, mesh sizes
            OUTPUT:
                 quadpts: npts-by-1
                 weights: npts-by-1
                       h:  shape=[1]  
        """

        quadpts = (self.quadpts*h + x[0:-1] + x[1:]) / 2
        weights = np.tile(self.weights, quadpts.shape[1])
        quadpts = quadpts.flatten().reshape(-1,1)
        weights = weights.flatten().reshape(-1,1) 
        h = np.tile(h,(len(self.quadpts),1))
        h = h.flatten().reshape(-1,1)  
        return Quadrature(self.qtype, self.device, quadpts, weights, h)
    
    def rectangle_quadpts(self, rectangle, h):
        """ The Gauss-Lengendre quadrature information on a discretized mesh of 2d rectangle [a,b]*[c,d].
            Usually the mesh is uniform.
            INPUT:
                interval: np.array object
                       h: np.array object, mesh sizes
            OUTPUT:
                 quadpts: npts-by-2
                 weights: npts-by-1
                       h:  shape=[2]  
            Examples
            -------
            rectangle = np.array([[0, 1], [0, 1]], dtype=np.float64)
            h = np.array([0.01, 0.01], dtype=np.float64)
        """
        
        Nx = int((rectangle[0][1] - rectangle[0][0])/h[0]) + 1
        Ny = int((rectangle[1][1] - rectangle[1][0])/h[1]) + 1
        
        x = np.linspace(rectangle[0][0], rectangle[0][1], Nx)
        y = np.linspace(rectangle[1][0], rectangle[1][1], Ny)
        xp = (self.quadpts*h[0] + x[0:-1] + x[1:]) / 2
        yp = (self.quadpts*h[1] + y[0:-1] + y[1:]) / 2
        
        xpt, ypt = np.meshgrid(xp.flatten(), yp.flatten())
        xpt = xpt.flatten().reshape(-1,1)
        ypt = ypt.flatten().reshape(-1,1)
        quadpts = np.concatenate((xpt,ypt), axis=1)
        
        weights_x = np.tile(self.weights, xp.shape[1])
        weights_y = np.tile(self.weights, yp.shape[1])
        weights_x, weights_y = np.meshgrid(weights_x.flatten(), weights_y.flatten())
        
        weights_x = weights_x.flatten().reshape(-1,1)
        weights_y = weights_y.flatten().reshape(-1,1)
        weights = weights_x * weights_y

        return Quadrature(self.qtype, self.device, quadpts, weights, h)
    
    def cuboid_quadpts(self, cuboid, h):
        """ The Gauss-Lengendre quadrature information on a discretized mesh of 3d cuboid [a,b]*[c,d]*[e,f].
            Usually the mesh is uniform.
            INPUT:
                interval: np.array object
                       h: np.array object, mesh sizes 
            OUTPUT:
                 quadpts: npts-by-3
                 weights: npts-by-1
                       h:  shape=[3]
            Examples
            -------
            cuboid = np.array([[0, 1], [0, 1], [0, 1]], dtype=np.float64)
            h = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        """

        Nx = int((cuboid[0][1] - cuboid[0][0])/h[0]) + 1
        Ny = int((cuboid[1][1] - cuboid[1][0])/h[1]) + 1
        Nz = int((cuboid[2][1] - cuboid[2][0])/h[2]) + 1
        
        x = np.linspace(cuboid[0][0], cuboid[0][1], Nx)
        y = np.linspace(cuboid[1][0], cuboid[1][1], Ny)
        z = np.linspace(cuboid[2][0], cuboid[2][1], Nz)
        
        xp = (self.quadpts*h[0] + x[0:-1] + x[1:]) / 2
        yp = (self.quadpts*h[1] + y[0:-1] + y[1:]) / 2
        zp = (self.quadpts*h[2] + z[0:-1] + z[1:]) / 2
        
        xpt, ypt, zpt = np.meshgrid(xp.flatten(), yp.flatten(), zp.flatten())
        xpt = xpt.flatten().reshape(-1,1)
        ypt = ypt.flatten().reshape(-1,1)
        zpt = zpt.flatten().reshape(-1,1)
        quadpts = np.concatenate((xpt,ypt,zpt), axis=1)
                       
        weights_x = np.tile(self.weights, xp.shape[1])
        weights_y = np.tile(self.weights, yp.shape[1])
        weights_z = np.tile(self.weights, zp.shape[1])
        weights_x, weights_y, weights_z = np.meshgrid(weights_x.flatten(), weights_y.flatten(), weights_z.flatten())
        
        weights_x = weights_x.flatten().reshape(-1,1)
        weights_y = weights_y.flatten().reshape(-1,1)
        weights_z = weights_z.flatten().reshape(-1,1)
        weights = weights_x * weights_y * weights_z
        
        return Quadrature(self.qtype, self.device, quadpts, weights, h)

    
class GaussLegendreQuadrature(GaussLegendreDomain):
    """ The cartesian Gauss-Lengendre quadrature information on the boundary
        of a certain domain for 1D, 2D and 3D.
    """

    def __init__(self, index_domian, index_boundary, device):
        super(GaussLegendreQuadrature, self).__init__(index_domian, device)
        self.index = index_boundary
        
    def point_quadpts(self, point):
        return super().point_quadpts(point)

    def interval_quadpts(self, interval, h):
        return super().interval_quadpts(interval, h)

    def interval_quadpts_exact(self, x, h):
        return super().interval_quadpts_exact(x, h)

    def rectangle_quadpts(self, rectangle, h):
        return super().rectangle_quadpts(rectangle, h)

    def cuboid_quadpts(self, cuboid, h):
        return super().cuboid_quadpts(cuboid, h)

    def interval_boundary_quadpts(self, interval):

        quadpts = interval.reshape(-1,1).astype(np.float64)
        weights = np.ones_like(quadpts).astype(np.float64)
        h = np.array([1.], dtype=np.float64)
        return Quadrature(self.qtype, self.device, quadpts, weights, h)

    def rectangle_boundary_quadpts(self, rectangle, h):
        
        quadrature = GaussLegendreQuadrature(self.index, self.device)
        interval_quadpts_0 = quadrature.interval_quadpts(rectangle[0].reshape(1,2), h[0])
        interval_quadpts_1 = quadrature.interval_quadpts(rectangle[1].reshape(1,2), h[1])

        x = np.ones_like(interval_quadpts_1.quadpts)
        y = np.ones_like(interval_quadpts_0.quadpts)

        pts_x_y_0 = np.concatenate((interval_quadpts_0.quadpts, rectangle[1][0]*y), axis=1)
        pts_x_y_1 = np.concatenate((interval_quadpts_0.quadpts, rectangle[1][1]*y), axis=1)
        pts_y_x_0 = np.concatenate((rectangle[0][0]*x, interval_quadpts_1.quadpts), axis=1)
        pts_y_x_1 = np.concatenate((rectangle[0][1]*x, interval_quadpts_1.quadpts), axis=1)
        wei_x_y_0 = wei_x_y_1 = interval_quadpts_0.weights
        wei_y_x_0 = wei_y_x_1 = interval_quadpts_1.weights

        # four segments taken part when calculating, with the right h[i]
        quadpts = np.concatenate((pts_x_y_0, pts_x_y_1, pts_y_x_0, pts_y_x_1), axis=0)
        weights = np.concatenate((wei_x_y_0, wei_x_y_1, wei_y_x_0, wei_y_x_1), axis=0) 

        return Quadrature(self.qtype, self.device, quadpts, weights, h)

    def cuboid_boundary_quadpts(self, cuboid, h):

        quadrature = GaussLegendreQuadrature(self.index, self.device)

        indxy = np.array([0, 1])
        indxz = np.array([0, 2])
        indyz = np.array([1, 2])

        rectangle_quadpts_xy = quadrature.rectangle_quadpts(cuboid[0].reshape(1,2), h[indxy])
        rectangle_quadpts_xz = quadrature.rectangle_quadpts(cuboid[1].reshape(1,2), h[indxz])
        rectangle_quadpts_yz = quadrature.rectangle_quadpts(cuboid[2].reshape(1,2), h[indyz])

        x = np.ones_like(rectangle_quadpts_yz.quadpts)
        y = np.ones_like(rectangle_quadpts_xz.quadpts)
        z = np.ones_like(rectangle_quadpts_xy.quadpts)

        pts_xy_z_0 = np.concatenate((rectangle_quadpts_xy.quadpts, cuboid[2][0]*z), axis=1)
        pts_xy_z_1 = np.concatenate((rectangle_quadpts_xy.quadpts, cuboid[2][1]*z), axis=1)
        pts_xz_y_0 = np.concatenate((rectangle_quadpts_xz.quadpts, cuboid[1][0]*y), axis=1)
        pts_xz_y_1 = np.concatenate((rectangle_quadpts_xz.quadpts, cuboid[1][1]*y), axis=1)
        pts_yz_x_0 = np.concatenate((rectangle_quadpts_xy.quadpts, cuboid[1][0]*x), axis=1)
        pts_yz_x_1 = np.concatenate((rectangle_quadpts_xy.quadpts, cuboid[1][1]*x), axis=1)

        wei_xy_z_0 = wei_xy_z_1 = rectangle_quadpts_xy.weights
        wei_xz_y_0 = wei_xz_y_1 = rectangle_quadpts_xz.weights
        wei_yz_x_0 = wei_yz_x_1 = rectangle_quadpts_yz.weights

        # six segments taken part when calculating, with the right h[i]
        quadpts = np.concatenate((pts_xy_z_0, pts_xy_z_1, pts_xz_y_0, pts_xz_y_1, pts_yz_x_0, pts_yz_x_1), axis=0)
        weights = np.concatenate((wei_xy_z_0, wei_xy_z_1, wei_xz_y_0, wei_xz_y_1, wei_yz_x_0, wei_yz_x_1), axis=0)

        return Quadrature(self.qtype, self.device, quadpts, weights, h)
