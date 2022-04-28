import numpy as np
from Quadrature import Quadrature

## =====================================
## random samples
## numpy    

class MonteCarloQuadrature(Quadrature):

    def __init__(self, device):
        """ The Monte-Carlo sampling information on an arbitrary domain.
            INPUT:
                device: cuda/cpu
        """     
        self.device = device

    def interval_samples(self, interval, number_of_samples):
        """ The Monte Carlo information on 1d interval [a,b].
        """  
        measure = interval[0][1] - interval[0][0]
        quadpts = interval[0][0] + np.random.rand(number_of_samples, 1) * measure
        quadpts = quadpts.astype(np.float64)
        weights = np.array([[measure]], dtype=np.float64)
        h = np.array([1 / number_of_samples])
        return Quadrature(self.device, quadpts, weights, h)

    def rectangle_samples(self, rectangle, number_of_samples):
        """ The Monte Carlo information on 2d rectangle [a,b]*[c,d].
        """    

        measure_0 = rectangle[0][1] - rectangle[0][0]
        measure_1 = rectangle[1][1] - rectangle[1][0]
        measure = measure_0 * measure_1

        quadpts_x = rectangle[0][0] + np.random.rand(number_of_samples, 1) * measure_0
        quadpts_y = rectangle[1][0] + np.random.rand(number_of_samples, 1) * measure_1
        quadpts_x = quadpts_x.astype(np.float64)
        quadpts_y = quadpts_y.astype(np.float64)

        quadpts = np.concatenate((quadpts_x, quadpts_y), axis=1)
        weights = np.array([[measure]], dtype=np.float64)
        h = np.array([1 / number_of_samples])

        return Quadrature(self.device, quadpts, weights, h)        

    def cuboid_samples(self, cuboid, number_of_samples):
        """ The Monte Carlo information on 3d cuboid [a,b]*[c,d]*[e,f].
        """    

        measure_0 = cuboid[0][1] - cuboid[0][0]
        measure_1 = cuboid[1][1] - cuboid[1][0]
        measure_2 = cuboid[2][1] - cuboid[2][0]
        measure = measure_0 * measure_1 * measure_2

        quadpts_x = cuboid[0][0] + np.random.rand(number_of_samples, 1) * measure_0
        quadpts_y = cuboid[1][0] + np.random.rand(number_of_samples, 1) * measure_1
        quadpts_z = cuboid[2][0] + np.random.rand(number_of_samples, 1) * measure_2
        quadpts_x = quadpts_x.astype(np.float64)
        quadpts_y = quadpts_y.astype(np.float64)
        quadpts_z = quadpts_z.astype(np.float64)

        quadpts = np.concatenate((quadpts_x, quadpts_y, quadpts_z), axis=1)
        weights = np.array([[measure]], dtype=np.float64)
        h = np.array([1 / number_of_samples])
        
        return Quadrature(self.device, quadpts, weights, h)      

    def circle_samples(self, center, radius, number_of_samples):
        """ The Monte Carlo information on 2d circle.
            The random samples are generated from the distibution of polar variables.
            Let X ~ U([0,1]), then radius ~ R * sqrt(X)
            Let Y ~ U([0,1]), then theta ~ 2*pi * Y
            x = radius * cos(theta)
            y = radius * sin(theta)
        """    

        pi = np.pi
        measure = pi * radius**2
        quadpts_radius = radius * np.sqrt(np.random.rand(number_of_samples, 1))
        quadpts_theta = 2 * pi * np.random.rand(number_of_samples, 1)
        quadpts_x = quadpts_radius * np.cos(quadpts_theta) + center[0]
        quadpts_y = quadpts_radius * np.sin(quadpts_theta) + center[1]

        quadpts = np.concatenate((quadpts_x, quadpts_y), axis=1).astype(np.float64)
        weights = np.array([[measure]], dtype=np.float64)
        h = np.array([1 / number_of_samples])
 
        return Quadrature(self.device, quadpts, weights, h)

    def ball_samples(self, center, radius, number_of_samples):
        """ The Monte Carlo information on 3d ball.
            The random samples are generated from the distibution of spherical variables.
            Let X ~ U([0, 1]), then radius ~ R * X^(1/3)
            Let Y ~ U([0, 1]), then phi ~ 2*pi * Y, ranges in [0, 2*pi]
            Let Z ~ U([-1,1]), then theta ~ arcsin(Z), ranges in [-pi/2, pi/2]
            x = radius * cos(theta) * cos(phi)
            y = radius * cos(theta) * sin(phi)
            z = radius * sin(theta)
        """  

        pi = np.pi
        measure = (4/3) * pi * radius**3
        quadpts_radius = radius * pow(np.random.rand(number_of_samples, 1), 1/3)
        quadpts_theta = np.arcsin(2*np.random.rand(number_of_samples, 1) - 1)
        quadpts_phi = 2 * pi * np.random.rand(number_of_samples, 1)

        quadpts_x = quadpts_radius * np.cos(quadpts_theta) * np.cos(quadpts_phi) + center[0]
        quadpts_y = quadpts_radius * np.cos(quadpts_theta) * np.sin(quadpts_phi) + center[1]
        quadpts_z = quadpts_radius * np.sin(quadpts_theta) + center[2]

        quadpts = np.concatenate((quadpts_x, quadpts_y, quadpts_z), axis=1).astype(np.float64)
        weights = np.array([[measure]], dtype=np.float64)
        h = np.array([1 / number_of_samples])
        return Quadrature(self.device, quadpts, weights, h)        


class BoundaryMonteCarloQuadrature(MonteCarloQuadrature):
    """ The cartesian Monte-Carlo sampling information on the boundary
        of a certain domain for 1D, 2D and 3D.
    """

    def __init__(self, device):
        self.device = device

    def interval_boundary_samples(self, interval):

        quadpts = interval.reshape(-1,1).astype(np.float64)
        weights = np.ones_like(quadpts).astype(np.float64)
        h = np.array([1/2], dtype=np.float64)
        return Quadrature(self.device, quadpts, weights, h)
        
    def rectangle_boundary_samples(self, rectangle, number_of_each_hyperplane):

        sampling = MonteCarloQuadrature(self.device)
        interval_samples_0 = sampling.interval_samples(rectangle[0].reshape(1,2), number_of_each_hyperplane)
        interval_samples_1 = sampling.interval_samples(rectangle[1].reshape(1,2), number_of_each_hyperplane)

        x = np.ones_like(interval_samples_1.quadpts)
        y = np.ones_like(interval_samples_0.quadpts)

        pts_x_y_0 = np.concatenate((interval_samples_0.quadpts, rectangle[1][0]*y), axis=1)
        pts_x_y_1 = np.concatenate((interval_samples_0.quadpts, rectangle[1][1]*y), axis=1)
        pts_y_x_0 = np.concatenate((rectangle[0][0]*x, interval_samples_1.quadpts), axis=1)
        pts_y_x_1 = np.concatenate((rectangle[0][1]*x, interval_samples_1.quadpts), axis=1)
        wei_x_y_0 = wei_x_y_1 = interval_samples_0.weights
        wei_y_x_0 = wei_y_x_1 = interval_samples_1.weights

        quadpts = np.concatenate((pts_x_y_0, pts_x_y_1, pts_y_x_0, pts_y_x_1), axis=0)
        weights = np.concatenate((wei_x_y_0, wei_x_y_1, wei_y_x_0, wei_y_x_1), axis=0) 
        h = np.array([1/number_of_each_hyperplane], dtype=np.float64)

        return Quadrature(self.device, quadpts, weights, h)    

    def cuboid_boundary_samples(self, cuboid, number_of_each_hyperplane):

        sampling = MonteCarloQuadrature(self.device)

        rectangle_samples_xy = sampling.rectangle_samples(cuboid[0].reshape(1,2), number_of_each_hyperplane)
        rectangle_samples_xz = sampling.rectangle_samples(cuboid[1].reshape(1,2), number_of_each_hyperplane)
        rectangle_samples_yz = sampling.rectangle_samples(cuboid[2].reshape(1,2), number_of_each_hyperplane)

        x = np.ones_like(rectangle_samples_yz.quadpts)
        y = np.ones_like(rectangle_samples_xz.quadpts)
        z = np.ones_like(rectangle_samples_xy.quadpts)

        pts_xy_z_0 = np.concatenate((rectangle_samples_xy.quadpts, cuboid[2][0]*z), axis=1)
        pts_xy_z_1 = np.concatenate((rectangle_samples_xy.quadpts, cuboid[2][1]*z), axis=1)
        pts_xz_y_0 = np.concatenate((rectangle_samples_xz.quadpts, cuboid[1][0]*y), axis=1)
        pts_xz_y_1 = np.concatenate((rectangle_samples_xz.quadpts, cuboid[1][1]*y), axis=1)
        pts_yz_x_0 = np.concatenate((rectangle_samples_xy.quadpts, cuboid[1][0]*x), axis=1)
        pts_yz_x_1 = np.concatenate((rectangle_samples_xy.quadpts, cuboid[1][1]*x), axis=1)

        wei_xy_z_0 = wei_xy_z_1 = rectangle_samples_xy.weights
        wei_xz_y_0 = wei_xz_y_1 = rectangle_samples_xz.weights
        wei_yz_x_0 = wei_yz_x_1 = rectangle_samples_yz.weights

        quadpts = np.concatenate((pts_xy_z_0, pts_xy_z_1, pts_xz_y_0, pts_xz_y_1, pts_yz_x_0, pts_yz_x_1), axis=0)
        weights = np.concatenate((wei_xy_z_0, wei_xy_z_1, wei_xz_y_0, wei_xz_y_1, wei_yz_x_0, wei_yz_x_1), axis=0)
        h = np.array([1/number_of_each_hyperplane], dtype=np.float64)

        return Quadrature(self.device, quadpts, weights, h)

    def circle_boundary_samples(self, center, radius, number_of_samples):
        """ The Monte Carlo information on 2d circle's boundary.
            The random samples are generated from the distibution of polar variables.
            Let radius = R being a constant.
            Let Y ~ U([0,1]), then theta ~ 2*pi * Y.
            x = radius * cos(theta),
            y = radius * sin(theta).
        """    

        pi = np.pi
        measure = 2 * pi * radius
        quadpts_radius = radius * np.ones(number_of_samples, 1)
        quadpts_theta = 2 * pi * np.random.rand(number_of_samples, 1)
        quadpts_x = quadpts_radius * np.cos(quadpts_theta) + center[0]
        quadpts_y = quadpts_radius * np.sin(quadpts_theta) + center[1]
        quadpts = np.concatenate((quadpts_x, quadpts_y), axis=1).astype(np.float64)
        weights = np.array([[measure]], dtype=np.float64)
        h = np.array([1 / number_of_samples])
        return Quadrature(self.device, quadpts, weights, h)

    def ball_boundary_samples(self, center, radius, number_of_samples):
        """ The Monte Carlo information on 3d shpere.
            The random samples are generated from the distibution of spherical variables.
            Let radius = R being a constant.
            Let Y ~ U([0, 1]), then phi ~ 2*pi * Y, ranges in [0, 2*pi].
            Let Z ~ U([-1,1]), then theta ~ arcsin(Z), ranges in [-pi/2, pi/2].
            x = radius * cos(theta) * cos(phi), 
            y = radius * cos(theta) * sin(phi),
            z = radius * sin(theta).
        """  

        pi = np.pi
        measure = 4 * pi * radius**2
        quadpts_radius = radius * np.ones(number_of_samples, 1)
        quadpts_theta = np.arcsin(2*np.random.rand(number_of_samples, 1) - 1)
        quadpts_phi = 2 * pi * np.random.rand(number_of_samples, 1)

        quadpts_x = quadpts_radius * np.cos(quadpts_theta) * np.cos(quadpts_phi) + center[0]
        quadpts_y = quadpts_radius * np.cos(quadpts_theta) * np.sin(quadpts_phi) + center[1]
        quadpts_z = quadpts_radius * np.sin(quadpts_theta) + center[2]

        quadpts = np.concatenate((quadpts_x, quadpts_y, quadpts_z), axis=1).astype(np.float64)
        weights = np.array([[measure]], dtype=np.float64)
        h = np.array([1 / number_of_samples])
        return Quadrature(self.device, quadpts, weights, h) 