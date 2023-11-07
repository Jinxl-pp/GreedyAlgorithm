import torch
import numpy as np

class Quadrature(object):
    def __init__(self, qtype, device, quadpts, weights, h):
        # weights = weights[:,0]
        self.qtype = qtype
        self.quadpts = torch.from_numpy(quadpts).type(dtype=torch.float64).to(device)
        self.weights = torch.from_numpy(weights).type(dtype=torch.float64).to(device)
        h = torch.tensor(h).type(dtype=torch.float64).to(device)
        self.area = torch.prod(h)

    def get_number_of_points(self):
        return self.weights.shape[0]

    def get_dimension(self):
        return self.quadpts.shape[1]   