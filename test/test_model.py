import sys
sys.path.append('../')

import torch
import numpy as np
from torch.nn.parameter import Parameter
from greedy.model import activation_function as af
from greedy.model import shallownet as sn


def _polar_to_cartesian(dim, theta):
    
    # coordinate transformation
    if dim == 1:
        w = theta[0].reshape(-1,1)
        b = theta[1].reshape(-1,1)
        return (w, b)
    elif dim == 2:
        w1 = torch.cos(theta[0]).reshape(-1,1)
        w2 = torch.sin(theta[0]).reshape(-1,1)
        b = theta[1].reshape(-1,1)
        return (w1, w2, b)
    elif dim == 3:
        w1 = torch.cos(theta[0]) * torch.sin(theta[1]).reshape(-1,1)
        w2 = torch.sin(theta[0]) * torch.sin(theta[1]).reshape(-1,1)
        w3 = torch.cos(theta[1]).reshape(-1,1)
        b = theta[2].reshape(-1,1)
        return (w1, w2, w3, b)
    else:
        raise RuntimeError("Dimension error.")
    

if __name__ == '__main__':
    
    # # 1D shallow-neural-net model 
    # in_dim = 1
    # width = 20
    # out_dim = 1
    # ftype = "relu"
    # degree = 2
    # sigma = af.ActivationFunction(ftype, degree)
    # snn = sn.ShallowNN(sigma, in_dim, width, out_dim)
    
    # # test model
    # print("model:")
    # print(snn)
    # print("layer1_weights:")
    # print(snn.layer1.weight.shape)
    # print(snn.layer1.weight)
    # print("layer1_bias:")
    # print(snn.layer1.bias.shape)
    # print(snn.layer1.bias)
    # print("layer2_weights:")
    # print(snn.layer2.weight.shape)
    # print(snn.layer2.weight)
    # print("layer2_bias:")
    # print(snn.layer2.bias)
    
    # # update snn model
    # w1 = torch.ones(width, 1) * 1 
    # b1 = torch.ones(width) * 2
    # w2 = torch.ones(1, width) * 3
    # parameters = (w2, w1, b1)
    # snn.update_neurons(parameters)
    # print("updated model:")
    # print("layer1_weights:")
    # print(snn.layer1.weight.shape)
    # print(snn.layer1.weight)
    # print("layer1_bias:")
    # print(snn.layer1.bias.shape)
    # print(snn.layer1.bias)
    # print("layer2_weights:")
    # print(snn.layer2.weight.shape)
    # print(snn.layer2.weight)
    # print("layer2_bias:")
    # print(snn.layer2.bias)
    
    # 2D shallow-neural-net model 
    in_dim = 2
    width = 20
    out_dim = 1
    ftype = "relu"
    degree = 2
    sigma = af.ActivationFunction(ftype, degree)
    snn = sn.ShallowNN(sigma, in_dim, width, out_dim)
    
    # test model
    print("model:")
    print(snn)
    print("layer1_weights:")
    print(snn.layer1.weight.shape)
    print(snn.layer1.weight)
    print("layer1_bias:")
    print(snn.layer1.bias.shape)
    print(snn.layer1.bias)
    print("layer2_weights:")
    print(snn.layer2.weight.shape)
    print(snn.layer2.weight)
    print("layer2_bias:")
    print(snn.layer2.bias)
    
    # update snn model
    w1 = torch.ones(width, 2) * 1 
    b1 = torch.ones(width) * 2
    w2 = torch.ones(1, width) * 3
    parameters = (w2, w1, b1)
    snn.update_neurons(parameters)
    print("updated model:")
    print("layer1_weights:")
    print(snn.layer1.weight.shape)
    print(snn.layer1.weight)
    print("layer1_bias:")
    print(snn.layer1.bias.shape)
    print(snn.layer1.bias)
    print("layer2_weights:")
    print(snn.layer2.weight.shape)
    print(snn.layer2.weight)
    print("layer2_bias:")
    print(snn.layer2.bias)