import sys
sys.path.append('../')

import torch
import matplotlib.pyplot as plt

from greedy.model import activation_function as af 

def show_model(sigma):
    
    # define samples and evaluations
    samples = torch.linspace(-5, 5, 1000)
    f_data = sigma.activate(samples)
    
    # plot the shape of model
    plt.plot(samples, f_data, label="activation", color="red", linewidth=1)
    plt.xlabel("sample_points")
    plt.ylabel("evaluation")
    plt.show()
    
# main process
if __name__ == "__main__":
    
    # activation function info
    ftype = "bspline"
    degree = 1
    sigma = af.ActivationFunction(ftype, degree)
    
    # show the shape of activation function
    show_model(sigma)
    
    # activation functions passed test:
    # 1. ftype = "sigmoid" 
    # 2. ftype = "relu", with degree = 1, 2, 3, ...
    # 3. ftype = "bspline", with degree = 2, 3, 4, 5
    # 4. ftype = "bspline", with degree = 1, <==> hat_function