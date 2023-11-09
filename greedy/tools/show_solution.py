import torch
import numpy as np
import matplotlib.pyplot as plt


def show_solution_1d(approximation, interval, label):
    
    min = interval[0][0]
    max = interval[0][1]
    num_samples = 10000
    samples = torch.linspace(min,max,num_samples).reshape(-1,1)
    aprox = approximation(samples).detach().numpy()
    
    plt.plot(samples,aprox,color='b',linewidth=1.0,linestyle="-.")
    plt.xlabel(label)
    plt.show()


def compare_solution_1d(exact_solution, approximation, interval):
    
    min = interval[0][0]
    max = interval[0][1]
    num_samples = 10000
    samples = torch.linspace(min,max,num_samples).reshape(-1,1)
    
    exact = exact_solution(samples).numpy()
    aprox = approximation(samples).detach().numpy()
    
    plt.plot(samples,exact,color='r',linewidth=1.0,linestyle="-",label="$f(x)$")
    plt.plot(samples,aprox,color='b',linewidth=1.0,linestyle="-.",label="$u_n(x)$")
    plt.legend(loc="upper left")
    plt.show()
    
    