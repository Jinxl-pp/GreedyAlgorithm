import sys
sys.path.append('../')

import torch
import numpy as np




class Test:
    
    def __init__(self):
        self.a_func = self.b_func
        
    def b_func(self, p):
        return 2. * p


if __name__ == '__main__':
    
    ###
    ### part 1
    ###
    # # requires_grad_(): use it after clone is ok, or use .detach() when finish using it.
    # a = torch.rand(5,1)
    # a.requires_grad_()
    # print(a)
    # a = a.detach()
    # print(a)
    # b = a.detach()
    # print(b)
    
    ###
    ### part 2
    ###
    # initialization 
    test = Test()
    c = torch.rand(5,1)
    print(c)
    print(test.a_func(c))
    print(test.b_func(c))
    
    ###
    ### part 3
    ###
    d = torch.rand(5,1)
    print(d)
    print(d.flatten())
    print(d.reshape(1,-1))
    e = torch.rand(5,2)
    print(e)
    print(e[:,0:1])
    
    ###
    ### part 3
    ###
    e = "fista"
    if e:
        f = 1
    print(f)