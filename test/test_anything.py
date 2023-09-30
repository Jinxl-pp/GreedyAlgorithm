import sys
sys.path.append('../')

import torch
import numpy as np




class Test:
    
    def __init__(self):
        self.a_func = self.b_func
        
    def b_func(self, p):
        return 2. * p
    
    
def partition(num_param, n):
    division = len(num_param) / n
    return [0]+[round(division * (i + 1)) for i in range(n)]


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
    
    ###
    ### part 4
    ###
    lst = range(63355)
    print(partition(lst,10))
    print(torch.linspace(0,1,10))
    
    ###
    ### part 5
    ###
    a = torch.tensor([1.,2.])
    b = torch.tensor([2.])
    c = torch.tensor([3.])
    items = (a,b,c)
    print(items)
    print(items[0:-1])
    print(items[0].pow(2).sum())
    
    ###
    ### part 5
    ###
    w = torch.rand(5,1)
    print(w)
    print(torch.ones(w.shape))
    