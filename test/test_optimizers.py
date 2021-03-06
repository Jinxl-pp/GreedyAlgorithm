import sys
sys.path.append('../')

import time
import torch
import numpy as np
from optimization import *
from torch.nn.parameter import Parameter

dtype = torch.float64
torch.set_default_dtype(dtype)

def quadratic(x):
    val = (x[0][0]-0.21).pow(4) + (x[1][0]-0.1).pow(4) + (x[2][0]-0.7001).pow(4)
    return val

if __name__ == '__main__':

    domain = torch.tensor([[0.2,3],[0.1,3],[0.7,3]])
    
    x1 = torch.tensor([[-2.2135]])
    x1 = Parameter(x1)

    x2 = torch.tensor([[2.5273]])
    x2 = Parameter(x2)

    x3 = torch.tensor([[-0.5273]])
    x3 = Parameter(x3)

    learning_rate = 1
    optimizer_1 = PGD([x1,x2,x3],#x.parameters(),
                    domain,
                    lr=learning_rate,
                    max_iter=20,
                    max_eval=None,
                    tolerance_grad=1e-08,
                    tolerance_change=np.finfo(float).eps,
                    history_size=100,
                    line_search_fn="armijo")  # arc_armijo > strong_wolfe > armijo

    optimizer_2 = FISTA([x1,x2,x3],#x.parameters(),
                    domain,
                    lr=learning_rate,
                    max_iter=20,
                    max_eval=None,
                    tolerance_grad=1e-08,
                    tolerance_change=np.finfo(float).eps,
                    history_size=100,
                    line_search_fn="arc_armijo")  # arc_armijo 

    optimizer = optimizer_2
    
    epochs = 2000
    pre_loss = 1
    tol_change = 1e-12 #1.0 * np.finfo(float).eps
    start_time = time.time()
    for epoch in range(epochs):
        print('epoch = ', epoch)
        def closure():
            optimizer.zero_grad()
            x = torch.cat([x1,x2,x3],axis=0)
            loss = quadratic(x)
            loss.backward()
            print('loss: {:.10e}, epoch = {:}'.format(loss.item(), epoch))
            # print(x[0][0].data)
            # print(x[1][0].data)
            # print(x[2][0].data)
            return loss
        
        optimizer.step(closure)
        new_loss = closure().detach()
        if torch.abs(new_loss - pre_loss) < tol_change:
            break
        else:
            pre_loss = new_loss
        
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

