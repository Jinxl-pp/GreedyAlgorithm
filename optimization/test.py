import time
import torch
import numpy as np
from pgd import PGD
from lbfgs import LBFGS
from torch.nn.parameter import Parameter
import torch.nn as nn

def quadratic(x):
    a = list(x.parameters())
    val = a[0].pow(4) + a[1].pow(2)
    return val


if __name__ == '__main__':

    domain = torch.tensor([[-3,3],[-3,3]])
    x = torch.zeros(2,1)
    x = Parameter(x)

    x = nn.Linear(1,1)


    learning_rate = 1
    optimizer = PGD(x.parameters(),
                    domain,
                    lr=learning_rate,
                    max_iter=20,
                    max_eval=None,
                    tolerance_grad=1e-08,
                    tolerance_change=np.finfo(float).eps,
                    history_size=100,
                    line_search_fn="arc_armijo")  # strong_wolfe

    epochs = 2000
    pre_loss = 1
    tol_change = 1.0 * np.finfo(float).eps
    start_time = time.time()
    for epoch in range(epochs):
        print('epoch = ', epoch)
        def closure():
            optimizer.zero_grad()
            loss = quadratic(x)
            loss.backward()
            print('loss: {:.10e}, epoch = {:}'.format(loss.item(), epoch))
            return loss
        
        optimizer.step(closure)
        new_loss = closure().detach()
        if torch.abs(new_loss - pre_loss) < tol_change:
            break
        else:
            pre_loss = new_loss
        
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

