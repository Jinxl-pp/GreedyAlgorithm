import numpy as np
from .prettytable import PrettyTable


def get_title_string(loss_type, ftype, degree, total_time):
    
    if ftype == "relu":
        f_string = 'relu_power = ' + '%d, '%degree
    elif ftype == "bspline":
        f_string = 'bspline_power = ' + '%d, '%degree
    else:
        f_string = 'sigmoidal function' 
        
    title_string = loss_type + ', ' + f_string + 'total time = {:.4f}s'.format(total_time)
    return title_string


def finite_neuron_method(num_neuron, l2_err, hm_err, atype, ftype, degree, total_time):
    
    N = int(np.log2(num_neuron))
    order = [2**(i+1) - 1 for i in range(N)]
    
    if len(l2_err) == N:
        l2_err = l2_err.cpu()
        hm_err = hm_err.cpu()
    else:
        l2_err = l2_err[order].cpu()
        hm_err = hm_err[order].cpu()
    l2_rate = np.log2(l2_err[:-1] / l2_err[1:])
    hm_rate = np.log2(hm_err[:-1] / hm_err[1:])
    l2_rate = np.concatenate((np.array([[0.]]),l2_rate), axis = 0)
    hm_rate = np.concatenate((np.array([[0.]]),hm_rate), axis = 0)
    
    table = PrettyTable(['N','l2_err','l2_rate','energy_err','energy_rate'])
    table.align['N'] = 'r'
    table.align['l2_err'] = 'c'
    table.align['l2_rate'] = 'c'
    table.align['energy_err'] = 'c'
    table.align['energy_rate'] = 'c'
    
    for i in range(N):
        if i == 0:
            rate1 = '-'
        else:
            rate1 = '{:.2f}'.format(l2_rate[i].item())
        if i == 0:
            rate2 = '-'
        else:
            rate2 = '{:.2f}'.format(hm_rate[i].item())
        table.add_row([order[i]+1, '{:.3e}'.format(l2_err[i].item()), rate1, \
                                   '{:.3e}'.format(hm_err[i].item()), rate2])
    print('\n')
    loss_type = atype + '-FNM'
    title_string = get_title_string(loss_type, ftype, degree, total_time)
    print(table.get_string(title = title_string))
    
    
def finite_neuron_method_fitting(num_neuron, fitting_err, atype, ftype, degree, total_time):
    
    N = int(np.log2(num_neuron))
    order = [2**(i+1) - 1 for i in range(N)]
    
    fitting_err = fitting_err[order].cpu()
    fitting_rate = np.log2(fitting_err[:-1] / fitting_err[1:])
    fitting_rate = np.concatenate((np.array([[0.]]),fitting_rate), axis = 0)
    
    table = PrettyTable(['N','fitting_err','fitting_rate'])
    table.align['N'] = 'r'
    table.align['fitting_err'] = 'c'
    table.align['fitting_rate'] = 'c'
    
    for i in range(N):
        if i == 0:
            rate1 = '-'
        else:
            rate1 = '{:.2f}'.format(fitting_rate[i].item())
        table.add_row([order[i]+1, '{:.3e}'.format(fitting_err[i].item()), rate1])
    print('\n')
    loss_type = atype + '-FNM'
    title_string = get_title_string(loss_type, ftype, degree, total_time)
    print(table.get_string(title = title_string))    
    

def pinn_method(num_neuron, loss, l2_err, hm_err, atype, ftype, degree, total_time):
    
    N = int(np.log2(num_neuron))
    order = [2**(i+1) - 1 for i in range(N)]
    
    loss = loss[order].cpu()
    l2_err = l2_err[order].cpu()
    hm_err = hm_err[order].cpu()
    loss_rate = np.log2(loss[:-1] / loss[1:])
    l2_rate = np.log2(l2_err[:-1] / l2_err[1:])
    hm_rate = np.log2(hm_err[:-1] / hm_err[1:])
    loss_rate = np.concatenate((np.array([[0.]]),loss_rate), axis = 0)
    l2_rate = np.concatenate((np.array([[0.]]),l2_rate), axis = 0)
    hm_rate = np.concatenate((np.array([[0.]]),hm_rate), axis = 0)
    
    table = PrettyTable(['N','loss','loss_rate','l2_err','l2_rate','energy_err','energy_rate'])
    table.align['N'] = 'r'
    table.align['loss'] = 'c'
    table.align['loss_rate'] = 'c'
    table.align['l2_err'] = 'c'
    table.align['l2_rate'] = 'c'
    table.align['energy_err'] = 'c'
    table.align['energy_rate'] = 'c'
    
    for i in range(N):
        if i == 0:
            rate1 = '-'
        else:
            rate1 = '{:.2f}'.format(loss_rate[i].item())
        if i == 0:
            rate2 = '-'
        else:
            rate2 = '{:.2f}'.format(l2_rate[i].item())
        if i == 0:
            rate3 = '-'
        else:
            rate3 = '{:.2f}'.format(hm_rate[i].item())
        table.add_row([order[i]+1, '{:.3e}'.format(loss[i].item()), rate1, \
                                   '{:.3e}'.format(l2_err[i].item()), rate2, \
                                   '{:.3e}'.format(hm_err[i].item()), rate3])
    print('\n')
    loss_type = atype + '-PINN'
    title_string = get_title_string(loss_type, ftype, degree, total_time)
    print(table.get_string(title = title_string))