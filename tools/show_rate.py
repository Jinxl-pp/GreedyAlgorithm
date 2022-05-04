"""
Added by Xianlin Jin
2022.04.28
"""
import numpy as np
from prettytable import PrettyTable


def oga_print(num_neuron, errl2_record, errhm_record, title_string):
    
    N = int(np.log2(num_neuron))
    order = [2**(i+1) - 1 for i in range(N)]
    
    errl2_record = errl2_record[order].cpu()
    errhm_record = errhm_record[order].cpu()
    ratel2_record = np.log2(errl2_record[:-1] / errl2_record[1:])
    ratehm_record = np.log2(errhm_record[:-1] / errhm_record[1:])
    ratel2_record = np.concatenate((np.array([[0.]]),ratel2_record), axis = 0)
    ratehm_record = np.concatenate((np.array([[0.]]),ratehm_record), axis = 0)
    
    table = PrettyTable(['N','err_l2','rate_l2','err_energy','rate_energy'])
    table.align['N'] = 'r'
    table.align['err_l2'] = 'c'
    table.align['rate_l2'] = 'c'
    table.align['err_energy'] = 'c'
    table.align['rate_energy'] = 'c'
    
    for i in range(N):
        if i == 0:
            rate1 = '-'
        else:
            rate1 = '{:.2f}'.format(ratel2_record[i].item())
        if i == 0:
            rate2 = '-'
        else:
            rate2 = '{:.2f}'.format(ratehm_record[i].item())
        table.add_row([order[i]+1, '{:.6e}'.format(errl2_record[i].item()), rate1, \
                                   '{:.6e}'.format(errhm_record[i].item()), rate2])
    print(table.get_string(title = title_string))