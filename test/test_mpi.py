# for multiple node cluster
# based on MPI

import numpy as np
from mpi4py import MPI 
import random

# to be parallelized 
def job(a, b):
    return a*b

def big_job_mpi(arr_a,arr_b):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    size_a, size_b = arr_a.shape[0], arr_b.shape[0]
    numjobs = size_a*size_b

    job_content = [] # the collection of parameters [a,b]
    for a_cur in arr_a:
        for b_cur in arr_b:
            job_content.append((a_cur,b_cur))

    # arrange the works and jobs
    if rank==0:
        # this is head worker
        # jobs are arranged by this worker
        job_all_idx =list(range(numjobs))  
        random.shuffle(job_all_idx)
        # shuffle the job index to make all workers equal
        # for unbalanced jobs
    else:
        job_all_idx=None
    
    job_all_idx = comm.bcast(job_all_idx,root=0)
    
    njob_per_worker = int(numjobs/size) 
    # the number of jobs should be a multiple of the NumProcess[MPI]
    
    this_worker_job = [job_all_idx[x] for x in range(rank*njob_per_worker, (rank+1)*njob_per_worker)]
    
    # map the index to parameterset [eps,anis]
    work_content = [job_content[x] for x in this_worker_job ]

    for a_piece_of_work in work_content:
        job(*a_piece_of_work)

if __name__=="__main__": 
    
    # parameter space to explore
    arr_a = np.linspace(0.03,0.5,36)    
    arr_b = np.linspace(0.05,0.95,36)
    
    big_job_mpi(arr_a,arr_b)
    pass