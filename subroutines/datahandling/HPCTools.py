# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:37:07 2020

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""

def set_num_threads(nt: int=1):
    """
    Set some machine parameters which control
    the number of threads being spawned.
    
    Default is set to 1.
    
    """
    import os
    import mkl
    mkl.set_num_threads(nt)
    nt=str(nt)
    os.environ['OPENBLAS_NUM_THREADS'] = nt
    os.environ['NUMEXPR_NUM_THREADS'] = nt
    os.environ['OMP_NUM_THREADS'] = nt
    os.environ['MKL_NUM_THREADS'] = nt
    
def get_num_procs(njobs:int)->int:
    """
    Small wrapper function used to get an integer number of cores, based on
    the user selection via njobs:
        
        - njobs = 0, set it to 1 core (serial)
        - njobs > 0, set it to njobs cores (parallel)
        - njobs < 0, set it to njobs * the number of physical cores as obtained by psutil.cpu_count
                    (This is useful for multi-socket setups, where psutil will only give the 
                    number of cores on a single socket)
    """
    import psutil
    
    if (njobs==0):
        njobs=1
    if (njobs<0):
        procs=psutil.cpu_count(logical=False)*abs(njobs)
    else:
        procs=njobs
    
    return procs