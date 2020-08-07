# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:10:49 2020

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
import numpy as np

def checkEqual(lst: list)->bool:
    """
    Small function to quickly check if all elements in a list have the same value. 
    
    Parameter:
        - lst : input list
    Returns:
        Boolean: True if all are equal, False else. 
    
    Source: http://stackoverflow.com/q/3844948/
    part of the thread: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    (Seems to be roughly the fastest...)
    """    
    return not lst or lst.count(lst[0]) == len(lst)

def checkEqualNDarray(lst: np.ndarray)->bool:
    """
    Small function to quickly check if all elements in a list have the same value. 
    
    Parameter:
        - lst : input list
    Returns:
        Boolean: True if all are equal, False else. 
    
    """        
    return len(lst)==0 or (lst==lst[0]).sum()==len(lst)
    