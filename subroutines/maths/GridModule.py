# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:15:17 2019

Function designed to generate small 1D grids.

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""


def Grid1D(Min: float=0.0, Max: float=1.0, NGrid: int=5, GridType: str=None ):
    """
    Creates a arry containing a grid.
    
    - Min : lower bound of the grid, float [DEFAULT = 0.0] (in case of log this is the power of 10 for the minimum)
    - Max : upper bound of the grid, float [DEFAULT = 1.0] (in case of log this is the power of 10 for the maximum)
    - NGrid : number of gridpoints, int [DEFAULT = 5, >=2] 
    - GridType: Tytpe of grid, string : linear (default), log
    """
    LST=[] #initialise empty list
    
    if GridType is None:
        GridType="linear"
    
    if NGrid<=2:
        if (GridType == "linear" ):
            LST.append(Min)
            LST.append(Max)
        elif (GridType == "log" ):
            LST.append(10**Min)
            LST.append(10**Max)
        else: #just default to linear
            LST.append(Min)
            LST.append(Max)
    else:
        step=(Max-Min)/(NGrid-1)
        if (GridType == "linear" ):
            for nr in range(NGrid):
                LST.append(Min+(nr*step))
        elif (GridType == "log" ):
            for nr in range(NGrid):
                LST.append(10 **(Min+(nr*step)))
        else: #just default to linear
            for nr in range(NGrid):
                LST.append(Min+(nr*step))
    
    return LST
        
    
    