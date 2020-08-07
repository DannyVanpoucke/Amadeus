# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:23:23 2020

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
import pandas as pd
import numpy as np

def Lin1DFunc(x:list, intercept: float, slope: float)-> list:
    """
    Create linear data: y=slope*x + intercept
    """
    return list(np.dot(slope,xi) + intercept for xi in x)

def LinNDFunc(x:list, intercept: float, slope: list)-> list:
    """
    Create linear data: y=slope1*x1+slope2*x2+...+slopeN*xN + intercept
    """
    return list(np.dot(slope,xi) + intercept for xi in x)

def LinPolyNDFunc(x:list, intercept: float, slope: list)-> list:
    """
    Create linear Polynomial data: y=slope1*x^1+slope2*x^2+...+slopeN*x^N + intercept
    """
    npow=[]
    for i in range(len(slope)):
        npow.append(i+1)
    
    features=[]
    for xi in x:
        features.append(np.power(xi,npow))
#    print("=============================")
#    print("slope=",slope)
#    print("x    =",x)
#    print("xi   =",features)
#    print("intercept=",intercept)
#    print("dot=",list(np.dot(slope,xi) for xi in features))
#    print("=============================")
    return list(np.dot(slope,xi) + intercept for xi in features)

def Exp1DFunc(x:list, intercept: float, slope: float)-> list:
    """
    Create a 1D exponential dataset: exp(slope*x) - intercept
    """
    return list(np.exp(slope*xi)-intercept for xi in x)

def SinFunc(x:list, intercept: float, slope: float)-> list:
    """
    Create a 1D sine dataset: sin(slope*x) + intercept
    """
    return list(np.sin(slope*xi)+intercept for xi in x)


def Create1DData(Func1D, length:int, intercept: float=0.0, coefficient: float=1.0, 
             lbound: float=0.0, ubound: float=1.0, 
             dist:str = None, dwidth: float=0.2) -> pd.DataFrame:
    """
    Create a noisy 1D dataset using a specific function.
        
    parameters:
        - length : size of the dataset
        - intercept : the intercept parameter of the function
        - coefficient: the coefficient (slope, factor,...) of the function
        - lbound and ubound: lower and uper bound of the x-range.
        - dist: Type of distribution. String = 'normal' or 'uniform' DEFAULT=normal
        - dwidth : the width of the distribution, Default=0.2
        - Func1D : 1-D function to generate data with. This function returns a list of results.
    """
    #create two lists of the correct length, filled with linear related data
    if (dist==None):
        dist='normal'
    wd=ubound-lbound
    x=lbound+np.random.random_sample(size=length)*wd
    y=list()
    
    if (dist=='normal'):
        rdn=(np.random.normal(size=length))*dwidth #gaussian  distribution with mean=0, and stdev=1
    elif (dist=='uniform'):
        rdn=(np.random.uniform(size=length)-1.0)*dwidth #uniform distribution from 0..1
    else:
        rdn=(np.random.normal(size=length))*dwidth #gaussian  distribution with mean=0, and stdev=1
        
    y=Func1D(x,intercept,coefficient)+rdn
    
    #transform in list of tuples
    xy_tuples=list(zip(x,y))
    #and convert into dataframe
    df=pd.DataFrame(xy_tuples, columns = ['Feat_x','Target_y'])
    
    return df


#def CreateExponential1D(length:int, intercept: float=0.0, scaleX: float=1.0, lbound: float=0.0, ubound: float=1.0) -> pd.DataFrame:
#    """
#    Create a (gaussian) noisy 1D exponential dataset: exp(slope*x) - intercept
#    
#    parameters:
#        - length : size of the dataset
#    """
#    #create two lists of the correct length, filled with linear related data
#    wd=ubound-lbound
#    x=lbound+np.random.random_sample(size=length)*wd
#    y=list()
#    
#    rdn=(np.random.normal(size=length))*0.2 #gaussian  distribution with mean=0, and stdev=1
#    y=Exp1DFunc(x,intercept,scaleX)+rdn
#    
#    #transform in list of tuples
#    xy_tuples=list(zip(x,y))
#    #and convert into dataframe
#    df=pd.DataFrame(xy_tuples, columns = ['Feat_x','Target_y'])
#    
#    return df


#def CreateSin1D(length:int, intercept: float=0.0, scaleX: float=1.0, lbound: float=0.0, ubound: float=1.0) -> pd.DataFrame:
#    """
#    Create a (gaussian) noisy 1D sine dataset: sin(slope*x) + intercept
#    
#    parameters:
#        - length : size of the dataset
#    """
#    #create two lists of the correct length, filled with linear related data
#    wd=ubound-lbound
#    x=lbound+np.random.random_sample(size=length)*wd
#    y=list()
#    
#    rdn=(np.random.normal(size=length))*0.05 #gaussian  distribution with mean=0, and stdev=1
#    y=SinFunc(x,intercept,scaleX)+rdn
#    
#    #transform in list of tuples
#    xy_tuples=list(zip(x,y))
#    #and convert into dataframe
#    df=pd.DataFrame(xy_tuples, columns = ['Feat_x','Target_y'])
#    
#    return df


def Lin3DFunc(x:list, intercept: float, slope: list)-> list:
    return slope[0]*x[0,:]+slope[1]*x[1,:]+slope[2]*x[2,:]+intercept


def CreateLinear3D(length:int, intercept: float=0.0, slope: list=[1.0,1.0,1.0], lbound: float=0.0, ubound: float=1.0) -> pd.DataFrame:
    """
    Create a noisy 3D linear dataset
    
    parameters:
        - length : size of the dataset
    """
    #create two lists of the correct length, filled with linear related data
    wd=ubound-lbound
    x1=lbound+np.random.random_sample(size=length)*wd
    x2=lbound+np.random.random_sample(size=length)*wd
    x3=lbound+np.random.random_sample(size=length)*wd
    x=list()
    x.append(x1)
    x.append(x2)
    x.append(x3)
    y=list()
    rdn=(np.random.normal(size=length))*0.2 #gaussian  distribution with mean=0, and stdev=1
    y=Lin3DFunc(x,intercept,slope)+rdn
        
    #transform in list of tuples
    xy_tuples=list(zip(x1,x2,x3,y))
    #and convert into dataframe
    df=pd.DataFrame(xy_tuples, columns = ['Feat_x1','Feat_x2','Feat_x3','Target_y'])
    
    return df

