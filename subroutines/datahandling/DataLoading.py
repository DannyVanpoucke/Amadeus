# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:22:58 2019

Functions related to loading the data-set into a pandas framework

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
import pandas as pd
from typing import Set, Any


def DataFrame2NumpyArrays(Frame: pd.DataFrame, NFeature:int, TargetIndex: int, NInfoCols:int=0) -> tuple:
    """
    Function that transforms the Olympic pandas dataframe into numpy arrays and returns them
    
    parameters:
        - Frame: pandas dataframe
        - NFeature: Number of feature columns
        - TargetIndex: index of the 'target'-column column to select, excluding the feature columns. 
                       If negative, all columns are returned.
        - NInfoCols: number of columns at the start of the row containing codes or information to skip. DEFAULT=0
        
    returns a tuple:
        - array of features, 1 row per data-point
        - array of targets (single column, unless TargetIndex < 0)
        - array of headers of the features
        - array of headers of the targets
        - array of codes (one per row)
    """
    import sys
   
    #copy the data to a numpy-array, column 0 is the "code"
    nparray=Frame.to_numpy()
    #print("The dataset dimension is:", nparray.shape)
    #and now extract the first columns as features
    features=nparray[:,NInfoCols:NInfoCols+NFeature]#note that python does not include the last number of a range...designflaw?
    if (NInfoCols>0):
        codes=nparray[:,0:NInfoCols]
    else:
        codes=None
    
    headersfeature=Frame.columns[NInfoCols:NInfoCols+NFeature]
    if (TargetIndex<0):
        #use all remaining columns
        targets=nparray[:,NInfoCols+NFeature:]
        headerstarget=Frame.columns[NInfoCols+NFeature:]    
    else:
        TargetIndex+=NFeature
        TargetIndex+=NInfoCols
        TargetIndex-=1 #start at zero
        if (TargetIndex<=nparray[0].size):
            targets=nparray[:,TargetIndex:TargetIndex+1]
            headerstarget=Frame.columns[TargetIndex:TargetIndex+1]
        else:
            print("TargetIndex out of scope:",TargetIndex)
            sys.exit()
          
    return features, targets, headersfeature, headerstarget, codes

#############################################################################
#############################################################################
def SelectTarget(Frame: pd.DataFrame, Features: Set[Any], Target: Set[Any]):
    """
    Take a pandas dataframe and return a new dataframe containing only the features + selected Target
    
    parameters:
        - Frame : pandas dataframe
        - Features, Target :  two sets with the names of the columns containing the features and one column of targets.
    
    returns a new dataframe
    """
    df = Frame.copy()
       
    all_cols: Set[Any] = set(Frame.columns)
    diff: Set[Any] = all_cols - Features - Target
    
    df.drop(diff, axis=1, inplace=True) # axis=1: columns
    #print("NEW PANDA:\n",df.head())

    return df


#############################################################################
#############################################################################
def ReadOlympicData(InFile: str, InfoPrint: bool, headerline: list=[1], skiplines:list=[2]) -> pd.DataFrame:
    """
    Function to read the Olympic data from a csv file.
    
    parameters:
        - InFile: string with the filename
        - InfoPrint : boolean indicating if information of the read dataframe should be printed.
        - headerline : an integer list, enumerating the lines with the header-indormation. Note that counting starts at 0. 
        - skiplines  : an integer list, enumerating the lines to be skipped. Note that counting starts at 0.            
    returns a pandas dataframe
    """
    #scikit prefers numpy arrays, but numpy will never succeed 
    #in reading the csv in the format provided, the iso is for UTF8
    df = pd.read_csv(InFile,sep=',',header=headerline,skiprows=skiplines,skip_blank_lines=True, encoding='ISO-8859–1')
    #Print some information of the dataframe if requested
    if (InfoPrint):
        print("Header of the dataset:")
        print(df.head(3))
        print("--------------------------------------")
        print("Type-info of the features and targets:")
        df.info()
        print("--------------------------------------")
        print("A summary of the numerical attributes:")
        print(df.describe())
    
    return df

#############################################################################
#############################################################################
def ReadCompanyData(Company: str, Data:str, InfoPrint: bool, headerline: list=[1], skiplines:list=[2]) -> pd.DataFrame:
    """
    Function to read the Olympic data from a csv file.
    
    parameters:
        - Company: string with the name of the company
        - Data   : string with the name of the dataset
        - InfoPrint : boolean indicating if information of the read dataframe should be printed.
        - headerline : an integer list, enumerating the lines with the header-indormation. Note that counting starts at 0. 
        - skiplines  : an integer list, enumerating the lines to be skipped. Note that counting starts at 0.            
    
    returns a pandas dataframe
    """
    #scikit prefers numpy arrays, but numpy will never succeed 
    #in reading the csv in the format provided, the iso is for UTF8
    import os
    
    DataCSV=Data+".csv"
    InFile=os.path.join("..","datasets",Company,DataCSV)
    
    df = pd.read_csv(InFile,sep=',',header=headerline,skiprows=skiplines,skip_blank_lines=True, encoding='ISO-8859–1')
    #Print some information of the dataframe if requested
    if (InfoPrint):
        print("Header of the dataset:")
        print(df.head(3))
        print("--------------------------------------")
        print("Type-info of the features and targets:")
        df.info()
        print("--------------------------------------")
        print("A summary of the numerical attributes:")
        print(df.describe())
    
    return df


