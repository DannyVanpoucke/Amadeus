# -*- coding: utf-8 -*-
#to get these to print all threads in spyder :
# Run > Configuration per file > Execute in an external system terminal
# And then add a read at the end of the program to prevent the console of closing
"""
Created on Fri Jan 10 16:19:58 2020

This script uses the Amadeus framework on an artificial dataset, with the aim of establishing
the virtue of the creation of averaged models.

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
import time
import os
import numpy as np
import pandas as pd
#The source is in various files in the folder subroutines,
# so need to add this to the path first
import sys
sys.path.append("subroutines/datahandling")
from DataLoading import DataFrame2NumpyArrays, SelectTarget
from HPCTools import set_num_threads

from TAmadeusFrameWork import TAmadeusFrameWork
from ArtificialDataSets import Create1DData, CreateLinear3D, Lin1DFunc, LinNDFunc, SinFunc
from MLpaperPostProcess import RunPostProcess_MLpaper

if __name__ == '__main__':
    print("=============Start of Framework==============")
    start = time.perf_counter_ns()
    #Some initialisation
    parprocs = -1   #give me the number of physical CPU's...
                    #this is wrong on BReniac, because there are 2 sockets...no solution--> -2
    if len(sys.argv)>=2: #unless the user provided the number to use as argument
        parprocs = int(sys.argv[1])
        
    set_num_threads(1) #set number of BLAS/OPENMP/MKL-threads that sklearn/numpy is allowed to spawn
        
    #datasizes=[10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 750, 1000]
    datasizes=[20]
    predsize=1000
    Gspread=0.10  # for Sin3x: 0.05, linear 1D: 0.75
    MyArtificialModel=[0.50, 6.0]
    MyArtModelFunc = SinFunc  #SinFunc    #function to generate the data
    MyFitModelFunc = LinNDFunc  #LinNDFunc  #the ML-fit function, self-coded
    MyPredict = Create1DData(Func1D=MyArtModelFunc, length=predsize, intercept=MyArtificialModel[0], 
                             coefficient=MyArtificialModel[1], dist='normal',dwidth=Gspread)
    MyFrame_Base = Create1DData(Func1D=MyArtModelFunc, length=np.amax(datasizes), intercept=MyArtificialModel[0], 
                                coefficient=MyArtificialModel[1], dist='normal',dwidth=Gspread)
    
    NCols=20 #11 columns: index of run,etc for linear fit # Poly6:=16
    #artmodname="1Dlin"
    artmodname="1Dsin3x"
    
    TargetOfChoice=1
    NFeatures=1
    NSKFeatures=10 #number of features in sk-learn model 1:linear fit, 3: poly-3
    
    NRuns=1000
    TTsplit=0.2 #20% test
    
    for ds in datasizes:
        start_ds = time.perf_counter_ns()
        InF = "Artificial "+artmodname+" model "+str(ds)+" datapoints."
        print(" 1. Loading data :", InF)
        MyFrame = pd.DataFrame(MyFrame_Base).iloc[0:ds].copy(deep=True)#no -1 is needed as python has this weirdt behaviour of not including the last element
        features, targets, headersfeature, headerstarget, codes=DataFrame2NumpyArrays(MyFrame, NFeature=NFeatures, TargetIndex=1)
        
        #store the data-sets in a file:
        #first the train/test data
        #then the 1000 point prediction data
        basedpf="datapoints_"+artmodname+"_"
        datapointfile=basedpf+str(ds)+".dat"
        baserf="results_"+artmodname+"_"
        resultsfile=baserf+str(ds)+".dat"
        if os.path.exists(datapointfile):
            os.remove(datapointfile) #clear the file before we start
        if os.path.exists(resultsfile):
            os.remove(resultsfile)   #clear the file before we start
        
        dpf=open(datapointfile,"a+") #open file for appending
        dpf.write("#  %d \n" % ds)
        dpf.close()
        MyFrame.to_csv(datapointfile,header=None, index=None, sep=' ', mode='a')
        dpf=open(datapointfile,"a+") #open file for appending
        dpf.write("\n \n# %d \n" % predsize)
        dpf.close()
        MyPredict.to_csv(datapointfile,header=None, index=None, sep=' ', mode='a')
            
        
        print(" 2. Preparation and selection of data")
        #The Set function makes a set of the array...
        #  the target needs to be put in [], otherwise 
        #  we get a set of characters
        
        ModelFrame=SelectTarget(MyFrame,set(headersfeature.values),set([headerstarget[TargetOfChoice-1]]))#-1 as Python starts at zero
        
        dpf=open(resultsfile,"a+") #open file for appending
        dpf.write("#  %d    %d \n" % (NRuns+1, NCols))
        dpf.close()
        
        print(" 3.a. Starting model-framework: full dataset")
        Amadeus=TAmadeusFrameWork(dataset=ModelFrame,njobs=parprocs,test_size=0,
                                  maxRuns=1, printFileStatistics=resultsfile,PreStdScaler=True)
        failcnt=Amadeus.RunModelEnsemble()
        
        print(" 3.b. Starting model-framework: train-test splitting")
        Amadeus=TAmadeusFrameWork(dataset=ModelFrame,njobs=parprocs,test_size=TTsplit,
                                  maxRuns=NRuns, printFileStatistics=resultsfile,PreStdScaler=True)
        failcnt=Amadeus.RunModelEnsemble()
        
        
        end_ds = time.perf_counter_ns()
        print("Total time:", (end_ds-start_ds)/1000000000, "s")
        print("Sanity checks needed : ",failcnt," reshuffles")
        
  

    print("Starting Post-processing")
    RunPostProcess_MLpaper(basedata=basedpf,baseresult=baserf,datasizes=datasizes,
                           predictionData=MyPredict, NumDim=NFeatures, NumSKDim=NSKFeatures,
                           theoryModel=MyArtificialModel,
                           modelFunction=MyArtModelFunc,
                           fitFunction=MyFitModelFunc)
    
    end = time.perf_counter_ns()
    print("Totals time:", (end - start)/1000000000, "s")
    
    if len(sys.argv)<2: #not used in another program...so ask the user to finish the terminal
        inp=input("gimme something to read")
