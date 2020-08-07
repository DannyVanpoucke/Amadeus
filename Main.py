# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:19:26 2020

This is a main jobscript. It should just prepare (aka load) the data,
and feed it to our framework. 

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
import time
#The source is in various files in the folder subroutines,
# so need to add this to the path first
import sys
sys.path.append("subroutines/datahandling")
from DataLoading import ReadOlympicData, DataFrame2NumpyArrays, SelectTarget
from TAmadeusFrameWork import TAmadeusFrameWork



if __name__ == '__main__':
    print("=============Start of Framework==============")
    start = time.perf_counter_ns()
    #Some initialisation
    parprocs = -1   #give me the number of physical CPU's...
                    #this is wrong on BReniac, because there are 2 sockets...no solution--> -2
    if len(sys.argv)>=2: #unless the user provided the number to use as argument
        parprocs = int(sys.argv[1])
        
    
    InF = "../datasetOlympic25.csv"
    print(" 1. Loading data :", InF)
    OlympicFrame = ReadOlympicData(InF, False)
    features, targets, headersfeature, headerstarget, codes=DataFrame2NumpyArrays(OlympicFrame,NFeature=3, TargetIndex=-1,NInfoCols=1)
    
    
    print(" 2. Preparation and selection of data")
    #The Set function makes a set of the array...
    #  the target needs to be put in [], otherwise 
    #  we get a set of characters
    TargetOfChoice=3
    ModelFrame=SelectTarget(OlympicFrame,set(headersfeature.values),set([headerstarget[TargetOfChoice]]))
    
    print(" 3. Starting model-framework")
    Amadeus=TAmadeusFrameWork(njobs=parprocs,dataset=ModelFrame,maxRuns=4)
    failcnt=Amadeus.RunModelEnsemble()
    
    end = time.perf_counter_ns()
    print("Total time:", (end-start)/1000000000, "s")
    print("Sanity checks needed : ",failcnt," reshuffles")
    if len(sys.argv)<2: #not used in another program...so ask the user to finish the terminal
        inp=input("gimme something to read")
        
    
    
    
    
    
    