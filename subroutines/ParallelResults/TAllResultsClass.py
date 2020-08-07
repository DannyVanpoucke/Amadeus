# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:11:00 2019

Class containing all our results of the various calculations.


@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
from TModelResults import TModelResults
from TDataPoint import TDataPoint
from typing import List


class TAllResultsClass:
    """
    All information and functionality relevant to collecting and analysing data
    should be collected into this class.
    
    The data is collected per model, and each model's data is accessed via a 
    dictionary: (model-name)-->(model-data)
    
    It has the following properties:
        - NModels : integer value giving the number of models
        - Results : dictionary of model-name:TModelResults pairs
        - NDataRuns : integer initialising the number of dataruns for each ModelResult. Default is zero.
    
    
    """
    def __init__(self, NDataRuns: int=0):
        """
        Constructor, creating an empty dict.
        """
        self.NModels=0
        self.Results=dict()
        self.NDataRuns=NDataRuns
    
    def addModel(self, Name, NameModel : str):
        """
        Extends the dictionary with an empty modelResults
        
            - Name : string containing the name of the model (unique)
            - NameModel : string containing a general name of the model
        """
        import sys
        
        self.NModels +=1
        if Name not in self.Results:
            self.Results[Name] = TModelResults(Name, NameModel, self.NDataRuns)
        else:
            print("ERROR: duplicate model in RESULTS. Terminating")
            sys.exit()
    
    def getNModels(self)-> int:
        """
        Getter for the number of models
        """
        return self.NModels
    
    def setDataPoints(self, DataPointArray: List[TDataPoint], Print: bool=False, PrintCoef: bool=False ):
        """
        Puts the results collected in an array of TDataPoints into the 
        arrays of the specific TModelResults.
        
        parameters:
            - DataPointArray : array of TDataPoint objects
            - Print : optional boolean indicating of a quality results block needs to be written.
            - PrintCoef : optional boolean indicating if the coefficients (hyper & fitting) should be
                        printed in the datablock.
        """
        
        for dp in DataPointArray:
            self.Results[dp.name].setDataPoint(dp)
                        #dp.index, dp.mean, dp.std, dp.RMSEtrain, dp.MAEtrain,
                        #dp.RMSEtest, dp.MAEtest, dp.trainLoO_mean, dp.trainLoO_2sig, 
                        #dp.trainCV5_mean, dp.trainCV5_2sig, dp.modelCoef)
            
        if Print:
            for dp in DataPointArray:
                self.Results[dp.name].printQualityDataPoint(dp.index, PrintCoef)
                
    def setDataPoint(self, dp: TDataPoint, Print: bool=False, PrintCoef: bool=False ):
        """
        Puts the results collected in an array of TDataPoints into the 
        arrays of the specific TModelResults.
        
        parameters:
            - DataPointArray : array of TDataPoint objects
            - Print : optional boolean indicating of a quality results block needs to be written.
            - PrintCoef : optional boolean indicating if the coefficients (hyper & fitting) should be
                        printed in the datablock.
        """
        
        self.Results[dp.name].setDataPoint(dp)
                        #dp.index, dp.mean, dp.std, dp.RMSEtrain, dp.MAEtrain,
                        #dp.RMSEtest, dp.MAEtest, dp.trainLoO_mean, dp.trainLoO_2sig, 
                        #dp.trainCV5_mean, dp.trainCV5_2sig, dp.modelCoef)
            
        if Print:
            self.Results[dp.name].printQualityDataPoint(dp.index, PrintCoef)
            
            
            
    
    
    
