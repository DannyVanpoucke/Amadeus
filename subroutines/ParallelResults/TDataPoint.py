# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:13:19 2019

Class to collect relevant information when running models in parallel. The 
results are storred in separate datapoints (this class) and later collected
(i.e. copied) to the arrays of the TAllResultsClass object

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""

class TDataPoint:
    """
    Class to collect relevant information when running models in parallel. The 
    results are storred in separate datapoints (this class) and later collected
    (i.e. copied) to the arrays of the TAllResultsClass object.
    
    properties:
        - index: integer index of the run this data belongs to
        - name : string containing the name of the model
        - mean : the mean of the data-set
        - std  : standard deviation of the dataset
        - RMSEtrain : RMSE of the training data
        - RMSEtest  : RMSE of the test data
        - MAEtrain  : Mean-Absolute-Error of the training data
        - MAEtest   : Mean-Absolute-Error of the test data
        - trainLoO_mean : Mean value of the RMSE values of Leave-one-Out on the training data
        - trainLoO_2sig : 2-sigma value of the RMSE values of Leave-one-Out on the training data
        - trainCV5_mean : Mean value of the RMSE values of 5-fold-Cross-Validation on the training data
        - trainCV5_2sig : 2-sigma value of the RMSE values of 5-fold-Cross-Validation on the training data
        - modelCoef     : dictionary with all model-related parameters (hyper & coefficients)
    """    
    def __init__(self):
        self.index=-1
        self.name="NoName"
        self.nameModel="NoName"
        self.mean=None
        self.std=None
        self.RMSEtrain=None
        self.RMSEtest=None
        self.MAEtrain=None
        self.MAEtest=None
        self.trainLoO_mean=None
        self.trainLoO_2sig=None
        self.trainCV5_mean=None
        self.trainCV5_2sig=None
        self.modelCoef=None
        
    def setQuality(self,mean, std, RMSEtrain, MAEtrain, RMSEtest, MAEtest, trainLoO_mean, 
                       trainLoO_2sig, trainCV5_mean, trainCV5_2sig):
        self.mean=mean
        self.std=std
        self.RMSEtrain=RMSEtrain
        self.RMSEtest=RMSEtest
        self.MAEtrain=MAEtrain
        self.MAEtest=MAEtest
        self.trainLoO_mean=trainLoO_mean
        self.trainLoO_2sig=trainLoO_2sig
        self.trainCV5_mean=trainCV5_mean
        self.trainCV5_2sig=trainCV5_2sig
        
    def setIndex(self, i: int):
        self.index=i
        
    def setName(self, name: str):
        self.name=name
        
    def setNameModel(self, nameModel: str):
        self.nameModel=nameModel
        
    def setModelCoef(self, coefs: dict):
        self.modelCoef=coefs #is this sufficient or do we need an explicit deep copy?


