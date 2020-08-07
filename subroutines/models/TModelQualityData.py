# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:11:59 2020

Class object containing a uniform data-quality measure 
that can be generated for all models.

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
from TModelResults import TModelResults
from Bootstrap import TBootstrap
import numpy as np

class TModelQualityData:
    """
    Class containing the quality measures of an averaged model
    """
    def __init__(self, EData: TModelResults):
        """
        Constructor calculating all the measures.
        
        MAE & RMSE:
            For our ensemble, we can make use of the out-of-bag (oob) estimates of all the separate instances.
            The estimate for the ensemble is then the average of the (oob) estimates. A 2*sigma value is 
            calculated from the standard deviation over the ensemble. In addition, we calculate 
            the CI via boostrapping over the entire ensemble.
            
        The quality results are stored in a dictionary.
        
        parameters:
            - EData : The data obtained for a full ensemble, stored in a TModelResults object
        
        It sets the following properties:
            - EnsembleSize : number of samples in the ensemble (i.e. sub-system runs)
            - Quality : dictionary of quality mearsures
                
        
        """
        self.EnsembleSize=EData.NData
        self.Quality=dict()
        
        ####### OUT-OF-BAG PREDICTION ##################
        ################  MAE  #########################
        mean=np.mean(EData.MAEfulltest,axis=0)
        sig2=np.std(EData.MAEfulltest,axis=0)*2.0
        
        boot=TBootstrap(data=EData.MAEfulltest,Func=np.mean)
        boot.NPbootstrap(n_iter=2000, Jackknife=True)
        avgm, avgp = boot.ConfidenceInterval(CItype="BCa",alpha=0.05,n_samples=2000)#95%confidence interval
        self.Quality['MAEoob']=list([mean, sig2, avgm, avgp])
        
        ################  RMSE  #########################
        mean=np.mean(EData.RMSEfulltest,axis=0)
        sig2=np.std(EData.RMSEfulltest,axis=0)*2.0
        
        boot=TBootstrap(data=EData.RMSEfulltest,Func=np.mean)
        boot.NPbootstrap(n_iter=2000, Jackknife=True)
        avgm, avgp = boot.ConfidenceInterval(CItype="BCa",alpha=0.05,n_samples=2000)#95%confidence interval
        self.Quality['RMSEoob']=list([mean, sig2, avgm, avgp])
        
        ####### IN-BAG PREDICTION (AKA TRAINING) ##################
        ################  MAE  #########################
        mean=np.mean(EData.MAEfulltrain,axis=0)
        sig2=np.std(EData.MAEfulltrain,axis=0)*2.0
        
        boot=TBootstrap(data=EData.MAEfulltrain,Func=np.mean)
        boot.NPbootstrap(n_iter=2000, Jackknife=True)
        avgm, avgp = boot.ConfidenceInterval(CItype="BCa",alpha=0.05,n_samples=2000)#95%confidence interval
        self.Quality['MAEtrain']=list([mean, sig2, avgm, avgp])
        
        ################  RMSE  #########################
        mean=np.mean(EData.RMSEfulltrain,axis=0)
        sig2=np.std(EData.RMSEfulltrain,axis=0)*2.0
        
        boot=TBootstrap(data=EData.RMSEfulltrain,Func=np.mean)
        boot.NPbootstrap(n_iter=2000, Jackknife=True)
        avgm, avgp = boot.ConfidenceInterval(CItype="BCa",alpha=0.05,n_samples=2000)#95%confidence interval
        self.Quality['RMSEtrain']=list([mean, sig2, avgm, avgp])
        
        
        
    def QualitiesText(self)->str:
        
        text=" Prediction quality of training set:\n"
        lineMAE =" - MAE    = %0.5f (+- %0.5f )  95%%CI=[ %0.5f , %0.5f ] \n" % (self.Quality['MAEtrain'][0],
                                    self.Quality['MAEtrain'][1], self.Quality['MAEtrain'][2], 
                                    self.Quality['MAEtrain'][3])
        lineRMSE=" - RMSE   = %0.5f (+- %0.5f )  95%%CI=[ %0.5f , %0.5f ] \n" % (self.Quality['RMSEtrain'][0],
                                    self.Quality['RMSEtrain'][1], self.Quality['RMSEtrain'][2], 
                                    self.Quality['RMSEtrain'][3])
        text=text+lineMAE+lineRMSE
        
        text=text+"\n Out-Of-Bag error estimates:\n"
        lineMAE =" - MAE    = %0.5f (+- %0.5f )  95%%CI=[ %0.5f , %0.5f ] \n" % (self.Quality['MAEoob'][0],
                                    self.Quality['MAEoob'][1], self.Quality['MAEoob'][2], 
                                    self.Quality['MAEoob'][3])
        lineRMSE=" - RMSE   = %0.5f (+- %0.5f )  95%%CI=[ %0.5f , %0.5f ] \n" % (self.Quality['RMSEoob'][0],
                                    self.Quality['RMSEoob'][1], self.Quality['RMSEoob'][2], 
                                    self.Quality['RMSEoob'][3])
        text=text+lineMAE+lineRMSE+"\n"
        
        return text
        
    def getQuality(self,qualityname:str=None)->list:
        """
        Returns the values of a specific quality measure as a list: Mean-value, 2 sigma, CIlow, CIhigh
        parameter:
            - qualityname: string DEFAULT: MAEoob
                + MAEoob
                + RMSEoob
                + MAEstrain
                + RMSEstrain
        """
        QUALLST=["MAEoob","RMSEoob","MAEtrain","RMSEtrain"]
        if (qualityname is None) or (qualityname not in QUALLST):
            qualityname="MAEoob"
        
        return self.Quality[qualityname]
            
        
            
