# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:59:04 2019

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
import pandas as pd
from TLinearModel import TLinearModel
from TPolynomialModel import TPolynomialModel
from TPolynomialENRModel import TPolynomialENRModel
from TPolynomialLassoModel import TPolynomialLassoModel
from TLSSVMModel import TLSSVMModel
from TSVMLinearModel import  TSVMLinearModel
from TSVMPolynomialModel import TSVMPolynomialModel
from TSVM_RBFModel import TSVM_RBFModel
from ModelPipelineFactory import PipelineFactory
from sklearn.pipeline import Pipeline

def ModelFactory(ModelName: str, ObjName:str, Target, Feature: pd.DataFrame, 
                 Target_test, Feature_test: pd.DataFrame, dummyScaler:bool,
                 EnsemblePipeline: Pipeline ,**kwargs):
    """
    Wrapper function defering to the specific modelclass of interest.
    
    parameters:
     - ModelName : string containing the name of the model: "linear", "polynomial"
     - ObjName : string containing the name of this specific object.
     - Target  : The target(s) to use for training
     - Feature : The features to use & transform
     - Target_test  : The target(s) to use for final testing
     - Feature_test : The features to use & transform final testing
     - dummyScaler : True if the standard scaler should be a dummy
     - kwargs  : The keywords and values unique for the specific modelbuilder 
    
    returns: 
     - ModelClass Object : An object of the specified Model Class. This class 
                     gives access to the pipeline object containing the 
                     preprocessing transformations (excluding the fitter function),
                     the (model = ) fitter function to be used (should be an sklearn
                     function with "fit" method), and the transformed features as 
                     obtained by the pipeline. 
    """
    
    Pipeline=PipelineFactory(ModelName,dummyScaler,**kwargs)
    
    if ModelName == 'linear':
        ModelClass = TLinearModel(ObjName,Target, Feature, Target_test, Feature_test, Pipeline)
    elif ModelName == 'polynomial':
        ModelClass = TPolynomialModel(ObjName,Target, Feature, Target_test, Feature_test, Pipeline, EnsemblePipeline, **kwargs)
    elif ModelName == 'poly_lasso':
        ModelClass = TPolynomialLassoModel(ObjName,Target, Feature, Target_test, Feature_test, Pipeline, EnsemblePipeline, **kwargs)
    elif ModelName == 'poly_enr':
        ModelClass = TPolynomialENRModel(ObjName,Target, Feature, Target_test, Feature_test, Pipeline, EnsemblePipeline, **kwargs)
    elif ModelName=='LS-SVM':
        ModelClass = TLSSVMModel(ObjName,Target, Feature, Target_test, Feature_test, Pipeline, EnsemblePipeline, **kwargs)
    elif ModelName == 'SVMLinear':
        ModelClass = TSVMLinearModel(ObjName,Target, Feature, Target_test, Feature_test, Pipeline, **kwargs)
    elif ModelName == 'SVMPolynomial':
        ModelClass = TSVMPolynomialModel(ObjName,Target, Feature, Target_test, Feature_test, Pipeline, **kwargs)
    elif ModelName == 'SVM_RBF':
        ModelClass = TSVM_RBFModel(ObjName,Target, Feature, Target_test, Feature_test, Pipeline,**kwargs)
    else: #default option
        print("!!!!!!! ERROR : THE MODEL ",ModelName," HAS NOT BEEN IMPLEMENTED. DEFAULTING TO LINEAR MODEL.")
        ModelClass = TLinearModel(ObjName,Feature)
    
    #and return to sender
    return ModelClass