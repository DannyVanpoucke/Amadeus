# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:58:53 2020

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures


def PipelineFactory(ModelName: str, dummy: bool, **kwargs):
    """
    Wrapper function defering to the specific modelclass of interest.
    
    parameters:
     - ModelName : string containing the name of the model: "linear", "polynomial"
     - dummy   : Boolean indicating the standard scaler should be a dummy(perfoming no operation) or not
     - kwargs  : The keywords and values unique for the specific modelbuilder (use the same for the pipeline builder) 
    
    returns: 
     - Pipeline Object : An object of the sk-learn Pipeline Class.
    """   
    
    if ((ModelName == 'linear')or(ModelName == 'LS-SVM')):
        if (dummy):
            pipeline = Pipeline([
                ('std_scaler', StandardScaler(with_mean=False, with_std=False)),
                ])
        else:
            pipeline = Pipeline([
                ('std_scaler', StandardScaler()),  #scaling to be centered on 0, with unit variance...since the values are quite different, this will help things
                ])
    elif ((ModelName == 'polynomial')or(ModelName == 'poly_enr')or(ModelName == 'poly_lasso')):
            pipeline = MyPolyPipeline(dummy, **kwargs)            
#    elif ModelName == 'SVMLinear':
#        ModelClass = TSVMLinearModel(ObjName,Target, Feature, Target_test, Feature_test,**kwargs)
#    elif ModelName == 'SVMPolynomial':
#        ModelClass = TSVMPolynomialModel(ObjName,Target, Feature, Target_test, Feature_test,**kwargs)
#    elif ModelName == 'SVM_RBF':
#        ModelClass = TSVM_RBFModel(ObjName,Target, Feature, Target_test, Feature_test,**kwargs)
    else: #default option
        print("!!!!!!! ERROR : THE MODEL ",ModelName," HAS NOT BEEN IMPLEMENTED. DEFAULTING TO LINEAR PIPELINE MODEL.")
        pipeline = Pipeline([
                ('std_scaler', StandardScaler(with_mean=False, with_std=False)),
                ])
    #and return to sender
    return pipeline
    

def MyPolyPipeline(dummy:bool, Degree: int=2, Interaction: bool=False, Bias: bool=True, **kwargs) -> Pipeline:
    if (dummy):#if it is a dummy, it should not create polynomial features either
        pipeline=Pipeline([
            ('std_scaler', StandardScaler(with_mean=False, with_std=False)),#a standard scaler which does nothing
            ]) #don't include the fitter
    else:
            pipeline=Pipeline([
            ('poly_features',PolynomialFeatures(degree=Degree,interaction_only=Interaction ,include_bias=Bias)),# polynamial terms up to degree 3, and no bias column (this would be intercept in case of linear fit)
            ('std_scaler', StandardScaler()),  #scaling to be centered on 0, with unit variance...since the values are quite different, this will help things
            ]) #don't include the fitter
        
    return pipeline
    