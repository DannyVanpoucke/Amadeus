# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:01:25 2019

The ModelClass parent class. All the actual 
models are child classes with (if needed) overloaded methods

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
import pandas as pd
import numpy as np
import sys
sys.path.append("../ParallelResults")
from TModelResults import TModelResults
from TDataPoint import TDataPoint
from sklearn.exceptions import ConvergenceWarning

class TModelClass:
    """
    The parent class containing all required methods and properties. 
    
    Properties:
      - nameModel : string containing the name of the model
      - name : the name of this specific instance of the model
      - model: the sklearn model used
      - pipeline: a pipeline object containing the preprocessing 
              transformations (excluding the fitter function) 
      - target    : the training target data (pandas dataframe)
      - feature   : the original training feature data (remains pandas, so can be recopmbined with target)
      - feature_tf: the transformed features as obtained by the pipeline (this is a numpy array) 
      - target_test: the test target data
      - feature_test: the untransformed features for testing.
      - modelcoef : dictionary keeping track of all relevant model-parameters.
                      The values are lists with index 0 giving their print-line.
      - isAverage : Boolean indicating if this is an "average" model
      - hasCI     : Boolean indicating if a CI was calculated for the average model
      - CI        : Dictionary with the CI for the parameters of the averaged model
      - coefindex : integer giving the line index of coefficinet data to add in modelcoef.
      - sanityThresshold : floating point thresshold parameter used in sanitychecks. [DEFAULT= 1e+9]
    """
    def __init__(self, name: str, Target, Feature: pd.DataFrame, 
                 Target_test, Feature_test: pd.DataFrame, **kwargs):
        """
        Constructor of the class, initialising with a default name
        It requires:
            - name : the name of the object instance
            - Feature : a pandas dataframe containing the features
            - kwargs : a list of possible arguments provided specifically 
                    for each child class
        """
        self.nameModel='NoModelClass'
        self.name=name
        self.target=Target
        self.feature=Feature
        self.target_test=Target_test
        self.feature_test=Feature_test #transformation happens only at quality assesment
        self.modelcoef = dict()
        self.isAverage = False
        self.hasCI = False
        self.CI = dict()
        self.coefindex = 0 # the current line index of the modelcoef to add
        self.sanityThresshold = 1.0e9
        
        
    def fit(self):
        """
        Class-method wrapping the fit-method of the sklearn model.
        
           - Target : a pandas dataframe with the Target data belonging to the 
                   Features provided upon initialisation.
        """
        self.model.fit(self.feature_tf,self.target)
        print("FIT COEFF=",self.model.coef_," INTERCEPT=",self.model.intercept_)
        
        self.setCoefficients()
        print("did some fitting, Parent-style:",type(self.model).__name__)
        
    def fitSanityCheck(self):
        """
        Class method which should cover/deal with failures of sklearn.
        
        Due to the small data-sets, sklearn sometimes fails rather miserably 
        (even in case of a linear regression). This function should add the 
        "I" in AI, and try to catch and remediate the problem. This function needs to 
        be implemented for each model separately.
        
        Calling this function should be performed by the user. Placing it in the 
        fit function of the model creates a recursive loop, which may not end well. 
        """
        pass #body with no content
    
    def predict(self, Feature: pd.DataFrame) -> list:
        """
        Class-method wrapping around the predict method of the sklearn-model
        """
        return self.model.predict(Feature)
    
    def predictError(self, Feature: pd.DataFrame)-> tuple:
        """
        Class-method wrapping around the predict method of the sklearn-model, and 
        performing additional calculations needed to calculate confidence interval based
        errorbars.
        
        parameters:
            Feature: the features of the data to predict
            
        returns:
            a tuple of lists: Targets, CI (every row gives the CI for 1 target, first column low, second column high)
        """
        
        predict=self.model.predict(Feature)       # although a panda goes in, an nd-array comes out
        CI=np.array(list([i]*2 for i in predict)) # no errorbars yet...just placeholder
        return predict, CI
    
    #@ignore_warnings(category=ConvergenceWarning)
    def CV_score(self, Target: pd.DataFrame, CV: int):
        """
        Class-method wrapping the cross_val_score functionality of sklearn.
        
        Variables:
            - CV : number of folds for cross-validation
            - Target : a pandas dataframe with the Target data belonging to the 
                   Features provided upon initialisation.
        """
        import warnings
        from sklearn.model_selection import cross_val_score
        
        #!!!! NOTE THAT cross_val_score PERFORMS A SET OF FITS, 
        #!!!!  NOT TAKING THE intercept_ AND coef_ VALUES OF OUR "fit"
        #!!!!  AS SUCH THESE VALUES ARE OF LITTLE PRACTICAL USE AS THEY 
        #!!!!  ARE HEAVILY DEPENDANT ON THE SUBSET. GOOD FOR A BALL-PARK GUESS...MAYBE
        #catching away the warnings to keep output clean
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always",category=ConvergenceWarning)
            mse=cross_val_score(self.model , self.feature_tf , Target, 
                              scoring='neg_mean_squared_error', 
                              cv = CV,
                              error_score = np.nan)
        return mse
        
    def setAverageCoefficients(self,EnsembleData: TModelResults, setCI: bool):
        """
        Use the ensemble data to create an "average" model, and set the "coefficients"
        in the current model. This should be performed in each model separately
        
        parameters:
            - EnsembleData : a TModelResults object containing the arrays with quality data and coefficients/parameters of the model for all runs
            - setCI        : if True, calculate the 95% confidence interval
        """
        raise NotImplementedError("Please Implement this method")
    
    def printAverageCoefficients(self, File: str=None):
        """
        Print a block of information to a file, containing the averaged coefficients.
        
        parameters:
            - self:
            - File: string containing a filename, if None standard output is used. Default=None
        """
        raise NotImplementedError("Please Implement this method")
    
    
    def setCoefficients(self):
        """
        Class-method collecting and storing the fitting coefficients in the object
        """
        self.modelcoef.clear() #this is the first place it should appear...clear it if approached a second time
        self.coefindex=0       #correct start of indexing
        self.modelcoef['header']=[self.coefindex,"--------- Model-coefficients :",self.name," ------"]
        self.coefindex+=1
        #actual implementation should be done in each of the child-classes
   
    def setSanityThresshold(self, thress: float):
        self.sanityThresshold=thress

     
    def getQualityMeasures(self) -> TDataPoint:
        """
        Class-method returning the quality measures of the current model.
        No printing should happen here!
        
        return:
            - datapoint: a TDataPoint object containing all relevant information.
        """
        from sklearn.metrics import mean_squared_error,mean_absolute_error
        from sklearn.model_selection import LeaveOneOut
        import numpy as np
        
        #the training data
        mean= np.mean(self.target)
        std = np.std(self.target)
        feature_pred = self.predict(self.feature_tf)
        RMSEtrain = np.sqrt(mean_squared_error(self.target, feature_pred))
        MAEtrain = mean_absolute_error(self.target, feature_pred)
        #Leave-One-Out Cross-validation
        LoO_CV=LeaveOneOut()
        scores = np.sqrt(-self.CV_score(self.target,CV=LoO_CV)) #minus because of the NEGATIVE_MSE   --> old: CV=self.feature_tf.shape[0]
        trainLoO_mean=scores.mean()
        trainLoO_2sig=scores.std()*2.0
        #5-fold Cross-validation
        scores = np.sqrt(-self.CV_score(self.target,CV=5)) #minus because of the NEGATIVE_MSE
        trainCV5_mean=scores.mean()
        trainCV5_2sig=scores.std()*2.0
        
        #The test data
        feature_test_tf = self.pipeline.transform(self.feature_test) #No fitting on the test-data
        feature_pred_test=self.predict(feature_test_tf)
        RMSEtest=np.sqrt(mean_squared_error(self.target_test, feature_pred_test))
        MAEtest = mean_absolute_error(self.target_test, feature_pred_test)
        
        #now add the reults to our model-results
        datapoint=TDataPoint()
        datapoint.setName(self.name)
        datapoint.setNameModel(self.nameModel)
        datapoint.setQuality(mean, std, RMSEtrain, MAEtrain, RMSEtest, MAEtest, 
                            trainLoO_mean, trainLoO_2sig, 
                            trainCV5_mean, trainCV5_2sig)
        datapoint.setModelCoef(self.modelcoef)
        return datapoint
        
        
        
        