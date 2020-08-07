# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:12:13 2019

Because SVM's come in different shapes and forms, this class will be the parentclass
for the various SVMs. Specifics of the different kernels should be dealt with
in the child SVM-model-classes. General access should be provided through this
parent SVM-class.

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
#import sys ... for some reason this does not work here,and needs to be in the main program file
#sys.path.append("../maths")
import pandas as pd
from TModelClass import TModelClass

class TSVMClass(TModelClass):
    """
    Child class representing the SVM classes
    """
    def fit(self):
        """
        Class-method wrapping the fit-method of the sklearn model
           - Target : a pandas dataframe with the Target data belonging to the 
                   Features provided upon initialisation.
        """
        
        if (not self.hyperTuned):
            self.fitHyper() #this also sets the actual model
            
        self.model.fit(self.feature_tf,self.target)
        self.setCoefficients()
        print("did some fitting, Parent-style:",type(self.model).__name__)
        
    def fitHyper(self):
        """
        Class-method performing a gridsearch to define optimum values for the 
        C and epsilon hyper-parameters.
           
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import LeaveOneOut
        from sklearn.svm import SVR
        from GridModule import Grid1D
        import numpy as np
        
        #   [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
        LoO_CV=LeaveOneOut()
        Clow=-3
        Chigh=3
        NG=7
        Elow=-3
        Ehigh=3
        Clst=Grid1D(Min=Clow,Max=Chigh,NGrid=NG,GridType="log")
        Elst=Grid1D(Min=Elow,Max=Ehigh,NGrid=NG,GridType="log")
        done=False
        
        while not done:
            param_grid=[{'C':Clst, 
                         'epsilon':Elst}]

            grid_search=GridSearchCV(self.model,param_grid,
                                 scoring='neg_mean_squared_error', #RMSE^2, neg-> higher value is better
                                 n_jobs=-1,             # parallel, all cores
                                 pre_dispatch='n_jobs', # spawn as many jobs as there will run in parallel
                                 cv=LoO_CV,             # Leave-One-Out Cross-validation (small data)
                                 error_score=np.NaN,    # Some parameter-combos may lead to failure to fit. (Robustness to Failure)
                                                        #  Instead of crashing the entire CV job, just set 
                                                        #  their score to NaN, and contine.. as such they 
                                                        #  are ignored in the end compilation of the best result
                                
                                 )
            grid_search.fit(self.feature_tf,self.target)
            print('best param= ',grid_search.best_params_)
            print('best score= ',grid_search.best_score_)
            #now check if we are not on the edge:
            Cval=grid_search.best_params_.get('C')
            Cpow=int(np.log10(Cval))#get rid of rounding errors
            if ((Cpow>Clow)and(Cpow<Chigh)):#it is on the inside
                Cdone=True
                Clst=[Cval]
            else:#it is on an edge
                Cdone=False
                if (Cpow==Clow):#lower bound
                    Chigh=Clow+1
                    Clow=Chigh-NG+1#7 points have 6 steps in between
                else: #upper bound
                    Clow=Chigh-1
                    Chigh=Clow+NG-1
                Clst=Grid1D(Min=Clow,Max=Chigh,NGrid=NG,GridType="log")
                
            Eval=grid_search.best_params_.get('epsilon')
            Epow=int(np.log10(Eval))#get rid of rounding errors
            if ((Epow>Elow)and(Epow<Ehigh)):#it is on the inside
                Edone=True
                Elst=[Cval]
            else:#it is on an edge
                Edone=False
                if (Epow==Elow):#lower bound
                    Ehigh=Elow+1
                    Elow=Ehigh-NG+1#7 points have 6 steps in between
                else: #upper bound
                    Elow=Ehigh-1
                    Ehigh=Elow+NG-1
                Elst=Grid1D(Min=Elow,Max=Ehigh,NGrid=NG,GridType="log")
            
            done=(Cdone and Edone)
        

        print('best param= ',grid_search.best_params_)
        print('best score= ',grid_search.best_score_)
        self.best_C=grid_search.best_params_.get('C')
        self.best_e=grid_search.best_params_.get('epsilon')
        
        #and now we set the actual model
        self.hyperTuned=True
        self.model=SVR(kernel='linear',
                       C=self.best_C,
                       epsilon=self.best_e,
                       cache_size=500, #cach size in MB, can improve speed
                       )
        
    def fitSanityCheck(self)->int:
        """
        Class method which should cover/deal with failures of sklearn.
        
        Can we cover all SVM sanity in 1 function?
        
        This function returns the number of shuffles needed to regain sanity.
        """
        import sys
        
        cnt=0
        insane=False #larger than 5% warnings
        
        if (cnt>0):#update the coefficients
            self.setCoefficients()
            
        if insane:
            print("EPIC FAIL, n_alphas=10K is just crazy. Attempts at sanity failed in the ",self.name,". Terminating this sick job!")
            sys.exit()
        
        return cnt
        
        
        
        
        
        
        
        
        
        
        
        