# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:02:02 2019

The SVM class with Radial Basis Function kernel.

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
import pandas as pd
from TSVMClass import TSVMClass

class TSVM_RBFModel(TSVMClass):
    """
    Child class of the SVM-class representing a specific type(kernel) of SVM : rbf SVM
    """
    def __init__(self,name,Target, Feature: pd.DataFrame, 
                 Target_test, Feature_test: pd.DataFrame,
                 gamma: str, shrinking: bool, max_iter: int=-1):#rbf keywords
        """
        Constructor to set up a linear SVM model. 
        Any SVM model has two hyper-parameters: epsilon and C. Both are selected 
        using a gridsearch approach, and then storred as "best_e" and "best_C"
    
        parameters:
         - name : the name of the object instance
         - Feature : The features to use & transform
         - Target     : the training target data
         - Target_test: the test target data
         - Feature_test: the untransformed features for testing.
         - gamma : Kernel coefficient for ‘rbf’. 
                   if gamma='scale' is passed then it uses 1/(n_features*X.var()) 
                   as value of gamma,if ‘auto’, uses 1 / n_features.
                   [default = scale]
         - shrinking : Whether to use the shrinking heuristic. [DEFAULT = True]
         - max_iter : Hard limit on iterations within solver, or -1 for no limit. [DEFAULT = -1]
         
        It sets the following properties
            - pipeline : a pipeline object containing the preprocessing transformations (excluding the fitter function)
            #- CVmodel  : the fitter function to be used (should be an sklearn function with "fit" method)
            - model : The model is only set once the CV_model has run a fit-operation
            - feature_tf: the transformed features as obtained by the pipeline    
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        
        super().__init__(name,Target, Feature,Target_test, Feature_test)
        self.nameModel='rbf SVM Model'
        self.name=name
        print("Initialising the child class:",self.nameModel)
        #create a pipeline (can be extended to contain more functions, p67)
        self.pipeline = Pipeline([
            ('std_scaler', StandardScaler()),  #scaling to be centered on 0, with unit variance...since the values are quite different, this will help things
        ]) #don't include the fitter
        
        self.feature_tf = self.pipeline.fit_transform(Feature) #this is a numpy array...
        
        #to keep track of some options:
        self.gamma=gamma
        self.shrinking=shrinking
        self.max_iter=max_iter
        self.model=SVR(kernel='rbf',gamma=self.gamma, shrinking=self.shrinking, max_iter=self.max_iter) #we need a temporary starting model for the grid search...here we need to set all parameters except C and epsilon
        self.best_C=0   #C:penalty giving rise to regularization. Little C means big regularization
        self.best_e=0   #epsilon: width of the street...wants to be as big as possible(as many targets n the street without violations) 
        self.best_score=0 #the score of the "best" gridrearch result
        self.hyperTuned=False #Hyper parameters are not yet tuned
    
    #def fit(self, Target: pd.DataFrame):
        """
        Class-method wrapping the fit-method of the sklearn model
           - Target : a pandas dataframe with the Target data belonging to the 
                   Features provided upon initialisation.
        """
    #    from sklearn.svm import SVR
        
    #    self.model=self.CVmodel
    #    self.model.fit(self.feature_tf,Target)
    #    print("did some fitting, Parent-style:",type(self.model).__name__)
           
    def fitHyper(self):
        """
        Class-method performing a gridsearch to define optimum values for the 
        C and epsilon hyper-parameters.
           --> focus on SVM with linear kernel
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import LeaveOneOut
        from sklearn.svm import SVR
        from GridModule import Grid1D
        import numpy as np
        
        #   [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
        #LoO_CV=LeaveOneOut()
        LoO_CV=5
        Xlow=-12
        Xhigh=6
        NG=7
        Clst=Grid1D(Min=Xlow,Max=Xhigh,NGrid=NG,GridType="log")
        Elst=Grid1D(Min=Xlow,Max=Xhigh,NGrid=NG,GridType="log")
        Glst=Grid1D(Min=Xlow,Max=Xhigh,NGrid=NG,GridType="log")
        
        #coarse grid
        param_grid=[{'C':Clst, 
                     'epsilon':Elst,
                     'gamma': Glst}]

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
        print('coarse: best param= ',grid_search.best_params_)
        print('        best score= ',grid_search.best_score_)
        
        Cval=grid_search.best_params_.get('C')
        Cpow=int(np.log10(Cval))#get rid of rounding errors   
        NG=5
        Clow=Cpow-2
        Chigh=Cpow+2
        
        Eval=grid_search.best_params_.get('epsilon')
        Epow=int(np.log10(Eval))#get rid of rounding errors   
        Elow=Epow-2
        Ehigh=Epow+2
        
        Gval=grid_search.best_params_.get('gamma')
        Gpow=int(np.log10(Gval))#get rid of rounding errors   
        Glow=Gpow-2
        Ghigh=Gpow+2
        Clst=Grid1D(Min=Clow,Max=Chigh,NGrid=NG,GridType="log")
        Elst=Grid1D(Min=Elow,Max=Ehigh,NGrid=NG,GridType="log")
        Glst=Grid1D(Min=Glow,Max=Ghigh,NGrid=NG,GridType="log")
        
        #dens grid
        param_grid=[{'C':Clst, 
                     'epsilon':Elst,
                     'gamma': Glst}]

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
        self.best_score=grid_search.best_score_    
        print('dens :  best param= ',grid_search.best_params_)
        print('        best score= ',grid_search.best_score_)
        self.best_C=grid_search.best_params_.get('C')
        self.best_e=grid_search.best_params_.get('epsilon')
        self.best_gamma=grid_search.best_params_.get('gamma')
        
        #and now we set the actual model
        self.hyperTuned=True
        self.model=SVR(kernel='rbf',
                       gamma=self.best_gamma,
                       C=self.best_C,
                       epsilon=self.best_e,
                       shrinking=self.shrinking,
                       cache_size=500, #cach size in MB, can improve speed
                       max_iter=self.max_iter,  # maximum number of iterations for the solver...default is -1 (infinite)...which it will sometimes take so it shoul dbe avoided
                       )



    def fitHyper_old(self):
        """
        Class-method performing a gridsearch to define optimum values for the 
        C and epsilon hyper-parameters.
           --> focus on SVM with linear kernel
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
        Glow=-3
        Ghigh=3
        Clst=Grid1D(Min=Clow,Max=Chigh,NGrid=NG,GridType="log")
        Elst=Grid1D(Min=Elow,Max=Ehigh,NGrid=NG,GridType="log")
        Glst=Grid1D(Min=Glow,Max=Ghigh,NGrid=NG,GridType="log")
        done=False
        oldScore=1.0E10 #something very bad
        thress=1.0E-3
        
        cnt=0
        while not done:
            cnt+=1
            param_grid=[{'C':Clst, 
                         'epsilon':Elst,
                         'gamma': Glst}]

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
            newScore=grid_search.best_score_
            print(cnt,' best param= ',grid_search.best_params_)
            print(cnt,' best score= ',grid_search.best_score_)
            if abs((newScore-oldScore)/newScore)<thress:
                #score isn't changeing enough we reached the end
                done=True
                print(" +-+-+-+ DONE abs< thress")
            elif (cnt>10):
                #stop running, something bad is going on
                print("WARNING in TSVMLinearModel. Hypertunning takes too many runs. Stopping and hoping for the best.")
                done=True
            else:
                print(" +-+-+-+ ELSE abs< thress")
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
                    
                print("    ++++++ Clow < Cval < Chigh == ",Clow," < ",Cpow," < ",Chigh)
                print("    ++++++ Clst =",Clst )
                    
                Eval=grid_search.best_params_.get('epsilon')
                Epow=int(np.log10(Eval))#get rid of rounding errors
                if ((Epow>Elow)and(Epow<Ehigh)):#it is on the inside
                    Edone=True
                    Elst=[Eval]
                else:#it is on an edge
                    Edone=False
                    if (Epow==Elow):#lower bound
                        Ehigh=Elow+1
                        Elow=Ehigh-NG+1#7 points have 6 steps in between
                    else: #upper bound
                        Elow=Ehigh-1
                        Ehigh=Elow+NG-1
                    Elst=Grid1D(Min=Elow,Max=Ehigh,NGrid=NG,GridType="log")
                
                print("    ++++++ Elow < Eval < Ehigh == ",Elow," < ",Epow," < ",Ehigh)
                print("    ++++++ Elst =",Elst )
                
                Gval=grid_search.best_params_.get('gamma')
                Gpow=int(np.log10(Gval))#get rid of rounding errors
                if ((Gpow>Glow)and(Gpow<Ghigh)):#it is on the inside
                    Gdone=True
                    Glst=[Gval]
                else:#it is on an edge
                    Gdone=False
                    if (Gpow==Glow):#lower bound
                        Ghigh=Glow+1
                        Glow=Ghigh-NG+1#7 points have 6 steps in between
                    else: #upper bound
                        Glow=Ghigh-1
                        Ghigh=Glow+NG-1
                    Glst=Grid1D(Min=Glow,Max=Ghigh,NGrid=NG,GridType="log")
                
                print("    ++++++ Glow < Gval < Ghigh == ",Glow," < ",Gpow," < ",Ghigh)
                print("    ++++++ Glst =",Glst )
                
                
                done=(Cdone and Edone and Gdone)
            oldScore=newScore #store for the next round of while
            
        self.best_score=newScore    
        print('best param= ',grid_search.best_params_)
        print('best score= ',grid_search.best_score_)
        self.best_C=grid_search.best_params_.get('C')
        self.best_e=grid_search.best_params_.get('epsilon')
        self.best_gamma=grid_search.best_params_.get('gamma')
        
        #and now we set the actual model
        self.hyperTuned=True
        self.model=SVR(kernel='rbf',
                       gamma=self.best_gamma,
                       C=self.best_C,
                       epsilon=self.best_e,
                       shrinking=self.shrinking,
                       cache_size=500, #cach size in MB, can improve speed
                       max_iter=self.max_iter,  # maximum number of iterations for the solver...default is -1 (infinite)...which it will sometimes take so it shoul dbe avoided
                       )
    
    def setCoefficients(self):
        """
        Class-method printing the fitting coefficients for a polynomial regression 
        with elastic net regularization
        """
        import numpy as np
        super().setCoefficients()
        
        #--------- hyper parameters -------------------
        self.modelcoef['header_hyperparameter']=[self.coefindex,"The selected hyper-parameters for the ",self.nameModel," are:"]
        line="  - C-penalty = %0.3e  " % (self.best_C )
        self.modelcoef['C-penalty']=[-(self.coefindex+1),line]
        line="  - epsilon   = %0.3e  " % (self.best_e )
        self.modelcoef['epsilon']=[-(self.coefindex+2),line]
        line="  - gamma   = %0.3e  " % (self.best_gamma )
        self.modelcoef['gamma']=[-(self.coefindex+3),line]
        line="  - score   = %0.3f  " % (self.best_score )
        self.modelcoef['score']=[-(self.coefindex+4),line]
        line="  - fit OK   = %r  " % (self.model.fit_status_ == 0 )
        self.modelcoef['fitOK']=[-(self.coefindex+5),line]
        #------------usual coefficients-----------------
        self.coefindex+=6
        self.modelcoef['header_coef']=[self.coefindex,"The coefficients for each target (one per row) are given by:"]
        #self.modelcoef['coef_']=[self.coefindex+1,np.matrix(self.model.dual_coef_)]
        self.modelcoef['coef_']=[self.coefindex+1," "]
        self.modelcoef['header_intercept']=[self.coefindex+2,"The intercepts for each target (one per row) are given by:"]
        self.modelcoef['intercept_']=[self.coefindex+3,np.matrix(self.model.intercept_)]
        self.coefindex+=4

