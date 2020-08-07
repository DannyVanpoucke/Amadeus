# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:54:25 2019

The Build-Model-List needs to become a class such 
that we can start it up, and extend it at will

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""

class TModelList:
    """
    A class which contains a list of models to use. 
    
    Properties:
      - mdc : a dictionary of dictionaries.
              The parent dictionary has the model name as key, 
              and a child dictionary containing all options+values 
              as key-value pairs. 
    """
    def __init__(self):
        """
        Constructor of the class, initialising with an empty dictionary
        """
        self.mdc=dict()
        
    def GetModelDictionary(self) -> dict:
        """
        Class-method returning the model dictionary (a dictionary of 
        dictionaries)
        """
        return self.mdc
    
    def AddLinearModel(self):
        """
        Class-method wich adds a linear model to the list.
        """
        odc=dict()
        odc['modelType']='linear'
        self.mdc['linear']=odc
    
    def AddPolynomialModel(self,minDegree: int=2, maxDegree: int=5, 
                           interaction: bool=False, bias: bool=False):
        """
        Class-method wich adds a set of polynomial models to the list.
        The polynomials will have degrees going from minDegree..maxDegree.
        
        parameters:
         - minDegree : Integer indicating the minimal degree of the polynomial.[DEFAULT = 2]
         - maxDegree : Integer indicating the maximal degree of the polynomial.[DEFAULT = 5]
         - interaction : Boolean indicating if only interaction features are produced. [DEFAULT = False]
         - bias : Boolean indicating if a bias column is added (i.e., where all powers 
                  are zero, acts as intercept in linear model). [DEFAULT = False]
        """
        for r in range(minDegree,maxDegree+1): #+1 since python does not include the end-point
            odc=dict()
            odc['modelType']='polynomial'
            odc['Degree']=r
            odc['Interaction']=interaction
            odc['Bias']=bias
            #as it is a dictionary, we doan't want to get duplicate names
            s='polynomial_D'+str(r)+'_'+str(interaction)[0:1]+str(bias)[0:1]
            self.mdc[s]=odc
            
    def AddRegularizedPolyModel(self,minDegree: int=2, maxDegree: int=5, 
                           interaction: bool=False, bias: bool=False, N_Mix: int=10, 
                                CrossVal: int=5, MixVal: float=0.5 ):
        """
        Class-method wich adds a set of polynomial models to the list.
        The polynomials will have degrees going from minDegree..maxDegree.
        
        parameters:
         - minDegree : Integer indicating the minimal degree of the polynomial.[DEFAULT = 2]
         - maxDegree : Integer indicating the maximal degree of the polynomial.[DEFAULT = 5]
         - interaction : Boolean indicating if only interaction features are produced. [DEFAULT = False]
         - bias : Boolean indicating if a bias column is added (i.e., where all powers 
                  are zero, acts as intercept in linear model). [DEFAULT = False]
         - N_Mix   : Number of mixing ratios between Ridge and Lasso Regression to investigate. float: 0..1.0  [DEFAULT = 10]
                 Values are generated from 0.1 to 1, with 0.1 and 1 the outer borders, unless N_Mix<2, 
                 in which case 0.5 is set as only value, or the value supplied in MixVal. No values larger 
                 than 40 are accepted. If provided, N_Mix is set to 40
         - CrossVal: How-many-fold crossvalidation is required? [DEFAULT = 5]
         - MixVal : If only one mixing value is chosen, then this will be it. [DEFAULT = 0.5]
        """
        for r in range(minDegree,maxDegree+1): #+1 since python does not include the end-point
            odc=dict()
            odc['modelType']='poly_enr'
            odc['Degree']=r
            odc['Interaction']=interaction
            odc['Bias']=bias
            odc['N_Mix']=N_Mix
            odc['CrossVal']=CrossVal
            odc['MixVal']=MixVal
            #as it is a dictionary, we don't want to get duplicate names
            s='poly_enr_D'+str(r)+'_'+str(interaction)[0:1]+str(bias)[0:1]
            self.mdc[s]=odc

    def AddLassoPolyModel(self,minDegree: int=2, maxDegree: int=5, 
                           interaction: bool=False, bias: bool=False, 
                                CrossVal: int=5 ):
        """
        Class-method wich adds a set of polynomial models to the list.
        The polynomials will have degrees going from minDegree..maxDegree.
        
        parameters:
         - minDegree : Integer indicating the minimal degree of the polynomial.[DEFAULT = 2]
         - maxDegree : Integer indicating the maximal degree of the polynomial.[DEFAULT = 5]
         - interaction : Boolean indicating if only interaction features are produced. [DEFAULT = False]
         - bias : Boolean indicating if a bias column is added (i.e., where all powers 
                  are zero, acts as intercept in linear model). [DEFAULT = False]
         - CrossVal: How-many-fold crossvalidation is required? [DEFAULT = 5]
        """
        for r in range(minDegree,maxDegree+1): #+1 since python does not include the end-point
            odc=dict()
            odc['modelType']='poly_lasso'
            odc['Degree']=r
            odc['Interaction']=interaction
            odc['Bias']=bias
            odc['CrossVal']=CrossVal
            #as it is a dictionary, we don't want to get duplicate names
            s='poly_lasso_D'+str(r)+'_'+str(interaction)[0:1]+str(bias)[0:1]
            self.mdc[s]=odc
      
    def AddLSSVM(self, kernel:str=None, degree:float=2, 
                         CrossVal: int=5 ): 
        """
        Class-method wich adds an LS-SVM model to the list.
        
        parameters:
         - kernel : one of the possible LS-SVM kernels: {'linear','poly','rbf'} [DEFAULT='rbf']
         - degree : the degree in case of a polynomial kernel [DEFAULT = 2]
         - CrossVal: How-many-fold crossvalidation is required? [DEFAULT = 5] 
        """
        if (kernel == None):
            kernel='rbf'

        odc=dict()
        odc['modelType']='LS-SVM'
        odc['Kernel']=kernel
        odc['Degree']=degree
        odc['CrossVal']=CrossVal
        s='LS-SVM_'+kernel
        if (kernel=='poly'):
            s=s+"_D"+str(degree)
        self.mdc[s]=odc
        
        
    def AddSVMLinearModel(self, max_iter: int=-1):
        """
        Class-method wich adds a linear SVM model to the list.
        
        parameters:
         - max_iter : Hard limit on iterations within solver, or -1 for no limit. [DEFAULT = -1]
        """
        odc=dict()
        odc['modelType']='SVMLinear'
        odc['max_iter']=max_iter
        s='SVMLinear'
        self.mdc[s]=odc  
        
    def AddSVMPolynomialModel(self, minDegree: int=2, maxDegree: int=5,  
                              gamma: str=None, coef0: float=0.0,
                              max_iter: int=-1):
        """
        Class-method wich adds a polynomial SVM model to the list.
        
        parameters:
         - minDegree : Integer indicating the minimal degree of the polynomial.[DEFAULT = 2]
         - maxDegree : Integer indicating the maximal degree of the polynomial.[DEFAULT = 5]
         - gamma : Kernel coefficient for ‘rbf’. 
                   if gamma='scale' is passed then it uses 1/(n_features*X.var()) 
                   as value of gamma,if ‘auto’, uses 1 / n_features.
                   [default = scale]
         - coef0 : Independent term in kernel function. It is only significant in ‘poly’.
         - max_iter : Hard limit on iterations within solver, or -1 for no limit. [DEFAULT = -1]
        """
        if (gamma == None):
            gamma='scale'
        
        for r in range(minDegree,maxDegree+1): #+1 since python does not include the end-point
            odc=dict()
            odc['modelType']='SVMPolynomial'
            odc['degree']=r
            odc['gamma']=gamma
            odc['coef0']=coef0
            odc['max_iter']=max_iter
            #as it is a dictionary, we doan't want to get duplicate names
            s='SVMPolynomial_D'+str(r)
            self.mdc[s]=odc
        
    def AddSVM_RBF(self, gamma: str=None, shrinking: bool=True, max_iter: int=-1 ):
        """
        Class-method wich adds a radial-basis function SVM model to the list.
        
        parameters:
         - gamma : Kernel coefficient for ‘rbf’. 
                   if gamma='scale' is passed then it uses 1/(n_features*X.var()) 
                   as value of gamma,if ‘auto’, uses 1 / n_features.
                   [default = scale]
         - shrinking : Whether to use the shrinking heuristic. [DEFAULT = True]
         - max_iter : Hard limit on iterations within solver, or -1 for no limit. [DEFAULT = -1]
         
         """
        if (gamma == None):
            gamma='scale'
        
        odc=dict()
        odc['modelType']='SVM_RBF'
        odc['gamma']=gamma
        odc['shrinking']=shrinking
        odc['max_iter']=max_iter
        s='SVM_RBF'
        self.mdc[s]=odc  
            
            
            
            
            
            
            
            
