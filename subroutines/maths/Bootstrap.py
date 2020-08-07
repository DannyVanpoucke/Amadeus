# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:04:36 2020

Module containing functionality to perform bootstrapping of a 1D data-set.


@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
import numpy as np
from scipy.special import erf, erfinv

divSqrt2=1.0/np.sqrt(2.0,dtype=np.float64)
Sqrt2=np.sqrt(2.0,dtype=np.float64)

def _sncdf(x: np.float64)->np.float64:
    """
        Calculate the standard normal cumulative distribution function for x.
        CDF=0.5*[1+erf({x-mu}/{sig*sqrt(2)})], for the standard case mu=0, and sig=1.0

        Parameter:
            - x: float
    """
    return 0.5*(1.0+erf(x*divSqrt2))

def _sndq(x: np.float64)->np.float64:
    """
        Calculate x'th quantile of the standard normal distribution function.
        Quant=mu+sig*sqrt(2)*erf^(-1)[2x-1], for the standard case mu=0, and sig=1.0

        NOTE:
            This function is the inverse of _sncdf :-)
        
        Parameter:
            - x: float in range 0..1
    """
    return Sqrt2*erfinv(2*x-1.0)

#make it a vector-function
sncdf=np.vectorize(_sncdf, [np.float64])
sndq=np.vectorize(_sndq, [np.float64])

def Bootstrap_1Col(col:int, coeflst:list, alpha:float)->tuple:
    """
    Single line parallellizable bootstrap for a single column of coefficients/data.
    
    Note that imbedding such a function into a class results in the entire class 
    being pickled for multiprocessing, which is in no way efficient as the other data 
    of the class is not touched by this function.
    
    parameters:
        - col: the index of the column (for administrative purposes upon return)
        - coeflst: the list of coefficients
        - alpha: the BCa alpha value for the CI, default=0.05
    return:
        tuple(col-index, CIlow, CIhigh)
    """
    
    boot=TBootstrap(data=coeflst,Func=np.mean)
    boot.NPbootstrap(n_iter=2000, Jackknife=True)
    avgm, avgp = boot.ConfidenceInterval(CItype="BCa",alpha=alpha)#95%confidence interval
    
    return tuple([col,avgm,avgp])

class TBootstrap(object):
    """
    Class encapsulating some bootstrap functionalities.
    
    properties:
        - data: the dataset provide by the user, to apply bootstrapping on. Should be a 1D (numpy) array
        - datasize: integer presenting the size of the data-set
        - n_bootstrap: number of bootstrap iterations/samples
        - statistics: list containing the statistics of n_bootstrap samples
        - mean: the mean of the statistics distribution.
        - _se_b: the standard error following the bootstrap way of the statistics distribution
        - _se_JafterB : the standard-error on the bootstrap-standard-errorthe using jackknife-after-bootstrap 
        - _JaB_theta_i: A list of Theta_i (the jackknife value for the statistics on the x_i sample) constructed during the Jackknife-after-Bootstrap
        - _JaB_theta_mean: the mean value of the above
    """
    def __init__(self, data: np.array, Func: None):
        """
        Simple initialisation of the Class, by loading the data of the user.
       
        parameters:
         - self: the class
         - data: the dataset provide by the user, to apply bootstrapping on. Should be a 1D (numpy) array
         - RNG : the seed for the 
         - Func: Function used to calculate the statistics of interest. It should have a shape: Func(data-array)->float
                 If no function is provided, the numpy.average function is assumed.
       
        """ 
        self.data=np.array(data) #make sure it is an np-array
        self.datasize=self.data.shape[0]
        
        #initialise the other properties on their zero's
        self.n_bootstraps=0 
        self.statistics=list()
        self.mean=None
        self._se_b=None
        self._se_JafterB=None
        self._JaB_theta_i=list()
        self._JaB_theta_mean=None
        self._JaBset=False
       
        if Func is None:
           self.StatisticsFunction=np.average
        else:
           self.StatisticsFunction=Func
           
    @property
    def se_b(self):
        """Getter for _se_b."""
        if (self._se_b == None):
            self.NPbootstrap(Jackknife=False)
        return self._se_b
    
    @property
    def se_JafterB(self):
        """Getter for _se_JafterB."""
        if (self._se_JafterB == None):
            self.NPbootstrap(Jackknife=True)
        return self._se_JafterB
       
    def NPbootstrap(self, n_iter: int=1000, Jackknife: bool=False):
        """
        Performs a nonparametric bootstrap running n_iter bootstrap samples. 
        Jackknife-after-bootstrap estimate of the accuracy is available.
        
       
        parameters:
            n_iter: integer number of bootstrap samples to be drawn. DEFAULT=1000
            Jackknife: boolean indicating if a Jackknife-after-bottstrap accuracy estimate is needed[OPTIONAL, DEFAULT= False]
        """
        #from sklearn.utils import resample
        #clear the statistics list, if that is not empty
        self.statistics.clear()
        if (n_iter<2):#make sure no illegal values are provided, if so, switch to default.
            n_iter=1000
        self.n_bootstraps=n_iter
        
        #np.random.seed()   #initialise the random number generator
        #seeds=np.random.randint(low=0,high=2**31,size=n_iter) #create a "different" seed for each sample
        #If we want to use jackknife-after-bootstrap we need to keep track of all sample-sets. 
        #So resampling should be done on integer indices which remain stored
        #print("n_iter=",n_iter)
        s_idx=self.GenBootstrap_idx(self.datasize,n_samples=n_iter)
        
        for ns in range(n_iter):
            sample=np.array([ self.data[idx] for idx in s_idx[ns] ])
            #sample=resample(self.data,replace=True,random_state=seeds[nr],stratify=None)# will not keep track of the indices
            stat=self.StatisticsFunction(sample)
            self.statistics.append(stat)
        #calculate the mean of the bootstrapped samples    
        self.mean=np.mean(self.statistics,axis=0)
        #The bootstrap standard error is estimated as the "empirical statndard deviation", p159
        self._se_b=np.std(self.statistics,axis=0,ddof=1)#ddof=1 to get unbiased estimator of the variance: divsion by N-1 instead of N          
        
        if (Jackknife):
            self.JackAfterBoot(s_idx)
        
    
    def JackAfterBoot(self,sample_idx: np.array):
        """
        Perform a Jackkife-after-bootstrap run using the integer index-lists
        generated for the bootstrap run.
        (cf chapt 19.4 of Efron and Tibshirani 1993)
        
        parameters:
            - sample_idx : 2D numpy-array, every row contains the selected indexes of 1 sample
        """
        import copy
        #import time
        
        #start = time.perf_counter_ns()
        
        si_avg=np.zeros(sample_idx.shape[1]) #to track the average of each set
        cnt_Bi=np.zeros(sample_idx.shape[1]) #count the number of sets 
        se_Bi=np.zeros(sample_idx.shape[1]) 
        #run over all samples, and check whether point i is missing
        #and calculate the averages and counters
        #to speed things up, only transform to sets once:
        sample_ids=list()
        for nr in range(sample_idx.shape[0]): #row-indices
            sample_ids.append(set(sample_idx[nr]))
        
        
        #end = time.perf_counter_ns()
        #print("---- Pre   :",(end-start)/1E6," ms")
        for nr in range(sample_idx.shape[0]): #row-indices
            #sample_ids=set(sample_idx[nr])
            for dpi in range(sample_idx.shape[1]): #the index of the missing datapoint
                if (dpi not in sample_ids[nr]):
                    cnt_Bi[dpi]+=1
                    si_avg[dpi]+=self.statistics[nr]
        
        #end = time.perf_counter_ns()
        #print("---- Loop_1:",(end-start)/1E6," ms  --> ",sample_idx.shape)
        for dpi in range(sample_idx.shape[1]): #now divide to get an average
            if (int(cnt_Bi[dpi])>0):#if we have no hits, si_avg should be zero anyhow
                si_avg[dpi]=si_avg[dpi]/cnt_Bi[dpi]
        #end = time.perf_counter_ns()
        #print("---- Loop_2:",(end-start)/1E6," ms  --> ")
        
        #keep track of these if we want to have confidence intervals lateron
        self._JaB_theta_i=copy.deepcopy(si_avg)
        self._JaB_theta_mean=np.mean(self._JaB_theta_i)
        
        #end = time.perf_counter_ns()
        #print("---- Inter :",(end-start)/1E6," ms ")
                
        #next calculate the SE_B(i), eq 19.8 p277
        for nr in range(sample_idx.shape[0]): #row-indices
            #sample_ids=set(sample_idx[nr])
            for dpi in range(sample_idx.shape[1]): #the index of the missing datapoint
                if (dpi not in sample_ids[nr]):
                    se_Bi[dpi]+=(self.statistics[nr]-si_avg[dpi])**2
        
        #end = time.perf_counter_ns()
        #print("---- Loop_3:",(end-start)/1E6," ms  --> ",sample_idx.shape)
        
        for dpi in range(sample_idx.shape[1]): #finish up the se calculation
            if (int(cnt_Bi[dpi])>0):#if we have no hits, si_avg should be zero anyhow
                se_Bi[dpi]=np.sqrt(se_Bi[dpi]/cnt_Bi[dpi])
            
        #end = time.perf_counter_ns()
        #print("---- Loop_4:",(end-start)/1E6," ms  --> ")
        
        #finally the Jackknife
        avg_se_Bi=0.0
        for dpi in range(sample_idx.shape[1]):
            avg_se_Bi+=se_Bi[dpi]
            
        avg_se_Bi=avg_se_Bi/sample_idx.shape[1] 
        var_jack=0
        for dpi in range(sample_idx.shape[1]):
            var_jack+=(se_Bi[dpi]-avg_se_Bi)**2
        var_jack=var_jack*((sample_idx.shape[1]-1)/sample_idx.shape[1])
        self._se_JafterB=np.sqrt(var_jack)
        self._JaBset=True
        #end = time.perf_counter_ns()
        #print("---- END      :",(end-start)/1E6," ms ")
        
        
    def GenBootstrap_idx(self, datasize: int, n_samples: int=1000)->np.array:
        """
          Returns a 2D numpy array of bootstrap ready indices. The indices for each sample are stored 
          as the rows.
          
          NOTE:
              The storage for this 2D array may be rather large, however, it allows one to use this 
              index list more than once, which would be the case when using a generator.
              (Think: Jackkife-after-bootstrap)
          
          Parameters:
              - datasize: the size each sample should be (we don't need the actual data, only the size of the dataset)
              - n_samples: the number of samples to create [OPTIONAL, DEFAULT = 1000]
        """
        idx=list()
        sizetype=np.int16
        if (datasize>32000):
            sizetype=np.int32
            
        for nr in range(n_samples):
            idx.append(np.random.randint(low=0,high=datasize,size=datasize, dtype=sizetype))
            #yield np.random.randint(data.shape[0], size=(data.shape[0],)) 
        return np.array(idx)
    
    def ConfidenceInterval(self, CItype: str=None, alpha: float=0.05,n_samples: int=1000) -> tuple:
        """
            Returns the confidence interval as a tuple: (low,high), with low and high the absolute
            positions of the edges of the confidence interval of the estimated statistic.
            
            Parameters:
                - CItype : Which type of confidence interval to use: [DEFAULT=stdint]
                    - stdint: use the standard interval of 1.96se_boot, only a 95% interval is possible here
                    - pc    : use the percentile method
                    - BCa   : use the bias corrected and accelarated confidence interval
                - alpha : the percentile to use for the confidence interval. [DEFAULT=0.05, i.e. the 95% interval]
                - n_samples : number of jackknife-bootstrap samples to be used in the BCa method. If none set 
                            the default(=1000) is used. Note that if a Jackknife-after-bootstrap was performed
                            in NPbootstrap (i.e. Jack=True), then this parameter is ignored, and the earlier
                            generated statistics are used to calculate the terms of BCa.
        """
        from ListTools import checkEqualNDarray, checkEqual
        import warnings
        warnings.filterwarnings('error')# otherwise dumb python can not catch a division by zero "RuntimeWarning
        
        
        if (CItype == None):
            CItype="stdint"
        
        if (CItype=="pc"):
            alow=alpha*0.5
            ahigh=1.0-alow
            #the numpy quantile function does the trick for us. (percentile just calls quantile=extra overhead)
            CIlow=np.quantile(self.statistics,alow,interpolation='linear') #no need to sort the array :-)
            CIhigh=np.quantile(self.statistics,ahigh,interpolation='linear')
        elif (CItype=="BCa"):
            #check if Jackknife afer bootstrap was performed
            if not self._JaBset:
                self.n_bootstraps=2000
                self.NPbootstrap(n_iter=self.n_bootstraps, Jackknife=True)
                
            alow=alpha*0.5
            ahigh=1.0-alow
            za_low=sndq(alow)
            za_high=sndq(ahigh)
            orginal_estimate=self.StatisticsFunction(self.data)
            z0=sndq(1.0*np.sum(self.statistics < orginal_estimate, axis=0)/self.n_bootstraps)#eq 14.14 p186...sndq is inverse of sncdf :-)
            anum=0.0
            aden=0.0
            for nr in range(len(self._JaB_theta_i)):
                tmp=self._JaB_theta_mean-self._JaB_theta_i[nr] # version of the original book...which most seem to be using--> derivation should be checked to be sure
                #tmp=self._JaB_theta_i[nr]-self._JaB_theta_mean # version of the ComputerAgeStatInference book
                anum+=tmp**3
                aden+=tmp**2
            
            try: #apparently aden can be zero??or something which throws a RuntimeWarning, but no usefull info (thx Python)
                ahat=anum/(6.0*(aden**1.5))
            except:
                #We have a list of identical values (or zero's)
                print("There were issues with anum=",anum," and aden=",aden,
                      " JaB_theta_i size= ",len(self._JaB_theta_i)," mean= ",self._JaB_theta_mean,
                      " All equal=",checkEqualNDarray(self._JaB_theta_i))
                #print("    mean=",self._JaB_theta_mean)
                #print("    vals=",self._JaB_theta_i[:],"\n")
                
                ahat=1.0
                
            #print("A_HAT=",ahat)
            #The associated percentiles
            if (not np.isinf(z0)):# if all values are the same, z0 will be -infty, as a result pc goes to the same value...
                pclow=sncdf( z0 + (z0+za_low)/(1-ahat*(z0+za_low)) )
                pchigh=sncdf( z0 + (z0+za_high)/(1-ahat*(z0+za_high)) )
            else:#so we set the pc bound to the bounds of the set (quick and dirty, needs to be checked wrt math logic
                pclow=0.0 #need to be in range [0..1]
                pchigh=1.0
            
#            print("STATISTICS=",self.statistics)
#            print("pclow=",pclow)
#            print("Z0=",z0,"    ahat=",ahat)
#            print("za_low=",za_low,"    za_high",za_high)
#            print("mean=",self.mean,"   stdev=",self._se_b)
            
            #the actual interval            
            CIlow=np.quantile(self.statistics,pclow,interpolation='linear') #no need to sort the array :-)
            CIhigh=np.quantile(self.statistics,pchigh,interpolation='linear')
            
        else:#move to default--> stdint
            CIlow=self.mean-1.96*self.se_b
            CIhigh=self.mean+1.96*self.se_b
            
        return CIlow, CIhigh
            
        
           
       
       
       
       
       
       