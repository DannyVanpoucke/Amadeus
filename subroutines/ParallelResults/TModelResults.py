# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:53:16 2019

Contains all results of one specific model. Performs the statistics on the set of results.


@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
from TDataPoint import TDataPoint


class TModelResults:
    """
    This class contains the following information:
        - Name  : string with the model name (unique)
        - NameModel : string with the model name (general class)
        - NData : integer number indicating the number of data-points
        - Average : average value of a dataset (array)
        - StDev   : standard deviation of a dataset (array)
        - RMSEfulltrain : RMSE on the full training set (absolute values, array)
        - RMSEfulltest : RMSE on the full test set (absolute values, array)
        - MAEfulltrain : Mean-Absolute-Error on the full training set (array)
        - MAEfulltest  : Mean-Absolute-Error on the full test set (array)
        - RMSEtrainLoO_mean : the mean RMSE on the training data using Leave-One-Out
        - RMSEtrainLoO_2sig : the 95% (2sigma) range
        - RMSEtrainCV5_mean : the mean RMSE on the training data using 5-fold Cross-Validation
        - RMSEtrainCV5_2sig : the 95% (2sigma) range
        - modelCoef : Array of dictionaries, containing the (hyper-)parameters for each model instance
    """
    def __init__(self, Name: str, NameModel: str, initialNDataRuns: int=0):
        """
        Initialise all arrays as zero-arrays of length initialNDataRuns.
        
        parameters:
            - Name :  name of the model (preferably unique)
            - initialNDataRuns : length of the arrays upon initialisation. Default=0
        """
        self.Name=Name
        self.NameModel=NameModel
        self.NData=initialNDataRuns
        self.Average=[0.0]*initialNDataRuns
        self.StDev=[0.0]*initialNDataRuns
        self.RMSEfulltrain=[0.0]*initialNDataRuns
        self.RMSEfulltest=[0.0]*initialNDataRuns
        self.MAEfulltrain=[0.0]*initialNDataRuns
        self.MAEfulltest=[0.0]*initialNDataRuns
        self.RMSEtrainLoO_mean=[0.0]*initialNDataRuns
        self.RMSEtrainLoO_2sig=[0.0]*initialNDataRuns
        self.RMSEtrainCV5_mean=[0.0]*initialNDataRuns
        self.RMSEtrainCV5_2sig=[0.0]*initialNDataRuns
        self.modelCoef=[0]*initialNDataRuns  #this will become an array of dictionaries
    
    def addDataPoint(self, dp: TDataPoint): 
                     #Avg, StD, RMSEtrain,  MAEtrain, RMSEtest, MAEtest: float, 
                     #trainLoO_mean, trainLoO_2sig, 
                     #trainCV5_mean, trainCV5_2sig:float,
                     #modelCoef: dict):
        """
        Adding the results of a single data-point to arrays of values.
        
        parameters:
            - self
            - dp : TDataPoint object of which the values need to be appended to the arrays
        """
        self.NData+=1
        self.Average.append(dp.mean)
        self.StDev.append(dp.std)
        self.RMSEfulltrain.append(dp.RMSEtrain)
        self.RMSEfulltest.append(dp.RMSEtest)
        self.MAEfulltrain.append(dp.MAEtrain)
        self.MAEfulltest.append(dp.MAEtest)
        self.RMSEtrainLoO_mean.append(dp.trainLoO_mean)
        self.RMSEtrainLoO_2sig.append(dp.trainLoO_2sig)
        self.RMSEtrainCV5_mean.append(dp.trainCV5_mean)
        self.RMSEtrainCV5_2sig.append(dp.trainCV5_2sig)
        self.modelCoef.append(dp.modelCoef)
        
    def setDataPoint(self, dp: TDataPoint): 
                     #index:int, Avg, StD, RMSEtrain, MAEtrain, RMSEtest, MAEtest: float, 
                     #trainLoO_mean, trainLoO_2sig, 
                     #trainCV5_mean, trainCV5_2sig:float,
                     #modelCoef: dict):
        """
        Setting the results of a single data-point.
        
        parameters:
            - index : position of the datapoint to modify. 
            --> if the index is larger than NData, then this data is appended at the end of the list
        """
        index=dp.index
        if (index<self.NData):
            self.Average[index]=dp.mean
            self.StDev[index]=dp.std
            self.RMSEfulltrain[index]=dp.RMSEtrain
            self.RMSEfulltest[index]=dp.RMSEtest
            self.MAEfulltrain[index]=dp.MAEtrain
            self.MAEfulltest[index]=dp.MAEtest
            self.RMSEtrainLoO_mean[index]=dp.trainLoO_mean
            self.RMSEtrainLoO_2sig[index]=dp.trainLoO_2sig
            self.RMSEtrainCV5_mean[index]=dp.trainCV5_mean
            self.RMSEtrainCV5_2sig[index]=dp.trainCV5_2sig
            self.modelCoef[index]=dp.modelCoef
        else:
            self.addDataPoint(dp)
            #Avg, StD, RMSEtrain, MAEtrain, RMSEtest, MAEtest, 
            #         trainLoO_mean, trainLoO_2sig, 
            #         trainCV5_mean, trainCV5_2sig, modelCoef)
        
        
    def printQualityDataPoint(self, index: int, printCoef: bool=False):
        """
        Prints a quality-control block of a model.
        
        parameters:
            - index: index of the data-point to print
            - printCoef : should the coefficients and hyper-parameters be printed as well?
            
        """
        
        if (index<0)or(index>=self.NData): #out of range: print warning and do nothing
            print("WARNING: index %i is out of range (>%i , or <0). Cannot printQualityDataPoint::TModelResults.")
            return
        #If we get here, we just print
        print("============ Quality measures Datablock ==================")
        print("Quality measures for model: ",self.Name," which is a ",self.NameModel," .")
        print("Real target properties: ")
        print("  - Average       = %0.3f" % self.Average[index])
        print("  - Standard Dev. = %0.3f" % self.StDev[index])
        print("")
        print("1. Measure on whole set:")
        pct=(self.RMSEfulltrain[index]/self.Average[index])*100.0
        print("RMSE of the prediction by the model= %0.3f  --> %0.2f %%" % (self.RMSEfulltrain[index], pct ))
        pct=(self.MAEfulltrain[index]/self.Average[index])*100.0
        print("MAE of the prediction by the model= %0.3f  --> %0.2f %%" % (self.MAEfulltrain[index], pct ))
        print("2. Measure of cross-validation  (95% interval):")
        pct=(self.RMSEtrainLoO_mean[index]/self.Average[index])*100.0
        print("Leave-One-Out: RMSE: %0.3f (+/- %0.3f)      --> %0.2f %%" % (self.RMSEtrainLoO_mean[index], self.RMSEtrainLoO_2sig[index], pct))
        pct=(self.RMSEtrainCV5_mean[index]/self.Average[index])*100.0
        print("5-fold CV    : RMSE: %0.3f (+/- %0.3f)      --> %0.2f %%" % (self.RMSEtrainCV5_mean[index], self.RMSEtrainCV5_2sig[index], pct))
        if printCoef:
            print("")
            textBlock=[""]*len(self.modelCoef[index])
            for key in self.modelCoef[index]:
                value=self.modelCoef[index][key] #this should be an array
                textBlock[abs(value[0])]=''.join(str(element) for element in value[1:])
            for i in range(len(self.modelCoef[index])):
                print(textBlock[i])
        print("=============================================================")
        
        
    def PrintResults(self):
        """
        Prints the averaged results for 1 model
        """    
        import numpy as np
                   
        print(self.Name,"  AVG= %0.4f  STDEV= %0.4f " % (np.mean(self.Average),np.mean(self.StDev)),
              "RMSE train: %0.4f RMSE test: %0.4f " % (np.mean(self.RMSEfulltrain),np.mean(self.RMSEfulltest)),
              "MAE train: %0.4f MAE test: %0.4f " % (np.mean(self.MAEfulltrain),np.mean(self.MAEfulltest)),
              "LoO_tr= %0.4f (+- %0.4f )" % (np.mean(self.RMSEtrainLoO_mean),np.mean(self.RMSEtrainLoO_2sig)),
              "CV5_tr= %0.4f (+- %0.4f )" % (np.mean(self.RMSEtrainCV5_mean),np.mean(self.RMSEtrainCV5_2sig))
              )
        
    def PrintStatistics(self, file: str=None):
        """
        Prints the list of results for all runs to a textfile
        
        parameters:
            file: filename of the file to append with the data. (default= None, i.e. model name)
        
        columns:
        - 1 : run index
        - 2 :
        
        """    
        import numpy as np
        
        if (file==None):
            filename=self.Name + ".dat"
            f=open(filename,"w+")
        else:
            f=open(file,"a+")
            
        
        #For LS-SVM we need to sort the coeff and add zeros for missing vectors
        if (self.NameModel=='LS-SVM'):
            mfc=[0]*self.NData 
            FullSet=set()
            for i in range(self.NData):
                FullSet.update(set(self.modelCoef[i]['data_pt_index'][1].flatten())) #flatten because nd-array
            SizeSet=max(FullSet)+1 #if for some reason some support vectors are missing altogether
            for i in range(self.NData):
                coef=np.zeros(2*SizeSet+1)#+1 ofset in modelcoef
                coefar=np.asarray(self.modelCoef[i]['coef_'][1]).flatten()
                idxar=self.modelCoef[i]['data_pt_index'][1].flatten()
                for j in range(self.modelCoef[i]['coef_'][1].shape[1]):
                    coef[idxar[j]+1]=coefar[j]
                    coef[SizeSet+idxar[j]+1]=idxar[j]
                locdict=dict()
                locdict['coef_']=coef
                mfc[i]=locdict
        else:#all non-LS-SVM
            mfc=self.modelCoef
        
        
        for i in range(self.NData):
            line=str(i+1)+"  "+str(self.RMSEfulltrain[i]
                 )+"  "+str(self.RMSEfulltest[i]
                 )+"  "+str(self.MAEfulltrain[i]
                 )+"  "+str(self.MAEfulltest[i]
                 )+"  "+str(self.RMSEtrainLoO_mean[i]
                 )+"  "+str(self.RMSEtrainLoO_2sig[i]
                 )+"  "+str(self.RMSEtrainCV5_mean[i]
                 )+"  "+str(self.RMSEtrainCV5_2sig[i])
            #the replace line-breaks are because of numpy's need to help out by inserting linebreaks...
            #this resolves the issue here, without breaking the numpy behaviour globally
            a=str(self.modelCoef[i]["intercept_"][1]).replace('[','').replace(']','').replace('\n','').replace('\r','')
            b=" ".join(str(n) for n in mfc[i]["coef_"][1:]).replace('[','').replace(']','').replace('\n','').replace('\r','')            
            line=line+"  "+a+"  "+b
            print(line)
            f.write(line+" \n")
        f.close()
        
        
    def PrintStatisticsHyperparameters(self, file: str=None):
        """
        Prints the list of hyperparameter-values for all runs to a textfile
        
        If no hyper-parameters are present, nothing is written.
        
        parameters:
            file: filename of the file to append with the data. (default= None, i.e. model name)
        
        columns:
        - 1 : run index
        - 2 :
        
        """    
        if (file==None):
            filename="HyperParam_"+self.Name + ".dat"
            f=open(filename,"w+")
        else:
            f=open(file,"a+")
        
        
        for i in range(self.NData):
            line=str(i+1)+"  "
            cnt=0
            for key, value in self.modelCoef[i].items():
                if (int(value[0])<0):
                    cnt+=1
                    line=line+value[1]+" "
            if (cnt>0): #there are hyperparameters to print
                print(line)
                f.write(line+" \n")
                
        f.close()
    
        
        
        
        