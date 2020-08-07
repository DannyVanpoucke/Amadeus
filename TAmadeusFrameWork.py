# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:14:35 2020

The heart of the Amadeus Framework

@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut 
import multiprocessing as mp

#The source is in various files in the folder subroutines,
# so need to add this to the path first
import sys
sys.path.append("subroutines/models")
sys.path.append("subroutines/ParallelResults")
sys.path.append("subroutines/maths")
sys.path.append("subroutines/datahandling")
from ModelList import TModelList 
from ModelClassFactory import ModelFactory
from TAllResultsClass import TAllResultsClass
from TModelResults import TModelResults
from TDataPoint import TDataPoint
from ModelPipelineFactory import PipelineFactory

class TAmadeusFrameWork: 
    """
    This is the class containing the black box framework which will be running all ML actions.
    
    It has the fllowing properties:
        - name: string with the name of this class
        - longname: string with the full acronym written out
        - num_workers: integer number, giving the number of parallel threads to use
        - dataset: The original dataset to model--> is pipelined into ModelFrame
        - predictset: the original dataset for test/prediction --> is pipelined into PredictFrame
        - preScale: Boolean indicating if a pipeline scaler is performed on "full"(true) or "train-only"(false) data 
        - ModelFrame: a pandas data-frame containing the data to model (train+validation to split). 
                    Because this frame is influenced by the pipeline, we make it a dictionary of 
                    modelframes, one for each pipeline version of the model.
        - PredictFrame: an optional pandas data-frame containing the test/prediction data for the model
                    Because this frame is influenced by the pipeline, we make it a dictionary of 
                    modelframes, one for each pipeline version of the model.
        - Pipeline : transformation pipeline to perform on the full dataset before 
                     train-validation splitting. This is an array, with a pipeline for 
                     each model considered
        - AverageModel : dictionary of the averaged sk-learn-models
        - ModelList : TModelList object of models being considered by the framework
            
    """
    def __init__(self, dataset:pd.DataFrame, predictset: pd.DataFrame=None, njobs:int=-1, test_size: float= 0.2, maxRuns: int=1000, 
                 printFileStatistics: str=None, PreStdScaler: bool=True):
        """
        Initiallisation of the Amadeus Framework.
        
        parameters:
            - dataset: a pandas data-frame containing the data to model
            - predictset: a pandas data-frame containing the data to model, [OPTIONAL, DEFAULT=None]
            - njobs: positive integer number indicating the number of processes to use 
                     for parallelisation. If set to a negative value the number of processes is set
                     to the number of physical cores times the absolute value given in njobs. 
                     (default = -1) \Todo: fix this for multi-cpu nodes.
            - test_size : fraction of the data to use as test-data for train_test_split. (default = 0.2  (aka 20%) )
            - maxRuns : the maximum number of runs allowed for averaging a model. (default=1000)
            - printFileStatistics : filename to print statistics data. (default= None, i.e. standard out)
            - PreStdScaler : Boolean indicating if the full-data needs to be pulled through a standard scaler before
                             train-test splitting. Default=True, as this is the only way to get a usefull pipeline for prediction
        """
        from HPCTools import get_num_procs
        
        self.name="Amadeus"
        self.longname="Artificial intelligence and Machine learning frAmework for DEsigning Useful materialS"
        self.num_workers=get_num_procs(njobs)
        self.test_split=test_size
        self.maxRuns=maxRuns
        self.printFileStatistics=printFileStatistics
        #some objects which need to be created later as they are model dependent
        self.dataset=dataset
        self.predictset=predictset
        self.preScale=PreStdScaler
        self.Pipeline=dict() #one pipeline per model
        self.ModelFrame=dict()    #this frame depends on the model, so it should be a list/dict
        self.PredictFrame=dict()  #this frame depends on the model, so it should be a list/dict
        self.AverageModel=dict()
        self.ModelList=None
#######################################################################
#######################################################################
    def _SetupPipeline(self, modelname:str, kwargs: dict):
        """
        Set up the pipeline depending on the model, and transform the data and
        prediction data into pipelined frames.
        
        parameters:
        - modelname : string of the specific model-instance(name) being run
        - kwargs    : deepcopy of the tuple of model-parameters
        """
        
        modelType=kwargs['modelType']    #a deepcopy is needed, otherwise modeltype is deleted in our ModelList...
        del kwargs['modelType']          #remove this key-value for further processing
        self.Pipeline[modelname]=PipelineFactory(modelType,not self.preScale,**kwargs) #if PreScale is true, then it should NOT be a dummy
                
        #Split the dataset panda into features and targets pandas, as we can not do a transform on only the feature columns
        #so we need to split feature from target, transform, and then join again
        dataset=copy.deepcopy(self.dataset) #better be safe than sorry with python
        predictset=copy.deepcopy(self.predictset)
        
        TargFrame=dataset[dataset.columns[-1]].copy(deep=True)   #split
        FeatFrame=dataset.drop(dataset.columns[-1],axis=1)
        
        FeatFrameScaled=pd.DataFrame(self.Pipeline[modelname].fit_transform(FeatFrame),index=FeatFrame.index)
        #FeatFrameScaled=pd.DataFrame(self.Pipeline[modelname].fit_transform(FeatFrame),
        #                             index=FeatFrame.index, columns = FeatFrame.columns)   #transform...and keep the old indexing
        
        print("Amadeus Pipeline STDSCALER  SIGMA=",self.Pipeline[modelname]['std_scaler'].scale_)
        print("Amadeus Pipeline STDSCALER  MEAN=",self.Pipeline[modelname]['std_scaler'].mean_)
        
        self.ModelFrame[modelname]=pd.concat([FeatFrameScaled,TargFrame],axis=1,join='inner',copy=True)
        if predictset is not None:
            #and the same for the validation set
            TargFrame=predictset[predictset.columns[-1]].copy(deep=True)   #split
            FeatFrame=predictset.drop(predictset.columns[-1],axis=1)
            FeatFrameScaled=pd.DataFrame(self.Pipeline[modelname].transform(FeatFrame),index=FeatFrame.index)   #only transform, no fit...and keep the old indexing
            #FeatFrameScaled=pd.DataFrame(self.Pipeline[modelname].transform(FeatFrame),index=FeatFrame.index, columns = FeatFrame.columns)   #only transform, no fit...and keep the old indexing
            self.PredictFrame[modelname]=pd.concat([FeatFrameScaled,TargFrame],axis=1,join='inner',copy=True)
        else:
            self.PredictFrame[modelname]=predictset

#######################################################################
#######################################################################        
#    def _doOneRun(self,ModelFrame: pd.DataFrame, ModelList: TModelList, mod: dict,  runIndex: int, runSeed: int, failcnt: list):
    def _doOneRun(self,ModelFrame: pd.DataFrame, kwargs: dict, mod: str,  runIndex: int, runSeed: int, failcnt: list) -> TDataPoint:
        """
        This function performs a single run of a set of parallel runs, checking/testing
        all our different models. As this is run in parallel, NO DATA CAN BE SAVED TO BE 
        USED OUTSIDE THE SCOPE OF THIS FUNCTION. The data we need to store, is put into a 
        special object which is returned at the end of this function.
        
        parameters:
            - ModelFrame: a pandas data-frame containing the data to model
            - kwargs    : deepcopy of the tuple of model-parameters
            - mod       : string of the specific model(name) being run..part of moving the model-loop to the outside
            - runIndex  : integer with the number of the specific parallel run
            - runSeed   : integer seed for setting up the RandomState in the train_test_split. 
                            It should be different for each run.
        
        return:
            - Array of TDataPoints
        """
        print("========= THIS IS RUN ",runIndex," ===============")
        #do an 80/20 split...only 25 data, so 5 for testing...will give rather large errors 
        # + random_state: it uses np.random, which probably uses a Mersenne Twister...we are not doing MC, so should be OK
        # + set Shuffle explicitly to true, as our data is sorted by humans
        # + although stratified sampling is a good idea...our dataset is too small, 
        #       I think stratifying will kill the possibility to make a random split--> so don't use
        
        try:
            train_f, test_f = train_test_split(ModelFrame,test_size=self.test_split,random_state=np.random.RandomState(seed=runSeed),shuffle=True)
        except ValueError: #maybe test_split=0, then we take both test and train equal to the full set
            if (abs(self.test_split)<0.01):
                train_f=ModelFrame
                test_f=ModelFrame
            else:
                print("ERROR: something is wrong with the train-test splitting. Terminating job.")
                sys.exit()
            
        #split the targets from the features:
        target=train_f[train_f.columns[-1]].copy(deep=True)
        feature=train_f.drop(train_f.columns[-1],axis=1)
        target_test=test_f[test_f.columns[-1]].copy(deep=True)
        feature_test=test_f.drop(test_f.columns[-1],axis=1)
        
        #dpArray=[]
        
        #loop over models
        #for mod in ModelList.GetModelDictionary():
        #kwargs=copy.deepcopy(ModelList.mdc[mod])#it's a dict, so need to get it like this, not as tuple in the for 
        modelType=kwargs['modelType']           #a deepcopy is needed, otherwise modeltype is deleted in our ModelList...
        del kwargs['modelType']                 #remove this key-value for further processing
            
        #setup the model and pipeline
        modelclass = ModelFactory(modelType,mod,target, feature, 
                                  target_test, feature_test, 
                                  self.preScale, self.Pipeline[mod],**kwargs)
        #perform a fit   
        modelclass.fit()
        #failcnt[runIndex]+=runIndex  # for testing purpose
        failcnt[runIndex]+=modelclass.fitSanityCheck()
        #print the fitting coefficients
        #modelclass.printCoefficients()
        #Calculate the Quality...printing outside parallel to prevent clutter
        datap=modelclass.getQualityMeasures()
        datap.setIndex(runIndex)
        #dpArray.append(datap)
        #and now delete the model-class, free up the memory
        del modelclass
        
        return datap #dpArray
#######################################################################
#######################################################################
    def RunModelEnsemble(self)->list:
        """
        When used commandline, the first extra variable is the number of cores to use...overrides the cpu_count of psutil
        
        returns:
            - an integer list of fails per model
        
        """
        print("=============Start of Framework==============")
        print("   I. Create a model list")
        manager = mp.Manager() #manager for sharing some data between parallel processes
        LoO_CV=LeaveOneOut()
        
        modellist=TModelList()
        #modellist.AddLinearModel()
        #modellist.AddPolynomialModel(minDegree=2, maxDegree=6, interaction=False, bias=False)
        #modellist.AddRegularizedPolyModel(minDegree=2, maxDegree=2, interaction=False, 
        #                                  bias=False, N_Mix=30, CrossVal=LoO_CV)#CrossVal=features.shape[0])
        #modellist.AddLassoPolyModel(minDegree=1, maxDegree=1, interaction=False, 
        #                                  bias=False, CrossVal=LoO_CV)#CrossVal=features.shape[0])
        modellist.AddLSSVM(kernel='rbf',CrossVal=LoO_CV)
        #modellist.AddSVMLinearModel(max_iter=10000)
        #modellist.AddSVMPolynomialModel(minDegree=3, maxDegree=3)
        #modellist.AddSVM_RBF()
        
        
        
        print("   II. Setup results-class")
        NRuns=self.maxRuns
        np.random.seed()   #initialise the random number generator
        seeds=np.random.randint(low=0,high=2**31,size=NRuns) #create a "different" seed for each run, to be used with the train_test_split  
        
        Results=TAllResultsClass(NDataRuns=NRuns)
        for mod in modellist.GetModelDictionary():
            Results.addModel(mod,modellist.mdc[mod]['modelType'])
            
        #for mod in ModelList.GetModelDictionary():
        print("   III. Start the parallel run over the datasplits")
        #Let's do some things in parallel    
        print("     * Checking ",Results.getNModels()," different models.")
        print("     * Will run all the models ",NRuns," times.")
        print("     * paralellisation over ",self.num_workers," processes.")
        #create our process pool
        pool=mp.Pool(processes=self.num_workers)
        modelFailCnt=[]
        
        for mod in modellist.GetModelDictionary():
            #add the corect pre-pipeline to the list
            #and perform the needed transform to get a pipelined data and perdict frame
            self._SetupPipeline(mod,copy.deepcopy(modellist.mdc[mod]))
            
            #keep track of fails using a multiprocess manager
            failcnt=manager.list([0]*NRuns)
            #let the drones to the work, one package at a time
            #drones=[pool.apply_async(self._doOneRun, args=(self.ModelFrame,modellist,mod,nr,seeds[nr],failcnt)) for nr in range(NRuns)]
            drones=[pool.apply_async(self._doOneRun, args=(self.ModelFrame[mod],copy.deepcopy(modellist.mdc[mod]),mod,nr,seeds[nr],failcnt)) for nr in range(NRuns)]
        
            for drone in drones:
                Results.setDataPoint(drone.get(), Print=True, PrintCoef=True)
            
            #store and reset the failcnt list
            modelFailCnt.append(sum(failcnt))
            failcnt[:]=[]
            
        print("   III. End the parallel run over the datasplits")
        pool.close() #end of parallel work
        pool.join() #and wait untill all tasks are done before continuing with the main thread
        
        setCI4Avg=(NRuns>1)
        for mod in modellist.GetModelDictionary():
            #Create an average model...
            self._CreateAverageModel(mod,copy.deepcopy(modellist.mdc[mod]),Results.Results[mod],setCI4Avg)
        
    
        print("")
        print("")
        print("   IV. Averaged Results")
        for mod in modellist.GetModelDictionary():
            Results.Results[mod].PrintResults()
            Results.Results[mod].PrintStatistics(file=self.printFileStatistics)
            hyperFile=self.printFileStatistics[:-4]+"_hyper.dat"
            Results.Results[mod].PrintStatisticsHyperparameters(file=hyperFile)
        
        self.ModelList=modellist
        print("-----------------------------")
        return modelFailCnt
    
#######################################################################
#######################################################################
    def _CreateAverageModel(self, model: str, kwargs: dict, results: TModelResults, setCI: bool=False):
        """
        This (private) function constructs an "average" version of a model, 
        and stores it in the dictionary: AverageModel.
        
        !!NOTE: We assume that the pipeline of teh model instances is just a dummy, so the 
        !!      only pipeline to use is the one at the ensemble level, stored in the Amadeus class
        
        parameters:
            - self  : AmadeusFramework
            - model : model-name to be used as key to recover the specific model
            - kwargs: deepcopy of the tuple of model-parameters
            - results: A TAllResultsClass object containing all relevant information 
            - setCI : calculate a 95% confidence interval
        """
        
        print("====== CREATING AVG MODEL ======")
        #Create a new sk-learn model
        
        # create some dummy target and features for the "creation" of the model:
        print("1) Setup target & features")
        target=self.ModelFrame[model][self.ModelFrame[model].columns[-1]].copy(deep=True)
        feature=self.ModelFrame[model].drop(self.ModelFrame[model].columns[-1],axis=1)
        target_test=target
        feature_test=feature
        
        modelType=kwargs['modelType']           #a deepcopy is needed, otherwise modeltype is deleted in our ModelList...
        del kwargs['modelType']                 #remove this key-value for further processing
        print("2) Model Factory")
        Avgmodel = ModelFactory(modelType,model,target, feature, 
                                target_test, feature_test, self.preScale,
                                self.Pipeline[model],**kwargs) #setup the model and pipeline
        #Avgmodel.fit() #perform a fit, not needed to modify the coefficients...at least for a linear regression   
        print("3) Average Coefficients ",setCI)
        Avgmodel.setAverageCoefficients(results, setCI=setCI)
        print("4) Print")
        Avgmodel.printAverageCoefficients()
        Avgmodel.printAverageCoefficients(File="AverageModels.dat")
        self.AverageModel[model]=Avgmodel

        print("====== END AVG MODEL CREATION ======")    
    
#######################################################################
#######################################################################    
    def PredictAverageModel(self, data: pd.DataFrame=None, hasTarget: bool=True, 
                            error: str=None, OutFile: str=None, WriteStatus: str=None ):
        """
        Takes a data-set (pandas dataframe) or uses the initially provided validation data, 
        and predicts the result using the average models.
        
        !!NOTE: We assume that the pipeline of teh model instances is just a dummy, so the 
        !!      only pipeline to use is the one at the ensemble level, stored in the Amadeus class
        
        parameters:
            - self
            - data: the dataframe containing the relevant data to predict. If None, then
                    the validation data will be used.
            - hasTarget: assume that Target data is available as the last single column.
            - error: Type of error to calculate on a set of data. None= no, RMSE or MAE, Default= None
            - OutFile: Name of the csv-file to which the results are written. Default="NewFile.csv"
            - WriteStatus: Status for writing the new file: 'a, a+, w or w+'. Default='w+'
            
        """
        from sklearn.metrics import mean_squared_error,mean_absolute_error
        import csv
        
        if error is None:
            error="No"
        if (error!="RMSE")and(error!="MAE"):
            error="No"
        if not hasTarget:
            error="No"
            
        if OutFile is None:
            OutFile="NewFile.csv"
        if WriteStatus is None:
            WriteStatus="w+" # overwrite, and create if need be
        
        calcError=True
#        #prepare target/feature data
#        if data is None: # we use the Validation data already loaded
#            if self.PredictFrame is not None: # split in target and feature:
#                target=self.PredictFrame[self.PredictFrame.columns[-1]].copy(deep=True)   #split
#                featuretf=self.PredictFrame.drop(self.PredictFrame.columns[-1],axis=1)  # transform was done during init
#            else:
#                raise Exception("PredictAverageModel requires actual data, as no ValidationFrame was loaded upon initialisation.")
#        else:# we have data:
#            if hasTarget : #split
#                target=data[data.columns[-1]].copy(deep=True)   #split
#                feature=data.drop(data.columns[-1],axis=1)    # transform was done during init
#                featuretf=pd.DataFrame(self.Pipeline.transform(feature),index=feature.index, columns = feature.columns)   #only transform, no fit...and keep the old indexing
#            else:
#                calcError=False #if no target data is available, no error can be calculated
#                featuretf=pd.DataFrame(self.Pipeline.transform(data),index=data.index, columns = data.columns)   #only transform, no fit...and keep the old indexing
#                target=pd.DataFrame(np.array([["/"] for i in range(len(data.index))]),columns=['No Targets'],index=data.index)
#                
        MissingPredictFrame=False
        for key,val in self.PredictFrame.items():
            if self.PredictFrame[key] is None:
                MissingPredictFrame=True
                break
        
        if (data is None)and(MissingPredictFrame):
            print("PredictAverageModel requires actual data, as no ValidationFrame was loaded upon initialisation.")
            sys.exit(1)
        
        #run over models and predict
        OFile=open(OutFile,WriteStatus,newline='')
        writer=csv.writer(OFile,delimiter=',')
        for mod in self.ModelList.GetModelDictionary():
            if data is None:#gettoing here means we have a predict frame for all pipelines/models
                target=self.PredictFrame[mod][self.PredictFrame[mod].columns[-1]].copy(deep=True)   #split
                featuretf=self.PredictFrame[mod].drop(self.PredictFrame[mod].columns[-1],axis=1)  # transform was done during init
                descriptors=self.predictset.drop(self.predictset.columns[-1],axis=1) #the not-pipelined predictor data
            else: # we use the data...and transform using the appropriate pipeline
                if hasTarget : #split
                    target=data[data.columns[-1]].copy(deep=True)   #split
                    feature=data.drop(data.columns[-1],axis=1)    # transform was done during init
                    descriptors=feature.copy(deep=True)
                    featuretf=pd.DataFrame(self.Pipeline[mod].transform(feature),index=feature.index, columns = feature.columns)   #only transform, no fit...and keep the old indexing
                else:
                    calcError=False #if no target data is available, no error can be calculated
                    descriptors=data.copy(deep=True)
                    featuretf=pd.DataFrame(self.Pipeline[mod].transform(data),index=data.index, columns = data.columns)   #only transform, no fit...and keep the old indexing
                    target=pd.DataFrame(np.array([["/"] for i in range(len(data.index))]),columns=['No Targets'],index=data.index)
            
            pdata=list()
            if calcError:
                predtarget,CItarget=self.AverageModel[mod].predictError(featuretf)
            else:
                predtarget=self.AverageModel[mod].predict(featuretf)
                CItarget=np.array(list([i]*2 for i in predtarget)) #no CI interval
            print("======== PREDICTION AVG MODEL =========")
            print(" ***** MODEL: ",mod)
            #print(" prediction:",type(predtarget),"\n",predtarget)
            #print(" target    :",type(target),"\n",target)
            pdata.append(["MODEL",mod])
            if (error=="RMSE"):
                FullError = np.sqrt(mean_squared_error(target, predtarget))
            elif (error=="MAE"):
                FullError = mean_absolute_error(target, predtarget)
            if 'FullError' in vars(): 
                print(error," = ",FullError)
                pdata.append([error, FullError])
            #write the header block to CSV
            writer.writerows(pdata)
            #Now add results as additional columns to the dataframe
            #print("descriptor ",type(descriptors))
            #print("targets ",type(target))
            pandaDescr=descriptors.copy(deep=True)#(For data=None, descriptors are dataframe) make a local copty of the desciptors/features without them being pipelined
            if hasTarget:
                pandaTarg=target.copy(deep=True).to_frame() #make a local copy...and transform the panda-series into a dataframe for the rest to work
                pandaBlock=pd.concat([pandaDescr,pandaTarg],axis=1) #add the target data
            else:
                pandaBlock=pandaDescr
            #pandaBlock=target.copy(deep=True).to_frame() #make a local copy...and transform the panda-series into a dataframe for the rest to work
            pandaBlock['prediction']=predtarget # adding predicted data
            if calcError:
                CIlo=CItarget[:,0] #first column
                CIhi=CItarget[:,1] #second column
                pandaBlock['CI low']=CIlo
                pandaBlock['CI high']=CIhi
            
            pandaBlock.to_csv(OFile,mode='a',header=True)
            
            writer.writerows([[ ]])
            
        OFile.close()
#######################################################################
#######################################################################    
    def to_csv(self,ifile:str=None,frame:str=None,model:str=None):
        """
        Print a data-frame to a CSV file, appending an opened file.
        
        Parameters:
            - ifile : Name of the input file, DEFAULT="Amadeus_modelFrame.csv"
            - frame : which frame to print: "model": modelFrame, "predict": predictFrame DEFAULT:"model" 
            - model : Which specific model should be printed: "all": all models, 
                      "best": the best model, "worst" the worst model DEFAULT="all"
        
        """
        if ifile is None:
            ifile="Amadeus_modelFrame.csv"
        
        if frame is None:
            frame="model"
        
        if model is None:
            model="all"
        
        if model == "all":
            if frame == "predict":
                for mod in self.ModelList.GetModelDictionary():
                    dpf=open(ifile,"a+") #open file for appending
                    dpf.write("#  %s Frame : %s \n" % (frame,mod))
                    dpf.close()   
                    self.PredictFrame[mod].to_csv(ifile,header=None, index=None, sep=' ', mode='a')
            else:#then it should be model
                for mod in self.ModelList.GetModelDictionary():
                    dpf=open(ifile,"a+") #open file for appending
                    dpf.write("#  %s Frame : %s \n" % (frame,mod))
                    dpf.close()   
                    self.ModelFrame[mod].to_csv(ifile,header=None, index=None, sep=' ', mode='a')
        else:# best or worst
            if model == "best":
                modelset="Best model: "
                qual=1.0E10
                modname="None"
                for mod in self.ModelList.GetModelDictionary():
                    modq=self.AverageModel[mod].Quality.getQuality("MAEoob")[0]
                    if modq<qual:
                        qual=modq
                        modname=mod
            else:#we go for the worst
                modelset="Worst model: "
                qual=-1.0E10
                modname="None"
                for mod in self.ModelList.GetModelDictionary():
                    modq=self.AverageModel[mod].Quality.getQuality("MAEoob")[0]
                    if modq>qual:
                        qual=modq
                        modname=mod
            
            dpf=open(ifile,"a+") #open file for appending
            dpf.write("#  %s Frame %s : %s \n" % (frame,modelset,modname))
            dpf.close()   
            if frame == "predict":
                self.PredictFrame[modname].to_csv(ifile,header=None, index=None, sep=' ', mode='a')
            else:#then it should be model
                self.ModelFrame[modname].to_csv(ifile,header=None, index=None, sep=' ', mode='a')
        
        
        
    
