# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:20:43 2020

contains functionality to perform the post-processing for the averaging paper


@author: Dr. Dr. Danny E. P. Vanpoucke
@web   : https://dannyvanpoucke.be
"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import multiprocessing as mp
import sys
sys.path.append("subroutines/maths")
sys.path.append("subrouties/datahandling")
from Bootstrap import TBootstrap


def plotGrid(grid: np.array ,xmin: float, xmax: float, ymin: float, ymax: float, plotname: str):
    import matplotlib.pyplot as plt
    
    #FIRST write the grid data to a file, one row per line, such that it can be plot by other software lateron as well
    ny, nx=grid.shape
    plottxt=plotname+".dat"
    if os.path.exists(plottxt):
        os.remove(plottxt) #clear the file before we start    
    pltheat=open(plottxt,"a+")
    for i in range(ny):
        pts=(f'{j:.6f}' for j in grid[i,:])
        line=' '
        line=line.join(pts)+"\n"
        pltheat.write(line)
    pltheat.close()

    #SECOND Generate the actual HEATMAP
    # generate 2 2d grids for the x & y bounds
    x, y = np.meshgrid(np.linspace(-xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    
    #we should exclude the zero's, to make sure we see where everything is.
    grid = np.ma.masked_where(grid == 0.00, grid) #grid is now a masked array
    cmap = plt.cm.Greys
    cmap.set_bad(color='#ff00ff') #fuchsia :-)

    z_min = np.amin(grid)
    z_max = np.amax(grid)
    
    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, grid, cmap=cmap, vmin=z_min, vmax=z_max)
    ax.set_title(plotname.replace('_',' '))
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plotpng=plotname+".png"
    plt.savefig(plotpng)
    plt.show(block=False)
    
   
   
    
def getOneLineTrainTest(ds:int,ttsplitResults:dict,fullSetResultsString:str) -> tuple:
    """
    Small function intended to run in parallel. (slow bootstrap)
    
    - ds : integer value indicating the size of the dataset
    - ttsplitResults : dict of train-test data in organised fashion
    - fullSetResultsString: string version of "fullSetResults[ds][1]"
    """
    line=str(ds)+"  " #first column is the datasize
    data=np.array(ttsplitResults[ds]) #needs to be a numpy array to do fortran type slicing...
        
        #1: RMSE train
        #2: RMSE test
        #3: MAE train
        #4: MAE test
        #5: avg RMSE LoO train
        #6: 2sig RMSE LoO train
        #7: avg RMSE 5-fold train
        #8: 2sig RMSE 5-fold train
        #9: intercept b
        #10: coef a1
        #11..: coef a2..
        
    collst=[1, 2, 3, 4, 5, 7] 
    for col in collst:    
        #RMSE train: avg, avg+2sig, avg-2sig, min, max
        avg=data[:,col].mean() #numpy mean
        boot=TBootstrap(data=data[:,col],Func=np.mean)
        boot.NPbootstrap(n_iter=2000, Jackknife=True)
        avgm, avgp = boot.ConfidenceInterval(CItype="BCa",alpha=0.05)#95%confidence interval
        #sig2=2.0*data[:,col].std()
        dmin=np.amin(data[:,col]) #numpy has no amin/amax for numpy nd arrays...
        dmax=np.amax(data[:,col])
        #avgm=avg-sig2
        #avgp=avg+sig2
        line=line+f'{avg:.7f}'+"  "+f'{avgm:.7f}'+"  "f'{avgp:.7f}' \
                +"  "+f'{dmin:.7f}'+"  "+f'{dmax:.7f}'+" | "
        
    line=line+fullSetResultsString+"\n"   
    
    return tuple([ds,line])
    
    
    


def RunPostProcess_MLpaper(basedata: str,baseresult: str,datasizes: list, 
                           predictionData: pd.DataFrame, NumDim: int,NumSKDim: int,
                           theoryModel: list,
                           modelFunction, fitFunction,
                           heatmap: bool=True,
                           n_procs: int=1):
    """
    - basedata : base-string of the datapoint-files (contains the model-coefficients)
    - baseresult: base string of the files containing the results per set
    - datasizes: list of ints giving the zise of the full datasets
    - predictionData: pandas-Frame containing the feature/target data of the artificial model for a set of datapoints to predict.
    - NumDim : the number of dimensions
    - NumSKDim : the number of features in the sk-learn model
    - theoryModel: list of floats with the intercept [index 0], and coefficients [indices 1: ] of the artificial model
    - modelFunction: is the "perfect" version of the function used to generate the data
    - fitFunction: is the function used in ML to create the fit. Note, this is handcoded, not sklearn type function
    - heatmap: bool indicating if heatmaps neet to be generated...DON'T USE FOR EXP DATA
    - n_procs: number of parallel processes to use (when calculating bootstrap CI). Default = 1
    """
    from HPCTools import get_num_procs
    
    print("START POST-PROCESS")
    print("===================")
    print("A. READING/COLLECTING DATA")
    print("---------------------------")
    nrDatasets=len(datasizes)
    #transform dataframe into array
    predictSet=np.array(predictionData.rename_axis('ID').values)
    
    #the datapoints
    allDataSetsFeatures=dict()
    allDataSetsTargets=dict()
    for ds in datasizes:
        dfn=basedata+str(ds)+".dat" #reconstruct filename
        dfile=open(dfn,"r")
        dscheck=int(dfile.readline().replace("#"," "))
        if (ds==dscheck):
            curlst_F=list()
            curlst_T=list()
            for dp in range(ds):
                data=dfile.readline().split()
                tmpl=list(float(x) for x in data[0:-1] )
                curlst_F.append(tmpl)#from the first to the 1 but last column (remember the -1 is not included in a range, so it is the same as Fortran -2)
                curlst_T.append(float(data[-1]))# take the last column
            allDataSetsFeatures[ds]=curlst_F
            allDataSetsTargets[ds]=curlst_T
            
        else:
            print("ERROR: INCONSISTENT DATASIZES IN ",dfn," ",ds," vs ",dscheck)
        dfile.close()
    
    #the RMSE's etc
    fullSetResults=dict()
    ttsplitResults=dict()
    for ds in datasizes:
        dfn=baseresult+str(ds)+".dat" #reconstruct filename
        dfile=open(dfn,"r")
        nruns, ncols=tuple(int(i) for i in dfile.readline().replace("#"," ").split())
        nruns-=1
        data=dfile.readline().split()
        curlst=list([int(data[0])])
        curlst.extend(list(float(i) for i in data[1:]))
        fullSetResults[ds]=curlst
        curlst=list()
        for dp in range(nruns):
            data=dfile.readline().split()
            clr=list([int(data[0])])
            clr.extend(list(float(i) for i in data[1:]))
            curlst.append(clr)
            
        ttsplitResults[ds]=curlst
        dfile.close()
        
    print("B. Generating RMSE-curves ")
    print("---------------------------")
    
    
    ############################################################################
    ##########  TRAIN-TEST RESULTS #############################################
    ############################################################################
    header=list()
    header.append("# datasize RMSE TRAIN: avg  CIlo  CIhi  min  max ")
    header.append("TEST: avg  CIlo  CIhi  min  max ")
    header.append("MAE TRAIN: avg  CIlo  CIhi  min  max ")
    header.append("MAE TEST : avg  CIlo  CIhi  min  max ")
    header.append("avg-RMSE-LoO: avg  CIlo  CIhi  min  max ")
    header.append("avg-RMSE-5CV: avg  CIlo  CIhi  min  max ")
    header.append(" RMSE-full-set \n")
    
    headerstr=""            
    headerstr=headerstr.join(header) #needs to be assigned because join "only" returns the string...but it needs a real string to be possible to call
    
    pltdat="PlotResults_TRAINTEST.dat"
    if os.path.exists(pltdat):
        os.remove(pltdat) #clear the file before we start    
    plttrainf=open(pltdat,"a+")
    plttrainf.write(headerstr)
    
    # Parallelisation for sections performing bootstraps. 
    # Parallelization only at the highest level of a datasize, not a column, 
    # this to keep overhead low, and deal with slowdowns due to large number of datasizes
    # 1. create our process pool
    pool=mp.Pool(processes=get_num_procs(n_procs))
    # 2. set drones to work
    drones=[pool.apply_async(getOneLineTrainTest, args=(ds,ttsplitResults,str(fullSetResults[ds][1]))) for ds in datasizes]
    # 3. as we can not assume the lines to be produced in the correct order
    #    and numbering is non-linear or incremental--> make it a dict
    lineDict=dict()
    
    for drone in drones:
        ds,line=drone.get() 
        lineDict[ds]=line
    # 4. wait untill all processes are finished
    pool.close()
    pool.join()
    # 5. and now do the writing in an orderly fashion
    for ds in datasizes:
        plttrainf.write(lineDict[ds])
        
#        line=str(ds)+"  " #first column is the datasize
#        data=np.array(ttsplitResults[ds]) #needs to be a numpy array to do fortran type slicing...
#
#        #0: index
#        #1: RMSE train
#        #2: RMSE test
#        #3: MAE train
#        #4: MAE test
#        #5: avg RMSE LoO train
#        #6: 2sig RMSE LoO train
#        #7: avg RMSE 5-fold train
#        #8: 2sig RMSE 5-fold train
#        #9: intercept b
#        #10: coef a1
#        #11..: coef a2..
#        
#        collst=[1, 2, 3, 4, 5, 7] 
#        for col in collst:    
#            #RMSE train: avg, avg+2sig, avg-2sig, min, max
#            avg=data[:,col].mean() #numpy mean
#            boot=TBootstrap(data=data[:,col],Func=np.mean)
#            boot.NPbootstrap(n_iter=2000, Jackknife=True)
#            avgm, avgp = boot.ConfidenceInterval(CItype="BCa",alpha=0.05)#95%confidence interval
#            #sig2=2.0*data[:,col].std()
#            dmin=np.amin(data[:,col]) #numpy has no amin/amax for numpy nd arrays...
#            dmax=np.amax(data[:,col])
#            #avgm=avg-sig2
#            #avgp=avg+sig2
#            line=line+f'{avg:.7f}'+"  "+f'{avgm:.7f}'+"  "f'{avgp:.7f}' \
#                        +"  "+f'{dmin:.7f}'+"  "+f'{dmax:.7f}'+" | "
#        
#        line=line+str(fullSetResults[ds][1])+"\n"   
#        plttrainf.write(line)
    plttrainf.close()
    
    ############################################################################
    ##########  ON PREDICTED DATA 1000 POINTS ##################################
    ##########  MODELCOEFFICIENTS ##############################################
    ############################################################################

    #How well do we predict 1000 datapoints
    headerstr=("# datasize RMSE: Theory    Avg-model    Full-Model     Best-RMSE    Worst-RMSE | MAE: Theory    Avg-model    Full-Model     Best-MAE    Worst-MAE \n ")           
    pltdat="PlotResults_RMSEonPredict1K.dat"
    if os.path.exists(pltdat):
        os.remove(pltdat) #clear the file before we start    
    plttrainf=open(pltdat,"a+")
    plttrainf.write(headerstr)
    #print(headerstr)
    
    #location to print model coefficients
    headermodel=("datasetsize ) Avg: Intercept Coeff1..n | Full: Intercept Coeff1..n  | best RMSE: Intercept Coeff1..n  | worst RMSE: Intercept Coeff1..n   \n ")           
    modeldat="PlotResults_Model_coefficients.dat"
    if os.path.exists(modeldat):
        os.remove(modeldat) #clear the file before we start    
    pltmodel=open(modeldat,"a+")
    pltmodel.write(headermodel)
    #print(headermodel)
    
    #loop over all datasetsizes
    for ds in datasizes:
        line=f'{ds:5}'+"  " #first column is the datasize
        linemodel=f'{ds:5}'+" ) " #first column is the datasize
        data=np.array(ttsplitResults[ds]) #needs to be a numpy array to do fortran type slicing...
        #0: run index
        #1: RMSE train
        #2: RMSE test
        #3: MAE train
        #4: MAE test
        
        #3 5: avg RMSE LoO train
        #4 6: 2sig RMSE LoO train
        #5 7: avg RMSE 5-fold train
        #6 8: 2sig RMSE 5-fold train
        #7 9: intercept b
        #8 10: coef a
        
        x=list(x for x in predictSet[:,0:-1]) #put the features in a list of lists, every row are the different features of 1 run
        
        #print("The X's:\n",x)
        #print("The X[1]:\n",x[0])
        
        
        
        #what is the error introduced due to the noise on our theoretical model (this should be the best)
        if len(theoryModel)>0 :
            a=theoryModel[1:]
            b=theoryModel[0]
            #pred=a*predictSet[:,0:-2] + b
            pred=modelFunction(x=x,intercept=b,slope=a)
            #pred=list(np.dot(a,x) + b for x in predictSet[:,0:-1] ) 
            rmseTheory=np.sqrt(mean_squared_error(predictSet[:,-1],pred)) 
            maeTheory=mean_absolute_error(predictSet[:,-1],pred)
        else:
            rmseTheory=0
            maeTheory=0
            
        #How well is the averaged model doing
        #a=data[:,8:].mean()
        a=list(column.mean() for column in data[:,10:].T) #the for returns rows, by transposing it gives the columns
        b=data[:,9].mean()
        #print("The AVG Intercept's:\n",b)
        #print("The AVG Coeffs's:\n",a)
        
        #pred=a*predictSet[:,0] + b
        pred=fitFunction(x=x,intercept=b,slope=a)
        #print("Prediction:\n",pred)
        
        
        #pred=list(np.dot(a,x) + b for x in predictSet[:,0:-1] ) 
        rmseAvg=np.sqrt(mean_squared_error(predictSet[:,-1],pred))
        maeAvg=mean_absolute_error(predictSet[:,-1],pred)
        a_coeffs=('  '.join(['%.7f']*len(a)))%tuple(a)
        linemodel=linemodel+f'{b:.7f}'+"   "+a_coeffs+" | "
        
#        print(" Intercept=",b)
#        print(" Coef     =",a)
#        print(" x's      =",x)
#        print(" Predict  =",pred)
#        print(" PredTarg =",predictSet[:,-1])
#        print("RMSE      =",rmseAvg)
        
        
        
        #How well does a model which used the full data-set (train+test) perform?
        a=fullSetResults[ds][10:]
        b=fullSetResults[ds][9]
        #pred=a*predictSet[:,0] + b
        pred=fitFunction(x=x,intercept=b,slope=a)
        #pred=list(np.dot(a,x) + b for x in predictSet[:,0:-1] ) 
        rmseFull=np.sqrt(mean_squared_error(predictSet[:,-1],pred))
        maeFull=mean_absolute_error(predictSet[:,-1],pred)
        a_coeffs=('  '.join(['%.7f']*len(a)))%tuple(a)
        linemodel=linemodel+f'{b:.7f}'+"   "+a_coeffs+" | "
        
        
        #How well is the "best RMSE of test" model doing
        #find the index of the best RMSE
        posRMSE=np.where(data[:,2] == np.amin(data[:,2])) #index returns the first occurence
        a=data[posRMSE[0][0],10:]
        b=data[posRMSE[0][0],9]
        pred=fitFunction(x=x,intercept=b,slope=a)
        rmseBestRMSE=np.sqrt(mean_squared_error(predictSet[:,-1],pred))
        a_coeffs=('  '.join(['%.7f']*len(a)))%tuple(a)
        linemodel=linemodel+f'{b:.7f}'+"   "+a_coeffs+" | "
        #How well is the "worst RMSE of test" model doing
        #find the index of the worst RMSE
        posRMSE=np.where(data[:,2] == np.amax(data[:,2])) #index returns the first occurence
        a=data[posRMSE[0][0],10:]
        b=data[posRMSE[0][0],9]
        pred=fitFunction(x=x,intercept=b,slope=a)
        rmseWorstRMSE=np.sqrt(mean_squared_error(predictSet[:,-1],pred))
        a_coeffs=('  '.join(['%.7f']*len(a)))%tuple(a)
        linemodel=linemodel+f'{b:.7f}'+"   "+a_coeffs+" \n "
        
        
        #How well is the "best MAE of test" model doing
        #find the index of the best MAE
        posMAE=np.where(data[:,4] == np.amin(data[:,4])) #index returns the first occurence
        a=data[posMAE[0][0],10:]
        b=data[posMAE[0][0],9]
        pred=fitFunction(x=x,intercept=b,slope=a)
        maeBestMAE=mean_absolute_error(predictSet[:,-1],pred)
        #How well is the "worst MAE of test" model doing
        #find the index of the worst MAE
        posMAE=np.where(data[:,4] == np.amax(data[:,4])) #index returns the first occurence
        a=data[posMAE[0][0],10:]
        b=data[posMAE[0][0],9]
        pred=fitFunction(x=x,intercept=b,slope=a)
        maeWorstMAE=mean_absolute_error(predictSet[:,-1],pred)
        
        line=line+f'{rmseTheory:.7f}'+"  "+f'{rmseAvg:.7f}'+"  "+f'{rmseFull:.7f}'+\
                "  "+f'{rmseBestRMSE:.7f}'+"  "+f'{rmseWorstRMSE:.7f}'+" | "+\
                f'{maeTheory:.7f}'+"  "+f'{maeAvg:.7f}'+"  "+f'{maeFull:.7f}'+\
                "  "+f'{maeBestMAE:.7f}'+"  "+f'{maeWorstMAE:.7f}'+" \n "
        
        plttrainf.write(line)
        pltmodel.write(linemodel)
        
    plttrainf.close()
    pltmodel.close()
        
    ############################################################################
    ##########  HEATMAPS #######################################################
    ############################################################################
    print("C. Generating HEATMAPs ")
    print("---------------------------")
    if heatmap:
        
        width=1.0
        dx=width*0.01
        nx=int(width/dx) +1
        
        ymin=np.amin(predictSet[:,-1])  #theoryModel[0]      #intercept
        ymax=np.amax(predictSet[:,-1])  #ymin+height #intercept + slope's
        height=ymax-ymin
        #print("HEIGHT=",height,"  ===  ",ymax," - ", ymin)
        
        
        dy=dx
        ny=int(height/dy) +1
        xval=list([0]*nx)
        for x in range(nx):
            xval[x]=x*dx
    
        NFeat=NumSKDim
        for dim in range(NumDim):
            dimi=NFeat-dim
            #print("DIM in NUMDIM=",dim,"of",NumDim," -> DIMI=",dimi,"   NFeature=",NFeat," theorymodel[1:]=",theoryModel[1:])
            for ds in datasizes:
                grid=np.zeros((ny,nx))
                data=np.array(ttsplitResults[ds]) #needs to be a numpy array to do fortran type slicing...
                num_rows, num_cols = data.shape
                #print("NUM_ROWS NUMCOLS=", num_rows, num_cols)
                for run in range(num_rows):
                    #a=data[run,num_cols-dimi]
                    #b=data[run,num_cols-NFeat-1]
                    #print("slope=",data[run,num_cols-dimi:])
                    
                    yr=fitFunction(x=xval,intercept=data[run,num_cols-NFeat-1],slope=data[run,num_cols-dimi:])
                    #print("YR=",yr)
                    yi=((yr-ymin)/height)*(ny-1) #array operation-> transformation to int can not be done at array level
                    
                    for x in range(nx):
                        #yr=a*xval[x]+b
                        #find position in grid
                        #yi=int(((yr-ymin)/height)*(ny-1))
                        yii=int(yi[x])
                        if (yii>-1)and(yii<ny):
                            grid[yii,x]+=1.0
                maxval=np.amax(grid)
                if (maxval==0):
                    print("WARNING: MAXVAL=0...NOTING IN HEATMAP--> NOT GOOD")
                    maxval=1
                grid=grid/maxval
                print("for ",ds," the maxval= ",maxval)
                
                plotname="Heatmap_Dim"+str(dimi)+"_"+str(ds)
                plotGrid(grid,xmin=0.0, xmax=1.0, ymin=ymin, ymax=ymax, plotname=plotname)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

