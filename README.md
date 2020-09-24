# AMADEUS
**Artificial intelligence and Machine learning frAmework for DEsigning Useful materialS**

AMADEUS is an artificial intellegence program capable of dealing with small datasets in a highly independent manner (with minimal interference of the user beyond providing input data). First results have been published in [\[1\]](#paper1_AvgModel) and was selected as a *featured article* by the Journal of Applied Physics [\[2\]](#paper1_scilight).  

## Usage
To run experiments using the current version of Amadeus two files need to be modified.
1. **The main-driver-script**
    
    Modify the Main.py script to run the experiments:
    - load & preprocess your data
    - run the experiments using a TAmadeus class instance
    - store the results
    
    Examples are to be found in the files indicated as main
    
2. **The TAmadeusFrameWork**
    
    The current development version of Amadeus is not fully automated yet. The user should switch on/off the models of interest in
    *def RunModelEnsemble(self)* by (un)commenting and setting further parameters of the models in the (un)commented lines. 
    (This is around line 200-250 at the moment of writing)
    



## Current Status (disclaimer)
The current version available is a development version.


## Bibliography
* <a name="paper1_AvgModel">\[1\]<a> *"Small Data Materials Design with Machine Learning: When the Average Model Knows Best"*,</br>
 Danny E. P. Vanpoucke, Onno S. J. van Knippenberg, Ko Hermans, Katrien V. Bernaerts, and Siamak Mehrkanoon, 
*Journal of Applied Physics* **128(5)**, 054901 (2020).</br>
DOI: [10.1063/5.0012285](https://dx.doi.org/10.1063/5.0012285)
* <a name="paper1_scilight">\[2\]<a> *"When the average model knows best"*,</br>
 S. Mandel, 
*AIP Scilight* of [\[1\]](#paper1_AvgModel), 7 august 2020.</br>
DOI: [10.1063/10.0001749](https://dx.doi.org/10.1063/10.0001749)
