# Higgs Boson Detection

## Files
* run.py : main program that compute our best solution (ridge regression) for this challenge
* proj1_helpers.py : helpers functions for loading and saving csv
* implementations.py : ML learning algorithm implementation
* expansion.py : provide function to expand a feature matrix
* common_functions.py : provide functions that are used by ML algorithms (example : cost and loss functions)

## Prerequisites
### Libraries
* Python 3, with the following libraries installed :
* NumPy, allows efficient computations
* Pandas, this library is used only for data loading and data subsetting 

### Data
In order to run the solution, the datasets for the Higgs Boson detection challenge are required. Both files (data.csv and test.csv) should be placed in the data folder.

## Running our solution
In order to run our solution, we provide the */scripts/run.py* file.
 ```
cd  path/scripts/run.py
python run.py
 ``` 
This will load the data, prepare it, train the model with already optimized hyper-parameters and compute the prediction for the test dataset. 

## Configurations
The */script/run.py* file can be configured by three variables :
* DATA_TRAIN_PATH : path of the train dataset (train.csv)
* DATA_TEST_PATH  : path of the test dataset (test.csv)
* OUTPUT_PATH     : path of the output csv that will contain predictions that are submittable on aicrowd.com

## Team:
- Bastien Beuchat
- Robin Mamié
- Jeremy Mion

