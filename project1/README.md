# Higgs Boson Detection

All the code used to reproduce the results of the Higgs Boson detection is available on he script folder. The score of 81.4% was achieved but because of an unseeded attempt, the code provided here can only reproduce a score of 81.2% on aicrowd.com.

## Files

The following files are located in the `scripts` folder.

* `run.py`: the main program that computes our best solution (ridge regression) for this challenge
* `proj1_helpers.py`: helpers functions for loading and saving csv
* `implementations.py`: ML learning algorithm implementation
* `expansion.py`: provides a function used to expand a feature matrix
* `common_functions.py`: provides functions that are used by ML algorithms (example: cost and gradient functions)

## Prerequisites

### Libraries

* Python 3, with the following libraries installed:
    * NumPy, allows efficient computations
    * Pandas, this library is used only for data loading and data subsetting (used in `run.py`)

### Data

In order to run the solution, the datasets for the Higgs Boson detection challenge are required. Both files (data.csv and test.csv) should be placed in the `data` folder.


## Running our solution

In order to run our solution, we provide the `scripts/run.py` file.

    cd  scripts
    python run.py

This will load the data, prepare it, train the model with already optimized hyper-parameters and compute the prediction for the test dataset.

This will create the file `predictions.csv` in the script folder.

## Configurations

The `scripts/run.py` file can be configured with these three variables:

* DATA_TRAIN_PATH: path of the train dataset (train.csv)
* DATA_TEST_PATH: path of the test dataset (test.csv)
* OUTPUT_PATH: path of the output csv that will contain predictions that can be submitted on aicrowd.com

## Machine learning algorithms

Six machines algorithms are implemented in implementations.py. Each one has its own method. The six methods return a tuple __(weights, loss)__. The six different signatures are the following:

| Function                                                             | Details                                             |
|----------------------------------------------------------------------|-----------------------------------------------------|
| least_squares GD(y, tx, initial w, max iters, gamma)                 | Linear regression using gradient descent            |
| least_squares_SGD(y, tx, initial w, max iters, gamma)                | Linear regression using stochastic gradient descent |
| least_squares(y, tx)                                                 | Least squares regression using normal equations     |
| ridge_regression(y, tx, lambda )                                     | Ridge regression using normal equations             |
| logistic_regression(y, tx, initial w, max iters, gamma)              | Logistic regression using  SGD                      |
| reg_logistic_regression(y, tx, lambda , initial w, max iters, gamma) | Regularized logistic regression using SGD           |

The labels fed into `logistic_regression` and `reg_logistic_regresssion` should be either 0s or 1s, and not -1s and 1. Otherwise, the loss will not make any sense.

More accurate documentation about the type of the parameters is available in the file `scripts/implementations.py`.

## Team:

- Bastien Beuchat
- Robin Mamié
- Jeremy Mion

