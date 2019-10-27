import numpy as np
import pandas as pd # Used only for loading and subsetting the dataset
from proj1_helpers import *
from implementations import *
from expansion import *

DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH  = '../data/test.csv'
OUTPUT_PATH     = 'predictions.csv'
POLYNOMIAL_EXPANSION_DEGREE = 13

"""
Data cleaning and data preparation functions
"""
def prepare_dataset(dataset):    
    """
    Prepare the Higgs boson dataset :
        - split the data set in different subset
        - handle unavailable cells
    
    Parameters
    ----------
    dataset: DataFrame
        Pandas dataframe containing the raw dataset of higgs boson
    Returns
    -------
    list
        List of Dataframe, each dataframe is a prepared part of the dataset
    """
    dataset_mass_def = dataset[dataset.DER_mass_MMC != -999].copy()
    dataset_mass_not_def = dataset[dataset.DER_mass_MMC == -999].copy()
    dataset_mass_not_def = dataset_mass_not_def.drop(['DER_mass_MMC'],1)

    def split_on_jet_num(dataset, DER_mass_MMC_is_defined):
        """
        Split the dataset in meaningful subdataset.
        Parameters
        ----------
        dataset: DataFrame
            Pandas dataframe containing the Higgs boson dataset to split
        DER_mass_MMC_is_defined: boolean
            boolean indicating if the dataset is with a mass or without a mass
        Returns
        -------
        list
            List of Dataframe, each dataframe is a prepared part of the dataset
        """
        dataset = dataset.replace(-999, np.nan)
        if(DER_mass_MMC_is_defined):
            dataset = mass_DER_importance_feature_expansion(dataset)
    
        pri0_to_drop =["DER_deltaeta_jet_jet","DER_mass_jet_jet","DER_prodeta_jet_jet","DER_lep_eta_centrality","PRI_jet_leading_pt","PRI_jet_leading_eta","PRI_jet_leading_phi","PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi"]
        pri1_to_drop = ["DER_deltaeta_jet_jet","DER_mass_jet_jet","DER_prodeta_jet_jet","DER_lep_eta_centrality","PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi"]


        pri0 = dataset[dataset.PRI_jet_num==0].copy()
        pri0 = pri0.drop(pri0_to_drop,1)
        pri0 = pri0.drop(["PRI_jet_num","PRI_jet_all_pt"],1)

        pri1 = dataset[dataset.PRI_jet_num == 1].copy()
        pri1 = pri1.drop(pri1_to_drop,1)
        pri1 = pri1.drop(["PRI_jet_num"],1)

        pri2 = dataset[dataset.PRI_jet_num == 2].copy()
        pri2 = pri2.drop(["PRI_jet_num"],1)

        pri3 = dataset[dataset.PRI_jet_num == 3].copy()
        pri3 = pri3.drop(["PRI_jet_num"],1)

        return [pri0,pri1,pd.concat([pri2,pri3])]
    
    return split_on_jet_num(dataset_mass_def, True) + split_on_jet_num(dataset_mass_not_def, False)

def extract_predictions(dataset):
    """
    Extract the y vectors of a Higgs boson dataset.

    Parameters
    ----------
    dataset: DataFrame
        Pandas dataframe containing the Higgs boson dataset
    Returns
    ----------
    Series
        Pandas serie that correspond to the y vector of the dataset
    """
    return dataset.Prediction.apply(lambda x: -1 if x == 'b' else 1)

def mass_DER_importance_feature_expansion(dataset):
    """
    Give some importance to the DER mass.

    Parameters
    ----------
    dataset: DataFrame
        Pandas dataframe containing the Higgs boson dataset
    Returns
    ----------
    dataset: DataFrame
        Pandas dataframe containing the Higgs boson dataset but with the new column
    """
    dataset['Mass_let_tau_sum_pt_ratio'] = dataset.DER_mass_MMC * dataset.DER_pt_ratio_lep_tau / (dataset.DER_sum_pt+1e-10)
    return dataset

def tilda_x(X):
    """
    Compute the tilda matrix of the feature matrix.

    Parameters
    ----------
    X: ndarray
        Feature matrix
    Returns
    ----------
    ndarray
        tilda matrix of the feature matrix
    """
    return np.c_[np.ones(X.shape[0]), X]

def normalize_dataset(dataset):
    """
    Normalize the dataset.

    Parameters
    ----------
    dataset: ndarray
        dataset to be normalized
    Returns
    ----------
    ndarray
        dataset normalized    
    """
    dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0)
    dataset = np.nan_to_num(dataset)
    dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis = 0)
    return dataset


"""
Prediction function
"""
def predict(test_dataset,test_tX,weights):
    """
    Predict the labels for the test dataset.

    Parameters
    ----------
    test_dataset: ndarray
        dataset for wich the prediction must be done
    test_tX: list
        list of ndarray representing each subdataset
    weights: ndarray
        weight of the trained model
    Returns
    ----------
    ndarray
        test dataset with prediction     
    """
    for idx, dataset in enumerate(test_tX):
        test_dataset[idx]['Prediction'] = predict_labels(weights[idx],dataset)
    return test_dataset

"""
Data preparation
"""
# Loading the data
raw_dataset = pd.read_csv(DATA_TRAIN_PATH, sep=',')
pd.options.display.max_columns = None

# Preparing the data
prepared_dataset = raw_dataset.drop(['Id'], 1)
prepared_dataset = prepare_dataset(prepared_dataset)
y = []

for idx, sub_dataset in enumerate(prepared_dataset):
    y.append(extract_predictions(sub_dataset))
    sub_dataset = sub_dataset.drop(['Prediction'],1).to_numpy()
    #Compute tilda x matrix for the sub_dataset
    prepared_dataset[idx] = tilda_x(normalize_dataset(polynomial_expansion(normalize_dataset(sub_dataset), POLYNOMIAL_EXPANSION_DEGREE)))


"""
Training the model
"""
# parameters optimized for the ridge regression computed by cross-validation
ridge_parameters = [0.004393970560760791, 0.001, 1e-05, 0.01, 0.00031622776601683794, 3.1622776601683795e-05]

weights = []
for i in range(len(prepared_dataset)):
    w, loss = ridge_regression(y[i], prepared_dataset[i], ridge_parameters[i])
    weights.append(w)

"""
Generate predictions for test set
"""
# Load the test set
test_dataset_raw = pd.read_csv(DATA_TEST_PATH, sep=',')

# Prepare the test set in the same format as the training set
test_dataset_prepared = test_dataset_raw.drop(['Prediction'], 1)
test_dataset_prepared = test_dataset_prepared.set_index(['Id'])
test_dataset_prepared = prepare_dataset(test_dataset_prepared)
test_tX = []
for idx , sub_dataset in enumerate(test_dataset_prepared):
    test_tX.append(tilda_x(normalize_dataset(polynomial_expansion(normalize_dataset(sub_dataset.to_numpy()), POLYNOMIAL_EXPANSION_DEGREE))))

# Compute the prediction with the trained model
test_prediction = predict(test_dataset_prepared,test_tX,weights)
test_prediction = pd.concat(test_prediction,sort=True)
test_prediction = test_prediction.sort_index()

# Save results
create_csv_submission(test_prediction.Prediction.keys(), test_prediction.Prediction.values, OUTPUT_PATH)
