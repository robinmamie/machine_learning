import numpy as np
import pandas as pd # Used only for loading and subsetting the dataset
from proj1_helpers import *
from implementations import *
from expansion import *

DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'
OUTPUT_PATH = 'predictions.csv'

"""
Data cleaning and preparation functions
"""

def cleanDataSet(dataset):
    dataset_mass_def = dataset[dataset.DER_mass_MMC != -999].copy()
    dataset_mass_not_def = dataset[dataset.DER_mass_MMC == -999].copy()
    dataset_mass_not_def = dataset_mass_not_def.drop(['DER_mass_MMC'],1)
    def splitOnJetNum(dataset, DER_mass_MMC_is_defined):
        dataset = dataset.replace(-999, np.nan)
        if(DER_mass_MMC_is_defined):
            dataset = nonPolyFeatureExpansion(dataset)
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
    
    return splitOnJetNum(dataset_mass_def, True) + splitOnJetNum(dataset_mass_not_def, False)

def extractPredictions(dataset):
    return dataset.Prediction.apply(lambda x: -1 if x == 'b' else 1)

def nonPolyFeatureExpansion(data):
    data['Mass_let_tau_sum_pt_ratio'] = data.DER_mass_MMC * data.DER_pt_ratio_lep_tau / (data.DER_sum_pt+1e-10)
    return data

def tildaNumpy(X):
    return np.c_[np.ones(X.shape[0]), X]

def normalizeDataset_numpy(dataset):
    dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0)
    dataset = np.nan_to_num(dataset)
    dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis = 0)
    return dataset

def normalizeDataset(dataset):
    dataset = (dataset - dataset.mean()) / dataset.std()
    dataset = dataset.fillna(0)
    dataset = (dataset - dataset.mean()) / dataset.std()
    return dataset

def predict(test_pri,test_pri_tX, w_pri):
    for idx, dataset in enumerate(test_pri_tX):
        test_pri[idx]['Prediction'] = predict_labels(w_pri[idx],dataset)
    return test_pri


# Loading the data
hb = pd.read_csv(DATA_TRAIN_PATH, sep=',')
pd.options.display.max_columns = None
hb = hb.drop(['Id'], 1)

POLYNOMIAL_EXPANSION_DEGREE = 13

pri = cleanDataSet(hb)
predictions = []
pri_cross_validation_test = []
prediction_cross_validation_test = []

for idx, dataset in enumerate(pri):
    predictions.append(extractPredictions(dataset))
    dataset = dataset.drop(['Prediction'],1)
    pri[idx] = tildaNumpy(normalizeDataset_numpy(polynomial_expansion( normalizeDataset(dataset).to_numpy(), POLYNOMIAL_EXPANSION_DEGREE)))


"""
Training the model
"""

# parameters optimized for the ridge regression computed by cross-validation
ridge_parameters = [0.004393970560760791, 0.001, 1e-05, 0.01, 0.00031622776601683794, 3.1622776601683795e-05]

w_pri = []
for i in range(len(pri)):
    w, loss = ridge_regression(predictions[i], pri[i], ridge_parameters[i])
    w_pri.append(w)

"""
Generate predictions for test set
"""
# Load the test set
hbt = pd.read_csv(DATA_TEST_PATH, sep=',')

# Put the test set in the same format as the training set
hbt = hbt.drop(['Prediction'], 1)
hbt = hbt.set_index(['Id'])
test_pri = cleanDataSet(hbt)
test_pri_tX = [] # tX arrays to run prediction on
for idx , dataset in enumerate(test_pri):
    test_pri_tX.append( tildaNumpy(normalizeDataset_numpy(polynomial_expansion( normalizeDataset(dataset).to_numpy(), POLYNOMIAL_EXPANSION_DEGREE))))

# Compute the prediction with the trained model
test_prediction = predict(test_pri,test_pri_tX,w_pri)
test_prediction = pd.concat(test_prediction,sort=True)
test_prediction = test_prediction.sort_index()

# Save results
create_csv_submission(test_prediction.Prediction.keys(), test_prediction.Prediction.values, OUTPUT_PATH)


