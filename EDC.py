from sklearn.metrics import euclidean_distances
import numpy as np
import pandas as pd
import Caldescriptors
import Stand_data
from Model import Lasso
from Model import SVM
from Model import KNN
from Model import DT
from Model import RF
from Model import Adab
from Model import XGB
from Model import ANN


file_path = "/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/ICatDataset1128.xlsx"  # Dataset Path
sheet_name = "Sheet2"

df = pd.read_excel(file_path, sheet_name=sheet_name)  # Read reaction data

file_path_solvent = "/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/ICat_solvent.xlsx"  # Dataset Path
sheet_name_solvent = "Sheet1"

df_solvent = pd.read_excel(file_path_solvent, sheet_name=sheet_name_solvent)  # Read reaction data

targets_temp, features_temp = Caldescriptors.caldescriptors(df, df_solvent)  # process dataset

features_fp = Stand_data.standardize_data(features_temp, save=False)


def custom_train_test_split(features, targets, train_ratio=0.8, random_seed=None):

    distances_matrix = euclidean_distances(features)

    np.random.seed(random_seed)

    num_samples = len(features)
    num_train_samples = int(train_ratio * num_samples)

    avg_distances = np.mean(distances_matrix, axis=1)
    train_indices = np.argsort(avg_distances)[:num_train_samples]

    mask = np.ones(num_samples, dtype=bool)
    mask[train_indices] = False
    test_indices = np.where(mask)[0]

    X_train, y_train = features.iloc[train_indices], targets.iloc[train_indices]
    X_test, y_test = features.iloc[test_indices], targets.iloc[test_indices]

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = custom_train_test_split(features_temp, targets_temp, train_ratio=0.8, random_seed=42)

print(y_train)
print(y_test)
print(X_train)
print(X_test)


train_indices = X_train.index
test_indices = X_test.index

samples_in_both_sets_train = train_indices.intersection(test_indices)
samples_only_in_train = train_indices.difference(test_indices)

samples_in_both_sets_test = test_indices.intersection(train_indices)
samples_only_in_test = test_indices.difference(train_indices)

print("The number of samples present in both the training set and the test set (training set):", len(samples_in_both_sets_train))
print("The number of samples exclusive to the training set:", len(samples_only_in_train))

print("The number of samples present in both the training set and the test set (test set):", len(samples_in_both_sets_test))
print("The number of samples exclusive to the testing set:", len(samples_only_in_test))

#'''
regression_functions = {
    #'Lasso': Lasso.optimize_lasso_hyperparameters,
    #'SVM':SVM.svm_regression_optimization,
    #'KNN':KNN.knn_regression_optimization,
    #'DT':DT.decision_tree_regression_optimization,
    #'RF':RF.random_forest_regression_optimization,
    #'Adab':Adab.adaboost_regression_optimization,
    'XGB':XGB.xgboost_regression_optimization,
    #'ANN':ANN.ANN_regression_optimization
}

results = {}

for regression_name, regression_function in regression_functions.items():

    if regression_name != 'ANN':

        best_params, ave_r2, y_train, y_train_pred, y_test, y_test_pred, train_r2, test_r2, train_rmse, test_rmse = regression_function(X_train, y_train, X_test, y_test, n_trials=100)

        results[regression_name] = {'best_params': best_params,
                                    'ave_r2': ave_r2,
                                    'y_train': y_train,
                                    'y_train_pred': y_train_pred, 
                                    'y_test': y_test,
                                    'y_test_pred': y_test_pred, 
                                    'train_r2': train_r2, 
                                    'test_r2': test_r2,
                                    'train_rmse': train_rmse,
                                    'test_rmse': test_rmse,
                                    }
    
    if regression_name == 'ANN':

        features, targets = ANN.preprocess_data(pd.DataFrame(features_fp),pd.DataFrame(targets_temp))
        best_params, ave_r2, y_train, y_train_pred, y_test, y_test_pred, train_r2, test_r2, train_rmse, test_rmse = regression_function(features, targets)

        
        outdata1 = pd.DataFrame(y_train)
        outdata2 = pd.DataFrame(y_train_pred)
        outdata3 = pd.DataFrame(y_test)
        outdata4 = pd.DataFrame(y_test_pred)


        results[regression_name] = {'best_params': best_params,
                                    'ave_r2': ave_r2,
                                    'y_train': y_train,
                                    'y_train_pred': y_train_pred, 
                                    'y_test': y_test,
                                    'y_test_pred': y_test_pred, 
                                    'train_r2': train_r2, 
                                    'test_r2': test_r2,
                                    'train_rmse': train_rmse,
                                    'test_rmse': test_rmse,
                                    }


for regression_name, result in results.items():
    print("------------------------------------------------------")
    print(f"Model: {regression_name}")
    print(f"best params: {result['best_params']}")
    print(f"cross 5-fold R2: {result['ave_r2']}")
    print(f"train r2: {result['train_r2']}")
    print(f"train rmse: {result['train_rmse']}")
    print(f"test r2: {result['test_r2']}")
    print(f"test rmse: {result['test_rmse']}")
    print("------------------------------------------------------")

#'''
