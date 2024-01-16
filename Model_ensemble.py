from Model import Lasso
from Model import SVM
from Model import KNN
from Model import DT
from Model import RF
from Model import Adab
from Model import XGB
from Model import ANN

from sklearn.model_selection import train_test_split
import pandas as pd



def modelcal(features_fp, targets_temp):

    regression_functions = {
        'Lasso': Lasso.optimize_lasso_hyperparameters,
        'SVM':SVM.svm_regression_optimization,
        'KNN':KNN.knn_regression_optimization,
        'DT':DT.decision_tree_regression_optimization,
        'RF':RF.random_forest_regression_optimization,
        'Adab':Adab.adaboost_regression_optimization,
        'XGB':XGB.xgboost_regression_optimization,
        'ANN':ANN.ANN_regression_optimization
    }

    results = {}

    # Invoke the function in the dictionary for hyperparameter optimization and store the results
    for regression_name, regression_function in regression_functions.items():

        if regression_name != 'ANN':

            X_train, X_test, y_train, y_test = train_test_split(features_fp, targets_temp, test_size=0.2, random_state=42)
            best_params, ave_r2, y_train, y_train_pred, y_test, y_test_pred, train_r2, test_r2, train_rmse, test_rmse = regression_function(X_train, y_train, X_test, y_test, n_trials=100)

            excel_file_train = '/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/' + str(regression_name) + '_train.xlsx'
            excel_file_train_pred = '/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/' + str(regression_name) + '_train_pred.xlsx'
            excel_file_test = '/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/' + str(regression_name) + '_test.xlsx'
            excel_file_test_pred = '/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/' + str(regression_name) + '_test_pred.xlsx'
            
            outdata1 = pd.DataFrame(y_train)
            outdata2 = pd.DataFrame(y_train_pred)
            outdata3 = pd.DataFrame(y_test)
            outdata4 = pd.DataFrame(y_test_pred)

            outdata1.to_excel(excel_file_train)
            outdata2.to_excel(excel_file_train_pred)
            outdata3.to_excel(excel_file_test)
            outdata4.to_excel(excel_file_test_pred)


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

            excel_file_train = '/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/' + str(regression_name) + '_train.xlsx'
            excel_file_train_pred = '/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/' + str(regression_name) + '_train_pred.xlsx'
            excel_file_test = '/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/' + str(regression_name) + '_test.xlsx'
            excel_file_test_pred = '/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/' + str(regression_name) + '_test_pred.xlsx'
            
            outdata1 = pd.DataFrame(y_train)
            outdata2 = pd.DataFrame(y_train_pred)
            outdata3 = pd.DataFrame(y_test)
            outdata4 = pd.DataFrame(y_test_pred)

            outdata1.to_excel(excel_file_train)
            outdata2.to_excel(excel_file_train_pred)
            outdata3.to_excel(excel_file_test)
            outdata4.to_excel(excel_file_test_pred)

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
    

    # Print the results for all models.
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


    return results