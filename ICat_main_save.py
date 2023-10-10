import numpy as np
import pandas as pd
import Caldescriptors
import Stand_data
import Model_ensemble
from Model import xgb_reg
from sklearn.model_selection import train_test_split


file_path = "/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/ICatDataset.xlsx" # Dataset Path
sheet_name = "Sheet1"

df = pd.read_excel(file_path, sheet_name=sheet_name) # Read reaction data

targets_temp, features_temp = Caldescriptors.caldescriptors(df) # process dataset

features_fp = Stand_data.standardize_data(features_temp, save = False)

#results = Model_ensemble.modelcal(features_fp, targets_temp)




# xgb model 
file_path2 = "/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/feature_name2.xlsx"
feature_name_fp_pp = pd.read_excel(file_path2, sheet_name=sheet_name)

#'''
X_train, X_test, y_train, y_test = train_test_split(features_fp, targets_temp, test_size=0.2, random_state=42)
final_r2, rmse, train_pred, y_pred, feature_importance_df = xgb_reg.xgboost_regression_optimization(X_train, y_train, X_test, y_test, feature_name = feature_name_fp_pp)
features_fp = pd.DataFrame(features_fp)
feature_importance_df.to_excel('/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/feature_importance_df_0926.xlsx')
#features_fp.to_excel('/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/features_0923.xlsx')
#'''
