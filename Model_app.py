import numpy as np
import pandas as pd
import Caldescriptors
import Stand_data
import xgboost as xgb

from sklearn.model_selection import train_test_split


file_path = "/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/ICatDataset_test.xlsx" # Dataset Path
sheet_name = "Sheet1"

df = pd.read_excel(file_path, sheet_name=sheet_name) # Read reaction data

targets_temp, features_temp = Caldescriptors.caldescriptors(df) # process dataset

features_fp = Stand_data.standardize_data(features_temp, save = False)

xgb_model = xgb.Booster()
xgb_model.load_model('/mnt/c/Ben_workspace/PythonCode/ICat/Model/ICat_xgb2.model') 

XGB_predictions = xgb_model.predict(xgb.DMatrix(features_fp))
xgb_pred = pd.DataFrame(XGB_predictions)

print(xgb_pred)