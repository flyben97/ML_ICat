import numpy as np
import pandas as pd
import Caldescriptors
import Stand_data
import Model_ensemble
from Model import xgb_reg

from sklearn.model_selection import train_test_split


file_path = "/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/ICatDataset1128.xlsx" # Dataset Path
sheet_name = "Sheet4"

df = pd.read_excel(file_path, sheet_name=sheet_name) # Read reaction data

file_path_solvent = "/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/ICat_solvent.xlsx" # Dataset Path
sheet_name = "Sheet1"

df_solvent = pd.read_excel(file_path_solvent, sheet_name=sheet_name) # Read reaction data

targets_temp, features_temp = Caldescriptors.caldescriptors(df,df_solvent) # process dataset

features_fp = Stand_data.standardize_data(features_temp, save = True)

print(features_temp)

results = Model_ensemble.modelcal(features_fp, targets_temp)












