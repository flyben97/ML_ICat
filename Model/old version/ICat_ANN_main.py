import numpy as np
import pandas as pd
import optuna
from Sta_data import standardize_data as sdata 
from MACCSKeys_fp import smi_fp
from Physicochem_prop import smi_pp
from descriptors import rdkit_descriptor as rd_ds
from descriptors import rdkit_fingerprint as rd_fp
import ICat_fingerprint as ICat_fp_cal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Model import ANN_reg 

file_path = "/mnt/c/Ben_workspace/PythonCode/ICat/Dataset/ICatDataset.xlsx"
sheet_name = "Sheet1"

df = pd.read_excel(file_path, sheet_name=sheet_name)

sub_smi = df.iloc[:,1]
precat_smi = df.iloc[:,2]

add_smi1 = df.iloc[:,5]
add_smi2 = df.iloc[:,7]

sol_smi1 = df.iloc[:,9]
sol_smi2 = df.iloc[:,10]
#====================================================================================
#========================MACCSKeys molecular fingerprint=============================
#====================================================================================

sub_fp_temp = smi_fp(sub_smi,df)
precat_fp_temp = smi_fp(precat_smi,df)
add1_fp_temp = smi_fp(add_smi1,df)
add2_fp_temp = smi_fp(add_smi2,df)
sol1_fp_temp = smi_fp(sol_smi1,df)
sol2_fp_temp = smi_fp(sol_smi2,df)

#====================================================================================

#====================================================================================
#=============================physicochemical properties=============================
#====================================================================================

sub_pp_temp = smi_pp(sub_smi,df)
precat_pp_temp = smi_pp(precat_smi,df)
add1_pp_temp = smi_pp(add_smi1,df)
add2_pp_temp = smi_pp(add_smi2,df)
sol1_pp_temp = smi_pp(sol_smi1,df)
sol2_pp_temp = smi_pp(sol_smi2,df)

#====================================================================================

eq1 = df.iloc[:,3]
temp = df.iloc[:,4]
eq2 = df.iloc[:,6]
eq3 = df.iloc[:,8]
vv = df.iloc[:,11:15]


ICat_fp = np.zeros((df.shape[0],61))

def map_to_intervals(value):
    if 0 <= value < 0.2:
        return [1, 0, 0, 0, 0]
    elif 0.2 <= value < 0.4:
        return [0, 1, 0, 0, 0]
    elif 0.4 <= value < 0.6:
        return [0, 0, 1, 0, 0]
    elif 0.6 <= value < 0.8:
        return [0, 0, 0, 1, 0]
    elif 0.8 <= value <= 1:
        return [0, 0, 0, 0, 1]
    else:
        return None

for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_NH_group(precat_smi[i])
    ICat_fp[i,0]= cfp_temp

for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.is_molecule_symmetric(precat_smi[i])
    ICat_fp[i,1]= cfp_temp

for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.contains_fluorine_element(precat_smi[i])
    ICat_fp[i,2]= cfp_temp

for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_trifluoromethyl(precat_smi[i])
    ICat_fp[i,3]= cfp_temp


template_molecule1 = "IC1=C(O[C@@H](C)C(NC2=C(C)C=C(C)C=C2C)=O)C=CC=C1O[C@@H](C)C(NC3=C(C)C=C(C)C=C3C)=O"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.calculate_similarity(template_molecule1, precat_smi[i])
    cfp_temp2 = map_to_intervals(cfp_temp)
    ICat_fp[i,4:9]= cfp_temp2

template_molecule2 = "IC1=C(O[C@@H](C)CNC(C2=C(C)C=C(C)C=C2C)=O)C=CC=C1O[C@@H](C)CNC(C3=C(C)C=C(C)C=C3C)=O"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.calculate_similarity(template_molecule2, precat_smi[i])
    cfp_temp2 = map_to_intervals(cfp_temp)
    ICat_fp[i,9:14]= cfp_temp2

template_molecule3 = "IC1=C(O[C@H](C)C(NC2=C(C)C=C(C)C=C2C)=O)C=CC=C1O[C@H](C)C(NC3=C(C)C=C(C)C=C3C)=O"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.calculate_similarity(template_molecule3, precat_smi[i])
    cfp_temp2 = map_to_intervals(cfp_temp)
    ICat_fp[i,14:19]= cfp_temp2

template_molecule4 = "CO[C@@H](C1=CN(CC2=CC=CC=C2)N=N1)C3=CC=CC=C3I"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.calculate_similarity(template_molecule4, precat_smi[i])
    cfp_temp2 = map_to_intervals(cfp_temp)
    ICat_fp[i,19:24]= cfp_temp2

template_molecule5 = "IC1=C2C(CCCC23O[C@@H](C(NC4=C(C)C=C(C)C=C4C)=O)[C@H](C(NC5=C(C)C=C(C)C=C5C)=O)O3)=CC=C1"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.calculate_similarity(template_molecule5, precat_smi[i])
    cfp_temp2 = map_to_intervals(cfp_temp)
    ICat_fp[i,24:29]= cfp_temp2

template_molecule6 = "COC1=CC=C(CC(C)(C)[C@@H]2OCC3=CC=CC=C3)C2=C1I"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.calculate_similarity(template_molecule6, precat_smi[i])
    cfp_temp2 = map_to_intervals(cfp_temp)
    ICat_fp[i,29:34]= cfp_temp2

template_molecule7 = "IC1=C2C(CC(C)(C)[C@@H]2OCC3=CC=CC=C3)=CC=C1OCC4=CC=CC=C4"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.calculate_similarity(template_molecule7, precat_smi[i])
    cfp_temp2 = map_to_intervals(cfp_temp)
    ICat_fp[i,34:39]= cfp_temp2

template_molecule8 = "IC1=C2C(CC(C)(C)[C@@H]2OCC3=CC=CC=C3)=CC=C1O[C@@H](C)C(NC4=C(C)C=C(C)C=C4C)=O"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.calculate_similarity(template_molecule8, precat_smi[i])
    cfp_temp2 = map_to_intervals(cfp_temp)
    ICat_fp[i,39:44]= cfp_temp2

template_molecule9 = "IC1=C([C@@H](O[Si](C(C)C)(C(C)C)C(C)C)C2=CN(CC3=CC=CC=C3)N=N2)C=CC=C1OC"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.calculate_similarity(template_molecule9, precat_smi[i])
    cfp_temp2 = map_to_intervals(cfp_temp)
    ICat_fp[i,44:49]= cfp_temp2


molecular_fragment1 = "IC1=C(OC)C=CC=C1OC"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_fragment(precat_smi[i], molecular_fragment1)
    ICat_fp[i,49]= cfp_temp

molecular_fragment2 = "O=CCC"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_fragment(precat_smi[i], molecular_fragment2)
    ICat_fp[i,50]= cfp_temp

molecular_fragment3 = "O=CNC1=CC=CC=C1"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_fragment(precat_smi[i], molecular_fragment3)
    ICat_fp[i,51]= cfp_temp

molecular_fragment4 = "IC1=C(O[C@@H](C)C=O)C=CC=C1O[C@@H](C)C=O"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_fragment(precat_smi[i], molecular_fragment4)
    ICat_fp[i,52]= cfp_temp

molecular_fragment5 = "CCCNC=O"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_fragment(precat_smi[i], molecular_fragment5)
    ICat_fp[i,53]= cfp_temp

molecular_fragment6 = "IC1=C(O[C@@H](C)CNC=O)C=CC=C1O[C@@H](C)CNC=O"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_fragment(precat_smi[i], molecular_fragment6)
    ICat_fp[i,54]= cfp_temp

molecular_fragment7 = "CNC(C1=C(C)C=C(C)C=C1C)=O"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_fragment(precat_smi[i], molecular_fragment7)
    ICat_fp[i,55]= cfp_temp

molecular_fragment8 = "O=CNC1=C(C)C=C(C)C=C1C"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_fragment(precat_smi[i], molecular_fragment8)
    ICat_fp[i,56]= cfp_temp

molecular_fragment9 = "CO[C@@H](C1=CN(C)N=N1)C2=CC=CC=C2I"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_fragment(precat_smi[i], molecular_fragment9)
    ICat_fp[i,57]= cfp_temp

molecular_fragment10 = "CN1N=NC([C@H](O[Si](C(C)C)(C(C)C)C(C)C)C2=CC=CC=C2I)=C1"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_fragment(precat_smi[i], molecular_fragment10)
    ICat_fp[i,58]= cfp_temp

molecular_fragment11 = "CC1(O[C@@H](C(NC2=C(C)C=C(C)C=C2C)=O)[C@H](C(NC3=C(C)C=C(C)C=C3C)=O)O1)C4=C(I)C=CC=C4"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_fragment(precat_smi[i], molecular_fragment11)
    ICat_fp[i,59]= cfp_temp

molecular_fragment12 = "IC1=C([C@@H](O[Si](C(C)C)(C(C)C)C(C)C)C2=CN(CC3=CC=CC=C3)N=N2)C=CC=C1"
for i in range(0,df.shape[0]):
    cfp_temp = ICat_fp_cal.molecule_contains_fragment(precat_smi[i], molecular_fragment12)
    ICat_fp[i,60]= cfp_temp

ICat_fp = pd.DataFrame(ICat_fp)


# 假设你有一个字典 "matrices" 存储了7个矩阵，例如：
fp_data = {
    'eq1':eq1,
    'temp':temp,
    'eq2':eq2,
    'eq3':eq3,
    'vv':vv,
    'sub': sub_fp_temp,
    #'submake':Sub_UDF_fp,
    'precat': precat_fp_temp,
    #'precat2': ICat_fp,
    'add1': add1_fp_temp,
    'add2': add2_fp_temp,
    'sol1': sol1_fp_temp,
    'sol2': sol2_fp_temp,
}

pp_data = {
    'eq1':eq1,
    'temp':temp,
    'eq2':eq2,
    'eq3':eq3,
    'vv':vv,
    'sub': sub_pp_temp,
    #'submake':Sub_UDF_fp,
    'precat2': precat_pp_temp,
    #'precat': ICat_fp,
    'add1': add1_pp_temp,
    'add2': add2_pp_temp,
    'sol1': sol1_pp_temp,
    'sol2': sol2_pp_temp,
}

fp_pp_data = {
    'eq1':eq1,
    'temp':temp,
    'eq2':eq2,
    'eq3':eq3,
    'vv':vv,
    'sub': sub_fp_temp,
    'precat': precat_fp_temp,
    'add1': add1_fp_temp,
    'add2': add2_fp_temp,
    'sol1': sol1_fp_temp,
    'sol2': sol2_fp_temp,
    'sub2': sub_pp_temp,
    'precat2': precat_pp_temp,
    'add12': add1_pp_temp,
    'add22': add2_pp_temp,
    'sol12': sol1_pp_temp,
    'sol22': sol2_pp_temp,
}



yield_data = df.iloc[:,15]
ee_data = df.iloc[:,16]

features_temp = pd.DataFrame(pd.concat(fp_pp_data, axis=1))

features_temp2, fmean, fstd_dev = sdata(features_temp)
features,targets = ANN_reg.preprocess_data(pd.DataFrame(features_temp2),pd.DataFrame(ee_data))

#print(targets.size())

#'''
# 创建Optuna Study对象并执行超参数优化
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: ANN_reg.objective(trial, features, targets), n_trials=100, n_jobs=-1)

# 打印优化的最佳超参数
print("Best Hyperparameters:")
print(study.best_params)

# 最佳R2
best_rmse = study.best_value

# 打印最佳RMSE
print(f"Best R2: {best_rmse:.4f}")
#'''

best_trial = study.best_trial
best_hidden_size = best_trial.params['hidden_layers_option']
best_lr = best_trial.params['learning_rate']

if best_hidden_size == 'option1':
    hidden_layers = [64, 32]
elif best_hidden_size == 'option2':
    hidden_layers = [128, 64, 32]
elif best_hidden_size == 'option3':
    hidden_layers = [256, 128, 64]
elif best_hidden_size == 'option4':
    hidden_layers = [256, 128, 64, 32]
elif best_hidden_size == 'option5':
    hidden_layers = [256, 128, 64, 32, 16]
elif best_hidden_size == 'option6':
    hidden_layers = [512, 256, 128, 64, 32, 16]

num_epochs = 200

ANN_reg.train_ann_regression(features, targets, hidden_layers, best_lr, num_epochs)