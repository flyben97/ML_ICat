from descriptors import rdkit_fingerprint as rd_fp
import numpy as np
import pandas as pd

def smi_fp(smi, df):
    fp_temp = np.zeros((df.shape[0],167))  # 创建一个空列表
    for i in range(0,df.shape[0]):
        cfp_temp = rd_fp(smi[i], fp_type="MACCSKeys")
        for j in range(0,167):
            fp_temp[i][j] = cfp_temp[j]
    fp_temp = pd.DataFrame(fp_temp)
    return fp_temp