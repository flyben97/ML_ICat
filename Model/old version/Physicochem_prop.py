from descriptors import rdkit_descriptor as rd_ds
import numpy as np
import pandas as pd

def smi_pp(smi, df):

    rd_temp = []  # 创建一个空列表

    for i in range(0,df.shape[0]):
        crd_ds = rd_ds(smi[i])
        rd_temp.append(crd_ds)  # 使用 append() 按行添加list

    pp_temp = pd.DataFrame(rd_temp) 
    pp_temp = pp_temp.fillna(0)  # 空值填充0

    return pp_temp