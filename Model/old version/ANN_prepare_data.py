import numpy as np
import pandas as pd

from descriptors import rdkit_descriptor as rd_ds
from descriptors import rdkit_fingerprint as rd_fp

from Model import Randomforest_Regression as rf
from Model import Xgboost_Regression as XGB
from Model import Adaboost_Regression as AdaB
from Model import KNN_Regression as KNN
from Model import SVM_Regression as SVM
from Model import Linear_Regression as Linear
from Model import Decisiontree_Regression as DT
from Model import LightGBM as lgbm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


file_path = "/mnt/d/PythonCode/ICat/Dataset/ICatDataset.xlsx"
sheet_name = "Sheet1"

df = pd.read_excel(file_path, sheet_name=sheet_name)

sub_smi = df.iloc[:,1]
precat_smi = df.iloc[:,2]

add_smi1 = df.iloc[:,5]
add_smi2 = df.iloc[:,7]

sol_smi1 = df.iloc[:,9]
sol_smi2 = df.iloc[:,10]


'''

for i in range(0,df.shape[0]):
    a = 1
    cfp_temp = rd_fp(sub_smi, fp_type="MACCSKeys")

'''

'''
#---------------------------------------------------------
rd_temp = []  # 创建一个空列表

for i in range(0,df.shape[0]):
    crd_ds = rd_ds(sub_smi[i])
    rd_temp.append(crd_ds)  # 使用 append() 按行添加list

sub_rd = pd.DataFrame(rd_temp) 
sub_rd = sub_rd.fillna(0)  # 空值填充0

del rd_temp

#---------------------------------------------------------
rd_temp = []  # 创建一个空列表

for i in range(0,df.shape[0]):
    crd_ds = rd_ds(precat_smi[i])
    rd_temp.append(crd_ds)  # 使用 append() 按行添加list

precat_rd = pd.DataFrame(rd_temp) 
precat_rd = precat_rd.fillna(0)  # 空值填充0

del rd_temp

#---------------------------------------------------------
rd_temp = []  # 创建一个空列表

for i in range(0,df.shape[0]):
    crd_ds = rd_ds(add_smi1[i])
    rd_temp.append(crd_ds)  # 使用 append() 按行添加list

add1_rd = pd.DataFrame(rd_temp) 
add1_rd = add1_rd.fillna(0)  # 空值填充0

del rd_temp

#---------------------------------------------------------
rd_temp = []  # 创建一个空列表

for i in range(0,df.shape[0]):
    crd_ds = rd_ds(add_smi2[i])
    rd_temp.append(crd_ds)  # 使用 append() 按行添加list

add2_rd = pd.DataFrame(rd_temp) 
add2_rd = add2_rd.fillna(0)  # 空值填充0

del rd_temp

#---------------------------------------------------------
rd_temp = []  # 创建一个空列表

for i in range(0,df.shape[0]):
    crd_ds = rd_ds(sol_smi1[i])
    rd_temp.append(crd_ds)  # 使用 append() 按行添加list

sol1_rd = pd.DataFrame(rd_temp) 
sol1_rd = sol1_rd.fillna(0)  # 空值填充0

del rd_temp

#---------------------------------------------------------
rd_temp = []  # 创建一个空列表

for i in range(0,df.shape[0]):
    crd_ds = rd_ds(sol_smi2[i])
    rd_temp.append(crd_ds)  # 使用 append() 按行添加list

sol2_rd = pd.DataFrame(rd_temp) 
sol2_rd = sol2_rd.fillna(0)  # 空值填充0

del rd_temp

#---------------------------------------------------------
# 组合数据集
df_temp1 = df.iloc[:,3]
df_temp2 = df.iloc[:,4]
df_temp3 = df.iloc[:,6]
df_temp4 = df.iloc[:,8]
df_temp5 = df.iloc[:,11:15]

df_temp6 = pd.concat([df_temp1, df_temp2], axis=1, ignore_index=True)
df_temp7 = pd.concat([df_temp6, df_temp3], axis=1, ignore_index=True)
df_temp8 = pd.concat([df_temp7, df_temp4], axis=1, ignore_index=True)
df_temp9 = pd.concat([df_temp8, df_temp5], axis=1, ignore_index=True)

df_temp10 = pd.concat([df_temp9, sub_rd], axis=1, ignore_index=True)
df_temp11 = pd.concat([df_temp10, precat_rd], axis=1, ignore_index=True)
df_temp12 = pd.concat([df_temp11, add1_rd], axis=1, ignore_index=True)
df_temp13 = pd.concat([df_temp12, add2_rd], axis=1, ignore_index=True)
df_temp14 = pd.concat([df_temp13, sol1_rd], axis=1, ignore_index=True)
df_temp15 = pd.concat([df_temp14, sol2_rd], axis=1, ignore_index=True)
'''

def smi_fp(smi):
    fp_temp = np.zeros((df.shape[0],167))  # 创建一个空列表
    for i in range(0,df.shape[0]):
        cfp_temp = rd_fp(smi[i], fp_type="MACCSKeys")
        for j in range(0,167):
            fp_temp[i][j] = cfp_temp[j]
    fp_temp = pd.DataFrame(fp_temp)
    return fp_temp

sub_fp_temp = smi_fp(sub_smi)
precat_fp_temp = smi_fp(precat_smi)
add1_fp_temp = smi_fp(add_smi1)
add2_fp_temp = smi_fp(add_smi2)
sol1_fp_temp = smi_fp(sol_smi1)
sol2_fp_temp = smi_fp(sol_smi2)

eq1 = df.iloc[:,3]
temp = df.iloc[:,4]
eq2 = df.iloc[:,6]
eq3 = df.iloc[:,8]
vv = df.iloc[:,11:15]

# 假设你有一个字典 "matrices" 存储了7个矩阵，例如：
matrices = {
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
}

features = pd.concat(matrices, axis=1)
# 在 combined_matrices 列表中有所有两两组合后的矩阵

yield_data = df.iloc[:,15]
ee_data = df.iloc[:,16]

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


def train_neural_network(train_loader, test_loader, input_size, hidden_layers, num_epochs=100, learning_rate=0.010223060090044686):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNetwork(input_size, hidden_layers, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # 计算训练集上的 RMSE 和 R2
        model.eval()
        train_preds = []
        train_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in train_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                outputs = model(batch_features)
                train_preds.extend(outputs.cpu().numpy())
                train_targets.extend(batch_targets.cpu().numpy())
        
        train_rmse = sqrt(mean_squared_error(train_targets, train_preds))
        train_r2 = r2_score(train_targets, train_preds)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}")
    
    model.eval()
    test_loss = 0
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets.unsqueeze(1))
            test_loss += loss.item()
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(batch_targets.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    test_rmse = sqrt(mean_squared_error(test_targets, test_preds))
    test_r2 = r2_score(test_targets, test_preds)
    
    print(f"Test Loss: {avg_test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}")
    
    return model

# 定义自定义数据处理函数
def prepare_data(features, targets, test_size=0.2, random_state=42, batch_size=128):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    class CustomDataset(Dataset):
        def __init__(self, features, targets):
            self.features = features
            self.targets = targets

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.targets[idx]

    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    test_dataset = CustomDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(NeuralNetwork, self).__init__()
        layers = [input_size] + hidden_layers + [output_size]
        self.fc_layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.fc_layers[:-1]:
            x = layer(x)
            x = self.relu(x)
        x = self.fc_layers[-1](x)
        return x

features = features
targets = ee_data

train_loader, test_loader = prepare_data(features, targets)

# 获取特征数量
input_size = features.shape[1]
# 自定义隐藏层层数和每层神经元数量
hidden_layers = [256, 128, 32]  # 例如，这里定义三个隐藏层分别为128、64和32个神经元

# 训练神经网络模型
trained_model = train_neural_network(train_loader, test_loader, input_size, hidden_layers)

'''
# 读取数据集
data_path = 'your_dataset.csv'
data = pd.read_csv(data_path)

# 提取特征和目标
features = data.drop(data.columns[-1], axis=1)  # 假设最后一列是目标列
targets = data[data.columns[-1]]  # 假设最后一列是目标列

# 调用数据处理函数并获取数据加载器
train_loader, test_loader = prepare_data(features, targets)

# 获取特征数量
input_size = features.shape[1]
# 自定义隐藏层层数和每层神经元数量
hidden_layers = [64, 32]  # 例如，这里定义两个隐藏层分别为64和32个神经元

# 训练神经网络模型
trained_model = train_neural_network(train_loader, test_loader, input_size, hidden_layers)


# 在这之后，您可以使用train_loader和test_loader来迭代数据进行训练和测试。
# 每个batch会包含特征和目标，可以直接传入PyTorch模型中进行训练和预测。
'''