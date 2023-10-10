import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

def preprocess_data(features, targets):
    def process_input(input_data):
      
        if isinstance(input_data, np.ndarray):
            if len(input_data.shape) == 1:
    
                return torch.tensor(input_data.reshape(-1, 1), dtype=torch.float32)
            else:
                return torch.tensor(input_data, dtype=torch.float32)
        elif isinstance(input_data, pd.DataFrame):
            data_as_numpy = input_data.to_numpy()
            if len(data_as_numpy.shape) == 1:

                return torch.tensor(data_as_numpy.reshape(-1, 1), dtype=torch.float32)
            else:
                return torch.tensor(data_as_numpy, dtype=torch.float32)
        else:
            raise ValueError("Unsupported data type. Please provide NumPy array or DataFrame.")
        
    processed_features = process_input(features)
    processed_targets = process_input(targets)

    return processed_features, processed_targets

class ComplexANN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ComplexANN, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

def train_ann_regression(features, targets, hidden_sizes, learning_rate, num_epochs):

    train_features, test_features, train_targets, test_targets = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    input_size = features.shape[1]
    output_size = targets.shape[1]
    model = ComplexANN(input_size, hidden_sizes, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_features)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

   
    #plt.plot(losses)
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.title('Training Loss Curve')
    #lt.show()

    model.eval()
    with torch.no_grad():
        y_test_pred = model(test_features).detach().numpy()
    
    test_rmse = np.sqrt(mean_squared_error(test_targets, y_test_pred))
    test_r2 = r2_score(test_targets, y_test_pred)

    model.eval()
    with torch.no_grad():
        y_train_pred = model(train_features).detach().numpy()
    
    train_rmse = np.sqrt(mean_squared_error(train_targets, y_train_pred))
    train_r2 = r2_score(train_targets, y_train_pred)

    return model, train_targets, y_train_pred, test_targets, y_test_pred, train_r2, test_r2, train_rmse, test_rmse

def train_ann_regression_with_cv(features, targets, hidden_sizes, learning_rate=0.001, num_epochs=200):

    rmse_scores = []
    r2_scores = []


    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, val_index in kf.split(features):
        train_features, val_features = features[train_index], features[val_index]
        train_targets, val_targets = targets[train_index], targets[val_index]


        input_size = features.shape[1]
        output_size = targets.shape[1]
        model = ComplexANN(input_size, hidden_sizes, output_size)


        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


        losses = []
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(train_features)
            loss = criterion(outputs, train_targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            predicted = model(val_features).detach().numpy()

        rmse = np.sqrt(mean_squared_error(val_targets, predicted))

        r2 = r2_score(val_targets, predicted)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

    average_r2 = np.mean(r2_scores)

    return average_r2  


def objective(trial,features,targets):

    '''
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 4)
    hidden_sizes = []
    for _ in range(num_hidden_layers):
        hidden_size = trial.suggest_int("hidden_size", 32, 256)
        hidden_sizes.append(hidden_size)
    '''

    hidden_layers_option = trial.suggest_categorical('hidden_layers_option', ['option1', 'option2', 'option3','option4', 'option5', 'option6'])
    if hidden_layers_option == 'option1':
        hidden_layers = [64, 32]
    elif hidden_layers_option == 'option2':
        hidden_layers = [128, 64, 32]
    elif hidden_layers_option == 'option3':
        hidden_layers = [256, 128, 64]
    elif hidden_layers_option == 'option4':
        hidden_layers = [256, 128, 64, 32]
    elif hidden_layers_option == 'option5':
        hidden_layers = [256, 128, 64, 32, 16]
    elif hidden_layers_option == 'option6':
        hidden_layers = [512, 256, 128, 64, 32, 16]
    #'''
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
   

    average_r2 = train_ann_regression_with_cv(
        features, targets, hidden_sizes=hidden_layers, learning_rate=learning_rate, num_epochs=200
    )

    return average_r2 

def ANN_regression_optimization(features, targets):

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial:objective(trial, features, targets), n_trials=100, n_jobs=-1)

    ave_r2 = study.best_value

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

    model, train_targets, y_train_pred, test_targets, y_test_pred, train_r2, test_r2, train_rmse, test_rmse = train_ann_regression(features, targets, hidden_layers, best_lr, num_epochs)

    return best_trial, ave_r2, train_targets, y_train_pred, test_targets, y_test_pred, train_r2, test_r2, train_rmse, test_rmse
