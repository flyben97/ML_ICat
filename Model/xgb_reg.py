import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

def xgboost_regression_optimization(X_train, y_train, X_test, y_test, feature_name):
    '''   
    params = {
        'n_estimators':  800, 
        'learning_rate':  0.035721,
        'max_depth': 12, 
        'min_child_weight':  9.677133, 
        'gamma': 0.858624, 
        'subsample':  0.679918, 
        'colsample_bytree': 0.414259, 
    }
    params = {
        'n_estimators':  50, 
        'learning_rate':  0.13418273566601466,
        'max_depth': 1, 
        'min_child_weight':  2.638843535216844, 
        'gamma': 22.784619952418645, 
        'subsample':  0.69760881, 
        'colsample_bytree': 0.8251507366635625, 
    }

    params = {
        'n_estimators':  600, 
        'learning_rate':  0.064774,
        'max_depth': 6, 
        'min_child_weight':  0.00423013, 
        'gamma': 59.377322, 
        'subsample':  0.841406, 
        'colsample_bytree': 0.307238, 
    }
   
    params = {
        'n_estimators':  500, 
        'learning_rate':  0.039517,
        'max_depth': 12, 
        'min_child_weight':  0.0000416479, 
        'gamma': 0.025734, 
        'subsample':  0.59953, 
        'colsample_bytree': 0.290323, 
    }

    '''
    params = {
        'n_estimators':  300, 
        'learning_rate': 0.034077577867180235,
        'max_depth': 12, 
        'min_child_weight':  0.00020427607003169405, 
        'gamma': 0.0068555569725657585, 
        'subsample':  0.38807507004052255, 
        'colsample_bytree': 0.6075962191609715, 
    }
     
    
   



    # 创建XGBoost回归器
    model = xgb.XGBRegressor(**params, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 使用最佳超参数在训练集上计算 R² 和 RMSE
    train_pred = model.predict(X_train)

    train_r2 = r2_score(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))


    print(f'R² on Training Set with Best Parameters: {train_r2}')
    print(f'RMSE on Training Set with Best Parameters: {train_rmse}')


    # 计算最终模型的 R²
    final_r2 = r2_score(y_test, y_pred)
    print(f'Final R² on Test Set: {final_r2}')
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE on Test Set: {rmse}')

    
    # 获取特征重要性
    feature_importance = model.feature_importances_

    # 创建一个包含特征名和对应重要性分数的DataFrame
    feature_importance_df = pd.DataFrame({'Feature': feature_name.columns, 'Importance': feature_importance})

    # 按特征重要性分数降序排序
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    
    model.save_model('/mnt/c/Ben_workspace/PythonCode/ICat/Model/ICat_xgb2.model')



    return final_r2, rmse, train_pred, y_pred,  feature_importance_df



