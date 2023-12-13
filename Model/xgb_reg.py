import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

def xgboost_regression_optimization(X_train, y_train, X_test, y_test, feature_name):
      
# detaG
    '''   
    # target: ee values
    params = {
        'n_estimators': 500,
        'learning_rate': 0.054304151188639314,
        'max_depth': 11, 
        'min_child_weight': 2.9994614406927522, 
        'gamma': 0.0014219012957643747, 
        'subsample': 0.897689757391236, 
        'colsample_bytree': 0.3753320973743237
    }
    '''
     # target: detaG values

    params = {

        'n_estimators': 550, 
        'learning_rate': 0.12222696963658544, 
        'max_depth': 9, 
        'min_child_weight': 5.50087269554148, 
        'gamma': 0.002397046948904361, 
        'subsample': 0.9944840879115209, 
        'colsample_bytree': 0.4359655012429955
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
    #'''
    # 创建一个包含特征名和对应重要性分数的DataFrame
    feature_importance_df = pd.DataFrame({'Feature': feature_name.columns, 'Importance': feature_importance})

    # 按特征重要性分数降序排序
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    #'''
    model.save_model('/mnt/c/Ben_workspace/PythonCode/ICat/Model/ICat_xgb1205_detaG.model')

   

    return final_r2, rmse, train_pred, y_pred,  feature_importance_df , y_train, y_test



