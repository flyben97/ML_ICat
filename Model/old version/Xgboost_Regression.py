import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def Xgboost_reg(X_train, X_test, y_train, y_test):
    # 定义XGB回归器的参数空间
    params_space = {
        "learning_rate": [0.05],
        "max_depth": [ 7],
        "n_estimators": [200, 300, 400],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.6, 0.8, 1],
        'reg_alpha': [ 0.01, 0.1,],
        'reg_lambda': [ 0.01, 0.1],
        #'gamma': [0, 0.1, 0.2, 0.3, 0.4]
    }

    # 初始化XGB回归器
    xgb_regressor = xgb.XGBRegressor(objective="reg:squarederror", seed=42)

    # 初始化网格搜索器
    grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=params_space, cv=5, n_jobs=-1)

    # 训练模型
    best_regressor = grid_search.fit(X_train, y_train)

    y_train_pred = best_regressor.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))



    # 输出最优参数和得分
    print(f"最优参数为：{grid_search.best_params_}")

    print(f"==========================================================================")
    print('训练集R2:', r2_train)
    print('训练集RMSE:', rmse_train)
    print(f"最优得分为：{grid_search.best_score_}")

    # 预测测试集的结果
    y_pred = grid_search.predict(X_test)

    # 计算模型的均方误差（MSE）
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    accuracy = grid_search.best_score_
    print(f"模型的MSE为：{rmse}")
    print(f"==========================================================================")
    return y_pred, accuracy, rmse
