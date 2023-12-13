import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def adaboost_regression(X_train, X_test, y_train, y_test):
    # 创建一个AdaBoost回归器对象
    ada_boost = AdaBoostRegressor(random_state=42)

    # 定义要搜索的超参数范围
    param_grid = {
        'n_estimators': [50, 100, 200, 300],  # 弱学习器数量
        'learning_rate': [0.01, 0.1, 1.0],  # 学习率
    }

    # 创建网格搜索对象，传入AdaBoost回归器和超参数范围
    grid_search = GridSearchCV(ada_boost, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

    # 在训练集上进行网格搜索，寻找最佳超参数组合
    best_regressor= grid_search.fit(X_train, y_train)

    y_train_pred = best_regressor.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    print(f"==========================================================================")
    print('训练集R2:', r2_train)
    print('训练集RMSE:', rmse_train)
    
    # 输出最佳超参数组合和对应的评分
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", -grid_search.best_score_)

    # 使用最佳超参数训练最终模型
    best_ada_boost = grid_search.best_estimator_
    best_ada_boost.fit(X_train, y_train)

    # 进行预测
    y_pred = best_ada_boost.predict(X_test)

    # 计算R2得分
    r2 = r2_score(y_test, best_ada_boost.predict(X_test))

    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_test, best_ada_boost.predict(X_test)))

    print(f"==========================================================================")
    print(f"模型的准确性为：{r2}")
    print(f"模型的均方根误差为：{rmse}")
    print(f"==========================================================================")

    return y_pred, r2, rmse


