from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def decision_tree_regression(X_train, X_test, y_train, y_test):
    # 划分训练集和测试集

    
    # 创建决策树回归模型
    model = DecisionTreeRegressor(random_state=42)

    param_grid = {
        'max_depth': [3, 5, 7, 20, None],
        'min_samples_split': [2, 5, 10, None]
    }


    # 使用GridSearchCV进行超参数优化
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    best_rf_regressor = grid_search.fit(X_train, y_train)

    y_train_pred = best_rf_regressor.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # 输出最佳超参数和对应的得分
    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best Score: ", -grid_search.best_score_)

    # 使用最佳超参数的模型进行预测
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    
    # 计算R2得分
    r2 = r2_score(y_test, predictions)
    
    # 计算均方根误差（RMSE）
    rmse = mean_squared_error(y_test, predictions, squared=False)
    
    print(f"==========================================================================")
    print('训练集R2:', r2_train)
    print('训练集RMSE:', rmse_train)
    print(f"模型的准确性为：{r2}")
    print(f"模型的均方根误差为：{rmse}")
    print(f"==========================================================================")

    return predictions, r2, rmse


