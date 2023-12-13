from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def randomforest_reg(X_train, X_test, y_train, y_test):
    # 定义要优化的超参数和它们的取值范围
    param_dist = {
                "n_estimators": [ 100, 200, 300, 400, 500]
                ,"max_depth": [3, 7, None]
                #,"max_features": np.arange(1, 13)
                }

    # 初始化随机森林回归器
    rf_regressor = RandomForestRegressor(random_state=42)

    # 初始化随机搜索器
    random_search = RandomizedSearchCV(rf_regressor, param_distributions=param_dist, n_iter=3, cv=5, n_jobs=-1)

    # 训练随机搜索器
    random_search.fit(X_train, y_train)

    # 输出最佳超参数
    print(f"==========================================================================")
    print(f"最佳超参数为：{random_search.best_params_}")

    # 使用最佳超参数训练模型
    best_rf_regressor = random_search.best_estimator_
    best_rf_regressor.fit(X_train, y_train)

    y_train_pred = best_rf_regressor.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # 预测测试集的结果
    y_pred = best_rf_regressor.predict(X_test)

    importances = best_rf_regressor.feature_importances_

    # 计算模型的准确性
    r2 = best_rf_regressor.score(X_test, y_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"==========================================================================")
    print('训练集R2:', r2_train)
    print('训练集RMSE:', rmse_train)
    print(f"模型的准确性为：{r2}")
    print(f"模型的均方根误差为：{rmse}")
    print(f"==========================================================================")

    return y_pred, r2, rmse, importances
