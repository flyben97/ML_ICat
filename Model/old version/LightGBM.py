from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm.sklearn import LGBMRegressor
import numpy as np

def lightGBM_reg(X_train, X_test, y_train, y_test):
    # 定义要优化的超参数和它们的取值范围
    param_grid = {
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.01,0.1,0.2],
    'max_depth': [3, 4, 5,None],
    'min_child_samples': [10, 20, 30,None]
}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 五折交叉验证
    # 初始化随机森林回归器
    lgbm_model = LGBMRegressor(random_state=42)
    # 初始化随机搜索器
    grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf, verbose=0, n_jobs=-1)
    # 训练随机搜索器
    grid_search.fit(X_train, y_train)

    # 获取最佳超参数配置
    best_params = grid_search.best_params_

    # 使用最佳超参数配置训练最终模型
    best_lgbm_model = LGBMRegressor(**best_params)
    best_lgbm_model.fit(X_train, y_train)

    # 输出最佳超参数
    print(f"==========================================================================")
    print(f"最佳超参数为：{best_params}")


    y_train_pred = best_lgbm_model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # 预测测试集的结果
    y_pred = best_lgbm_model.predict(X_test)

    importances = best_lgbm_model.feature_importances_

    # 计算模型的准确性
    r2 = best_lgbm_model.score(X_test, y_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"==========================================================================")
    print('训练集R2:', r2_train)
    print('训练集RMSE:', rmse_train)
    print(f"模型的准确性为：{r2}")
    print(f"模型的均方根误差为：{rmse}")
    print(f"==========================================================================")

    return y_pred, r2, rmse, importances
