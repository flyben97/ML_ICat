import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def knn_regression(X_train, X_test, y_train, y_test):
    """
    使用KNN算法进行回归任务，并进行超参数优化
    
    参数:
    X: 特征矩阵
    y: 目标向量
    test_size: 测试集比例，默认为0.2
    param_grid: 超参数搜索空间，默认为None
    cv: 交叉验证的折数，默认为5
    
    返回:
    测试集上的均方误差和R2分数
    """
    # 定义超参数的搜索空间
    param_grid = {
    'n_neighbors': [3, 5, 7],  # 最近邻数
    'weights': ['uniform', 'distance'],  # 权重函数
    'p': [1, 2]  # 距离度量
    }
    
    # 定义KNN回归器
    knn = KNeighborsRegressor()
    
    # 使用GridSearchCV进行超参数优化
    grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    y_train_pred = grid_search.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))


    # 在测试集上进行预测
    y_pred = grid_search.predict(X_test)
    
    # 计算均方误差
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # 计算R2分数
    r2 = r2_score(y_test, y_pred)

    print(f"==========================================================================")
    print("最佳参数组合: ", grid_search.best_params_)
    print('训练集R2:', r2_train)
    print('训练集RMSE:', rmse_train)
    print(f"模型的准确性为：{r2}")
    print(f"模型的均方根误差为：{rmse}")
    print(f"==========================================================================")

    return y_pred, r2, rmse


