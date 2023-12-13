import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def svm_regression(X_train, X_test, y_train, y_test):

    # 定义参数网格
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'rbf', 'sigmoid']}

    # 创建SVM回归模型
    svm = SVR()

    # 使用网格搜索进行超参数优化
    grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 打印最佳参数组合
    print("最佳参数组合: ", grid_search.best_params_)

    y_train_pred = grid_search.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # 在测试集上进行预测
    y_pred = grid_search.predict(X_test)

    # 计算均方误差和决定系数
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 打印评估指标
    print('训练集R2:', r2_train)
    print('训练集RMSE:', rmse_train)
    print("决定系数 (R2): ", r2)
    print("均方误差 (RMSE): ", rmse)
    
    
    return y_pred, r2, rmse
