from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def linear_regression(X_train, X_test, y_train, y_test):
    # 创建线性回归模型
    model = LinearRegression(n_jobs=-1)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测训练集和测试集
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算R2和RMSE
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print('训练集R2:', r2_train)
    print('训练集RMSE:', rmse_train)
    print('测试集R2:', r2_test)
    print('测试集RMSE:', rmse_test)
    print('线性回归系数:', model.coef_)
    print('线性模型的截距:', model.intercept_)

    
    return y_test_pred, r2_test, rmse_test



# 调用线性回归函数
#weights, intercept, y_train_pred, y_test_pred, r2_train, rmse_train, r2_test, rmse_test = linear_regression(X_train, X_test, y_train, y_test)











