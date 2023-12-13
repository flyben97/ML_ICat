import optuna
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

def linear_regression_optimization(X_train, y_train, X_test, y_test):
    # 创建线性回归模型
    regressor = LinearRegression()

    # 训练线性回归模型
    regressor.fit(X_train, y_train)

    # 预测测试集
    y_pred = regressor.predict(X_test)

    # 计算测试集上的R²
    final_r2 = r2_score(y_test, y_pred)
    print(f'Final R² on Test Set: {final_r2}')

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE on Test Set: {rmse}')

    # 计算训练集上的R²和RMSE
    train_pred = regressor.predict(X_train)
    train_r2 = r2_score(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    print(f'R² on Training Set: {train_r2}')
    print(f'RMSE on Training Set: {train_rmse}')

if __name__ == '__main__':
    from sklearn.datasets import load_boston

    # 加载示例数据集（波士顿房价数据集）
    boston = load_boston()
    X = boston.data
    y = boston.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 运行线性回归算法
    linear_regression_optimization(X_train, y_train, X_test, y_test)
