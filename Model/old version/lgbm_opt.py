import optuna
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

def lightgbm_regression_optimization(X_train, y_train, X_test, y_test, n_trials=100):
    def objective(trial):
        # 定义LightGBM的超参数候选范围
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200, 300]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'num_leaves': trial.suggest_int('num_leaves', 2, 500),
            'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        }

        # 创建LightGBM回归器
        regressor = lgb.LGBMRegressor(**params, random_state=42)

        # 使用Scikit-Learn的交叉验证计算R²
        r2_values = cross_val_score(regressor, X_train, y_train, cv=5, scoring='r2')

        # 计算五折交叉验证的平均R²
        r2 = np.mean(r2_values)

        return r2

    # 创建Optuna Study对象，设置优化目标为最大化R²
    study = optuna.create_study(direction='maximize')

    # 运行超参数优化
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)  # n_jobs=-1可加速并行优化

    # 打印最佳超参数组合和对应的最大R²
    print(f'Best trial: {study.best_trial.params}')
    print(f'Best R²: {study.best_value}')

    # 使用最佳超参数训练最终模型
    best_params = study.best_trial.params
    best_regressor = lgb.LGBMRegressor(**best_params, random_state=42)
    best_regressor.fit(X_train, y_train)

    # 预测测试集
    y_pred = best_regressor.predict(X_test)

    # 计算最终模型的 R²
    final_r2 = r2_score(y_test, y_pred)
    print(f'Final R² on Test Set: {final_r2}')

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE on Test Set: {rmse}')

    # 使用最佳超参数在训练集上计算 R² 和 RMSE
    train_pred = best_regressor.predict(X_train)
    train_r2 = r2_score(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    print(f'R² on Training Set with Best Parameters: {train_r2}')
    print(f'RMSE on Training Set with Best Parameters: {train_rmse}')

if __name__ == '__main__':
    from sklearn.datasets import load_boston

    # 加载示例数据集（波士顿房价数据集）
    boston = load_boston()
    X = boston.data
    y = boston.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 运行LightGBM回归算法和超参数优化
    lightgbm_regression_optimization(X_train, y_train, X_test, y_test, n_trials=100)
