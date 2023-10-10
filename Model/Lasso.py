import optuna
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

def optimize_lasso_hyperparameters(X_train, y_train, X_test, y_test, n_trials=100):

    def objective(trial):
        
        alpha = trial.suggest_float('alpha', 1e-5, 1, log=True)  # 使用log=True来表示对数均匀分布

        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)


        r2 = cross_val_score(lasso, X_train, y_train, cv=5, scoring='r2')

        ave_r2 = np.mean(r2)

        return ave_r2


    study = optuna.create_study(direction='maximize')  # 优化目标是最小化均方误差

    study.optimize(objective, n_trials=n_trials, n_jobs=-1)  # 您可以设置n_trials来控制搜索的次数

    # 获取最佳参数配置和最低的均方误差
    best_params = study.best_params
    ave_r2 = study.best_value

    lasso = Lasso(**best_params)
    lasso.fit(X_train, y_train)

    y_train_pred = lasso.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    y_test_pred = lasso.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return best_params, ave_r2, y_train, y_train_pred, y_test, y_test_pred, train_r2, test_r2, train_rmse, test_rmse


   







