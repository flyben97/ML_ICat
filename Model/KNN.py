import optuna
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

def knn_regression_optimization(X_train, y_train, X_test, y_test, n_trials=100):

    def objective(trial):

        n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])

        regressor = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)

        r2_values = cross_val_score(regressor, X_train, y_train, cv=5, scoring='r2')

        # 计算五折交叉验证的平均R²
        ave_r2 = np.mean(r2_values)

        return ave_r2

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)  

    best_params = study.best_trial.params
    ave_r2 = study.best_value


    best_regressor = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
    best_regressor.fit(X_train, y_train)

    y_train_pred = best_regressor.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))


    y_test_pred = best_regressor.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return best_params, ave_r2, y_train, y_train_pred, y_test, y_test_pred, train_r2, test_r2, train_rmse, test_rmse

