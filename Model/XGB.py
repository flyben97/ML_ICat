import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

def xgboost_regression_optimization(X_train, y_train, X_test, y_test, n_trials=100):
    def objective(trial):

        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [300, 350, 400, 450, 500, 550, 600, 800]),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e5, log=True),
            'gamma': trial.suggest_float('gamma', 1e-5, 1e5, log=True),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        }


        regressor = xgb.XGBRegressor(**params, random_state=42)

        r2_values = cross_val_score(regressor, X_train, y_train, cv=5, scoring='r2')

        ave_r2 = np.mean(r2_values)

        return ave_r2

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)  

    best_params = study.best_trial.params
    ave_r2 = study.best_value

    best_regressor = xgb.XGBRegressor(**best_params, random_state=42)
    best_regressor.fit(X_train, y_train)

    y_train_pred = best_regressor.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))


    y_test_pred = best_regressor.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return best_params, ave_r2, y_train, y_train_pred, y_test, y_test_pred, train_r2, test_r2, train_rmse, test_rmse
