import optuna
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

def svm_regression_optimization(X_train, y_train, X_test, y_test, n_trials=100):
    
    def objective(trial):
        # 定义离散的C和epsilon参数候选值
        C_values = [0.01, 0.1, 1.0, 10.0]
        epsilon_values = [0.01, 0.1, 0.5, 1.0]

        params = {
            'C': trial.suggest_categorical('C', C_values),
            'epsilon': trial.suggest_categorical('epsilon', epsilon_values),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        }

        regressor = SVR(**params)
        regressor.fit(X_train, y_train)

        r2 = cross_val_score(regressor, X_train, y_train, cv=5, scoring='r2')

        ave_r2 = np.mean(r2)

        return ave_r2

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    best_params = study.best_trial.params
    ave_r2 = study.best_value

    best_regressor = SVR(**best_params)
    best_regressor.fit(X_train, y_train)


    y_train_pred = best_regressor.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))


    y_test_pred = best_regressor.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return best_params, ave_r2, y_train, y_train_pred, y_test, y_test_pred, train_r2, test_r2, train_rmse, test_rmse
