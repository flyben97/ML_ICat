import optuna
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

def decision_tree_regression_optimization(X_train, y_train, X_test, y_test, n_trials=100):

    def objective(trial):
        
        max_depth = trial.suggest_int('max_depth', 1, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)

        regressor = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )

        r2_values = cross_val_score(regressor, X_train, y_train, cv=5, scoring='r2')


        ave_r2 = np.mean(r2_values)

        return ave_r2

    study = optuna.create_study(direction='maximize')

    study.optimize(objective, n_trials=n_trials, n_jobs=-1)  # n_jobs=-1可加速并行优化


    best_params = study.best_trial.params
    ave_r2 = study.best_value

    best_regressor = DecisionTreeRegressor(
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42
    )
    best_regressor.fit(X_train, y_train)


    y_train_pred = best_regressor.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))


    y_test_pred = best_regressor.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return best_params, ave_r2, y_train, y_train_pred, y_test, y_test_pred, train_r2, test_r2, train_rmse, test_rmse
