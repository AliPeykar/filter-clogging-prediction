"""
Hyperparameter optimization using Optuna.
"""

import numpy as np
import optuna
from sklearn.metrics import f1_score
from config import CONFIG, RANDOM_SEED


def optuna_objective_rf(trial, X_train, y_train, X_val, y_val, sample_weights=None):
    """
    Optuna objective function for Random Forest hyperparameter tuning.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation data
    sample_weights : array-like, optional
        Sample weights for training

    Returns:
    --------
    f1 : float
        F1 score on validation set
    """
    from sklearn.ensemble import RandomForestClassifier

    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.8]),
        'random_state': RANDOM_SEED,
        'class_weight': 'balanced',
        'n_jobs': -1
    }

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)

    return f1


def optuna_objective_xgb(trial, X_train, y_train, X_val, y_val, sample_weights=None):
    """
    Optuna objective function for XGBoost hyperparameter tuning.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation data
    sample_weights : array-like, optional
        Sample weights for training

    Returns:
    --------
    f1 : float
        F1 score on validation set
    """
    try:
        import xgboost as xgb
    except ImportError:
        return 0.0

    # Calculate scale_pos_weight
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / (n_pos + 1e-8)

    # Suggest hyperparameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'scale_pos_weight': scale_pos_weight,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbosity': 0
    }

    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)

    return f1


def optuna_objective_lgbm(trial, X_train, y_train, X_val, y_val, sample_weights=None):
    """
    Optuna objective function for LightGBM hyperparameter tuning.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation data
    sample_weights : array-like, optional
        Sample weights for training

    Returns:
    --------
    f1 : float
        F1 score on validation set
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return 0.0

    # Calculate scale_pos_weight
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / (n_pos + 1e-8)

    # Suggest hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'scale_pos_weight': scale_pos_weight,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }

    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)

    return f1


def optimize_hyperparameters(model_type, X_train, y_train, X_val, y_val,
                            sample_weights=None, n_trials=50, timeout=3600, verbose=True):
    """
    Optimize hyperparameters using Optuna.

    Parameters:
    -----------
    model_type : str
        Model type: 'rf', 'xgb', or 'lgbm'
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation data
    sample_weights : array-like, optional
        Sample weights for training
    n_trials : int
        Number of optimization trials
    timeout : int
        Timeout in seconds
    verbose : bool
        Print optimization progress

    Returns:
    --------
    best_params : dict
        Best hyperparameters found
    study : optuna.Study
        Optuna study object
    """
    # Select objective function
    if model_type == 'rf':
        objective_fn = optuna_objective_rf
    elif model_type == 'xgb':
        objective_fn = optuna_objective_xgb
    elif model_type == 'lgbm':
        objective_fn = optuna_objective_lgbm
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER OPTIMIZATION - {model_type.upper()}")
        print(f"{'='*60}")
        print(f"  Trials: {n_trials}")
        print(f"  Timeout: {timeout}s")

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )

    # Optimize
    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train, X_val, y_val, sample_weights),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose
    )

    best_params = study.best_params

    if verbose:
        print(f"\n✓ Optimization complete")
        print(f"  Best F1 score: {study.best_value:.4f}")
        print(f"  Best parameters:")
        for key, value in best_params.items():
            print(f"    {key}: {value}")
        print(f"{'='*60}\n")

    return best_params, study


def optimize_regression_hyperparameters(model_type, X_train, y_train, X_val, y_val,
                                       n_trials=50, timeout=3600, verbose=True):
    """
    Optimize hyperparameters for regression models.

    Parameters:
    -----------
    model_type : str
        Model type: 'rf' or 'xgb'
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation data
    n_trials : int
        Number of optimization trials
    timeout : int
        Timeout in seconds
    verbose : bool
        Print optimization progress

    Returns:
    --------
    best_params : dict
        Best hyperparameters found
    study : optuna.Study
        Optuna study object
    """
    from sklearn.metrics import mean_squared_error

    def objective_rf_regression(trial):
        from sklearn.ensemble import RandomForestRegressor

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Minimize RMSE
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    def objective_xgb_regression(trial):
        try:
            import xgboost as xgb
        except ImportError:
            return 1e9

        params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbosity': 0
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Minimize RMSE
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

    # Select objective
    if model_type == 'rf':
        objective_fn = objective_rf_regression
    elif model_type == 'xgb':
        objective_fn = objective_xgb_regression
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"REGRESSION HYPERPARAMETER OPTIMIZATION - {model_type.upper()}")
        print(f"{'='*60}")

    # Create study (minimize RMSE)
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )

    # Optimize
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose
    )

    best_params = study.best_params

    if verbose:
        print(f"\n✓ Optimization complete")
        print(f"  Best RMSE: {study.best_value:.2f}")
        print(f"  Best parameters:")
        for key, value in best_params.items():
            print(f"    {key}: {value}")
        print(f"{'='*60}\n")

    return best_params, study
