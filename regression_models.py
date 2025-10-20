"""
Regression models for continuous time-to-clog prediction.
Alternative to classification that uses all data points efficiently.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from config import CONFIG, RANDOM_SEED


class RegressionPredictor:
    """
    Regression-based predictor for continuous time-to-clog estimation.

    Advantages over classification:
    - Uses ALL data points (not just near-clog samples)
    - Predicts continuous time-to-clog
    - Can convert to risk scores when needed
    - No artificial threshold needed
    """

    def __init__(self, model_type='rf'):
        """
        Initialize regression predictor.

        Parameters:
        -----------
        model_type : str
            'rf' for RandomForestRegressor or 'xgb' for XGBRegressor
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.max_time = None
        self.is_fitted = False

    def fit(self, X_train, y_time_to_clog, censored_mask=None, verbose=True):
        """
        Train regression model to predict time-to-clog.

        Parameters:
        -----------
        X_train : array-like
            Training features
        y_time_to_clog : array-like
            Time to clog (continuous target)
        censored_mask : array-like, optional
            Boolean mask indicating censored samples (True = censored)
        verbose : bool
            Print training progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"REGRESSION MODEL TRAINING ({self.model_type.upper()})")
            print(f"{'='*60}")

        # Handle censored data by capping at maximum observed time
        y_train = y_time_to_clog.copy()

        if censored_mask is not None:
            # First, determine max_time from all data (both censored and uncensored)
            self.max_time = np.max(y_time_to_clog)

            # Cap censored values at max observed event time from uncensored samples
            uncensored_times = y_time_to_clog[~censored_mask]
            if len(uncensored_times) > 0:
                max_uncensored_time = np.max(uncensored_times)
                # For censored samples, use their actual value (which should be capped at horizon)
                # Don't override with max_uncensored_time as they may vary
                if verbose:
                    n_censored = censored_mask.sum()
                    n_uncensored = (~censored_mask).sum()
                    print(f"  Uncensored samples: {n_uncensored} (max: {max_uncensored_time:.1f})")
                    print(f"  Censored samples: {n_censored} (capped at horizon)")
            else:
                # All samples are censored - use max of all values
                if verbose:
                    print(f"  All samples censored (max time: {self.max_time:.1f})")
        else:
            self.max_time = np.max(y_train)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=0
            )

        elif self.model_type == 'xgb':
            try:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    verbosity=0
                )
            except ImportError:
                if verbose:
                    print("[!] XGBoost not installed. Using RandomForest instead.")
                self.model_type = 'rf'
                self.model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    random_state=RANDOM_SEED,
                    n_jobs=-1
                )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Fit model
        self.model.fit(X_train_scaled, y_train)

        if verbose:
            print(f"[OK] {self.model_type.upper()} regression model trained")
            print(f"  Training samples: {len(X_train)}")
            print(f"  Max time-to-clog: {self.max_time:.1f}")

        self.is_fitted = True
        return self

    def predict_time_to_clog(self, X_test):
        """
        Predict continuous time-to-clog.

        Parameters:
        -----------
        X_test : array-like
            Test features

        Returns:
        --------
        time_predictions : np.ndarray
            Predicted time until clogging (in steps)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_test_scaled = self.scaler.transform(X_test)
        time_predictions = self.model.predict(X_test_scaled)

        # Clip to valid range [0, max_time]
        time_predictions = np.clip(time_predictions, 0, self.max_time)

        return time_predictions

    def predict_risk_score(self, X_test, horizon=120):
        """
        Convert time-to-clog predictions to risk scores.

        Risk Score = 1 - (predicted_time / horizon)
        - Risk = 1.0: clogging imminent (time ≈ 0)
        - Risk = 0.5: clogging at horizon
        - Risk = 0.0: clogging far away (time ≥ horizon)

        Parameters:
        -----------
        X_test : array-like
            Test features
        horizon : float
            Forecast horizon (in steps)

        Returns:
        --------
        risk_scores : np.ndarray
            Risk scores in [0, 1] range
        """
        time_predictions = self.predict_time_to_clog(X_test)

        # Convert to risk: closer to clog = higher risk
        risk_scores = 1.0 - (time_predictions / horizon)
        risk_scores = np.clip(risk_scores, 0.0, 1.0)

        return risk_scores

    def get_feature_importance(self, feature_names):
        """
        Get feature importance from the model.

        Parameters:
        -----------
        feature_names : list
            Names of features

        Returns:
        --------
        importance_dict : dict
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            # XGBoost
            importances = self.model.get_booster().get_score(importance_type='gain')
            importances = [importances.get(f'f{i}', 0) for i in range(len(feature_names))]
            importances = np.array(importances)

        importance_dict = {
            name: imp for name, imp in zip(feature_names, importances)
        }

        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return importance_dict


class EnsembleRegressionPredictor:
    """
    Ensemble of multiple regression models for robust predictions.

    Combines predictions from Random Forest and XGBoost.
    """

    def __init__(self, models=['rf', 'xgb'], weights=None):
        """
        Initialize ensemble predictor.

        Parameters:
        -----------
        models : list
            List of model types to include
        weights : list, optional
            Weights for each model (default: equal weights)
        """
        self.models = {}
        self.model_types = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.is_fitted = False

    def fit(self, X_train, y_time_to_clog, censored_mask=None, verbose=True):
        """
        Train all models in the ensemble.

        Parameters:
        -----------
        X_train : array-like
            Training features
        y_time_to_clog : array-like
            Time to clog (continuous target)
        censored_mask : array-like, optional
            Boolean mask indicating censored samples
        verbose : bool
            Print training progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"ENSEMBLE REGRESSION TRAINING")
            print(f"{'='*60}")
            print(f"Models: {', '.join(self.model_types)}")

        for model_type in self.model_types:
            if verbose:
                print(f"\nTraining {model_type.upper()}...")

            model = RegressionPredictor(model_type=model_type)
            model.fit(X_train, y_time_to_clog, censored_mask=censored_mask, verbose=False)
            self.models[model_type] = model

        if verbose:
            print(f"\n[OK] Ensemble trained with {len(self.models)} models")

        self.is_fitted = True
        return self

    def predict_time_to_clog(self, X_test):
        """
        Predict time-to-clog using weighted ensemble.

        Parameters:
        -----------
        X_test : array-like
            Test features

        Returns:
        --------
        ensemble_predictions : np.ndarray
            Weighted average predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        predictions = []
        for model_type in self.model_types:
            pred = self.models[model_type].predict_time_to_clog(X_test)
            predictions.append(pred)

        # Weighted average
        ensemble_predictions = np.average(predictions, axis=0, weights=self.weights)

        return ensemble_predictions

    def predict_risk_score(self, X_test, horizon=120):
        """
        Predict risk scores using weighted ensemble.

        Parameters:
        -----------
        X_test : array-like
            Test features
        horizon : float
            Forecast horizon

        Returns:
        --------
        ensemble_risk_scores : np.ndarray
            Weighted average risk scores
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        risk_scores = []
        for model_type in self.model_types:
            risk = self.models[model_type].predict_risk_score(X_test, horizon=horizon)
            risk_scores.append(risk)

        # Weighted average
        ensemble_risk_scores = np.average(risk_scores, axis=0, weights=self.weights)

        return ensemble_risk_scores
