"""
Main FilterCloggingPredictor class - orchestrates all models and predictions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from config import CONFIG, RANDOM_SEED
from utils import (
    time_series_split_imbalanced,
    compute_temporal_weights,
    combine_sample_weights,
    optimize_threshold_by_cost
)
from data_processing import (
    load_and_prepare_data,
    compute_target_labels,
    prepare_features_and_targets
)
from feature_engineering import engineer_all_features
from anomaly_detection import AnomalyDetectionModule
from survival_models import CoxPredictionModel, RandomSurvivalForestModel
from regression_models import RegressionPredictor, EnsembleRegressionPredictor
from evaluation import evaluate_classification_model, evaluate_regression_model


class FilterCloggingPredictor:
    """
    Comprehensive filter clogging prediction system.

    Combines multiple approaches:
    1. Anomaly detection for early degradation detection
    2. Classification models for binary risk prediction
    3. Survival analysis for censored data handling
    4. Regression models for continuous time-to-clog estimation
    """

    def __init__(self, config=CONFIG):
        """
        Initialize predictor with configuration.

        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.scaler = StandardScaler()

        # Model components
        self.anomaly_detector = None
        self.classification_models = {}
        self.survival_models = {}
        self.regression_models = {}

        # Training data info
        self.feature_names = None
        self.is_fitted = False
        self.optimal_threshold = 0.5

    def fit(self, X_train, y_class, y_time, y_duration, y_event,
            X_val=None, y_class_val=None, X_healthy=None, verbose=True):
        """
        Train all models.

        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_class : np.ndarray
            Binary classification target
        y_time : np.ndarray
            Time-to-clog for regression
        y_duration : np.ndarray
            Duration for survival analysis
        y_event : np.ndarray
            Event indicator for survival analysis
        X_val, y_class_val : optional
            Validation data for hyperparameter tuning
        X_healthy : pd.DataFrame or np.ndarray, optional
            Healthy data samples for anomaly detection
        verbose : bool
            Print training progress
        """
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# FILTER CLOGGING PREDICTOR - TRAINING")
            print(f"{'#'*60}\n")

        # Store feature names and convert all inputs to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # Convert validation set if it's a DataFrame
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values

        # Convert X_healthy if it's a DataFrame
        if X_healthy is not None and isinstance(X_healthy, pd.DataFrame):
            X_healthy = X_healthy.values

        # 1. Train anomaly detector
        if self.config.get('anomaly_detection', {}).get('enabled', True):
            if verbose:
                print("\n[1/4] Training Anomaly Detector...")

            self.anomaly_detector = AnomalyDetectionModule(config=self.config)

            if X_healthy is not None:
                X_healthy_data = X_healthy
            else:
                # Use samples where time_to_clog > horizon
                horizon = self.config.get('forecast_horizon_steps', 120)
                healthy_mask = y_time > horizon
                X_healthy_data = X_train[healthy_mask]

            self.anomaly_detector.fit(X_healthy_data, verbose=verbose)

        # 2. Train classification models
        if verbose:
            print("\n[2/4] Training Classification Models...")

        # Compute sample weights
        class_weights = self._compute_class_weights(y_class)

        if self.config.get('use_temporal_weighting', True):
            temporal_weights = compute_temporal_weights(y_time, self.config)
            sample_weights = combine_sample_weights(class_weights, temporal_weights)
        else:
            sample_weights = class_weights

        # Scale features - convert to DataFrame to preserve feature names for LGBM
        X_train_scaled_array = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=self.feature_names)

        # Train each model type
        models_to_use = self.config.get('models_to_use', ['rf', 'xgb', 'lgbm'])

        for model_type in models_to_use:
            if verbose:
                print(f"\n  Training {model_type.upper()}...")

            model = self._create_classification_model(model_type)

            try:
                # LGBM needs DataFrame with column names, others work with either
                if model_type == 'lgbm':
                    model.fit(X_train_scaled, y_class, sample_weight=sample_weights)
                else:
                    model.fit(X_train_scaled_array, y_class, sample_weight=sample_weights)

                self.classification_models[model_type] = model

                if verbose:
                    print(f"  [OK] {model_type.upper()} trained successfully")

            except Exception as e:
                if verbose:
                    print(f"  [X] {model_type.upper()} training failed: {str(e)[:100]}")

        # Optimize threshold on validation set (only if we have models and they're fitted)
        if X_val is not None and y_class_val is not None and len(self.classification_models) > 0:
            if verbose:
                print("\n  Optimizing decision threshold...")

            try:
                X_val_scaled_array = self.scaler.transform(X_val)
                X_val_scaled = pd.DataFrame(X_val_scaled_array, columns=self.feature_names)

                # Get predictions directly from models instead of predict_proba
                probas = []
                for model_name, model in self.classification_models.items():
                    # Use DataFrame for LGBM, array for others
                    if model_name == 'lgbm':
                        proba = model.predict_proba(X_val_scaled)
                    else:
                        proba = model.predict_proba(X_val_scaled_array)
                    probas.append(proba)

                if len(probas) > 0:
                    avg_proba = np.mean(probas, axis=0)
                    y_proba = avg_proba[:, 1]

                    cost_fn = self.config.get('cost_fn', 100)
                    cost_fp = self.config.get('cost_fp', 1)

                    self.optimal_threshold, optimal_cost, _ = optimize_threshold_by_cost(
                        y_class_val, y_proba, cost_fn=cost_fn, cost_fp=cost_fp
                    )

                    if verbose:
                        print(f"  [OK] Optimal threshold: {self.optimal_threshold:.3f}")
                        print(f"    Minimum cost: {optimal_cost:.0f}")
                else:
                    if verbose:
                        print(f"  [X] No valid models for threshold optimization")
            except Exception as e:
                if verbose:
                    print(f"  [X] Threshold optimization failed: {str(e)[:100]}")

        # 3. Train survival models
        if self.config.get('use_survival', True):
            if verbose:
                print("\n[3/4] Training Survival Models...")

            survival_model_types = self.config.get('survival_models', ['cox', 'rsf'])

            if 'cox' in survival_model_types:
                cox_model = CoxPredictionModel()
                cox_model.fit(X_train, y_duration, y_event, self.feature_names, verbose=verbose)
                if cox_model.model is not None:
                    self.survival_models['cox'] = cox_model

            if 'rsf' in survival_model_types:
                rsf_model = RandomSurvivalForestModel(n_estimators=100, max_depth=10)
                rsf_model.fit(X_train, y_duration, y_event, verbose=verbose)
                if rsf_model.model is not None:
                    self.survival_models['rsf'] = rsf_model

        # 4. Train regression models
        if self.config.get('use_regression', True):
            if verbose:
                print("\n[4/4] Training Regression Models...")

            regression_model_types = self.config.get('regression_models', ['rf', 'xgb'])

            # Censored mask: samples where event didn't occur within horizon
            horizon = self.config.get('forecast_horizon_steps', 120)
            censored_mask = y_time > horizon

            for model_type in regression_model_types:
                reg_model = RegressionPredictor(model_type=model_type)
                reg_model.fit(X_train, y_time, censored_mask=censored_mask, verbose=verbose)

                if reg_model.is_fitted:
                    self.regression_models[model_type] = reg_model

        self.is_fitted = True

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# TRAINING COMPLETE")
            print(f"{'#'*60}")
            print(f"  Classification models: {len(self.classification_models)}")
            print(f"  Survival models: {len(self.survival_models)}")
            print(f"  Regression models: {len(self.regression_models)}")
            print(f"  Anomaly detector: {'[OK]' if self.anomaly_detector else '[X]'}")
            print(f"{'#'*60}\n")

    def predict_proba(self, X, use_anomaly=True):
        """
        Predict probabilities for classification.

        Parameters:
        -----------
        X : np.ndarray
            Features to predict
        use_anomaly : bool
            Whether to include anomaly detection scores

        Returns:
        --------
        proba : np.ndarray, shape (n_samples, 2)
            Predicted probabilities [P(healthy), P(clogged)]
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Prepare data - convert to DataFrame for LGBM
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X

        # Get predictions from each classifier
        probas = []

        for model_name, model in self.classification_models.items():
            # Use DataFrame for LGBM, array for others
            if model_name == 'lgbm':
                proba = model.predict_proba(X_df)
            else:
                proba = model.predict_proba(X if isinstance(X, np.ndarray) else X.values)

            # Ensure proba has 2 columns (handle models that only learned one class)
            if proba.shape[1] == 1:
                # If only one class, create 2-column array
                if hasattr(model, 'classes_') and len(model.classes_) == 1:
                    if model.classes_[0] == 0:
                        # Only learned class 0 (healthy)
                        proba_2col = np.column_stack([proba, np.zeros((len(X), 1))])
                    else:
                        # Only learned class 1 (clogged)
                        proba_2col = np.column_stack([np.zeros((len(X), 1)), proba])
                    proba = proba_2col
                else:
                    # Default: assume class 0
                    proba = np.column_stack([proba, np.zeros((len(X), 1))])

            probas.append(proba)

        # Average predictions
        if len(probas) > 0:
            avg_proba = np.mean(probas, axis=0)
        else:
            # Fallback to 50/50
            avg_proba = np.ones((len(X), 2)) * 0.5

        # Ensure avg_proba has correct shape
        if avg_proba.shape[1] != 2:
            avg_proba = np.ones((len(X), 2)) * 0.5

        # Add anomaly detection scores if enabled
        if use_anomaly and self.anomaly_detector is not None:
            # Reverse transform to original scale for anomaly detector
            X_original = self.scaler.inverse_transform(X)
            anomaly_results = self.anomaly_detector.predict_anomaly_scores(X_original)
            anomaly_scores = anomaly_results['ensemble_score']

            # Blend with classification probabilities
            # Reduce anomaly weight to 15% to reduce false positives (was 30%)
            blend_weight = self.config.get('anomaly_blend_weight', 0.15)
            avg_proba[:, 1] = (1 - blend_weight) * avg_proba[:, 1] + blend_weight * anomaly_scores
            avg_proba[:, 0] = 1 - avg_proba[:, 1]

        return avg_proba

    def predict(self, X, threshold=None, use_anomaly=True):
        """
        Predict binary labels.

        Parameters:
        -----------
        X : np.ndarray
            Features to predict
        threshold : float, optional
            Decision threshold (default: use optimal threshold from training)
        use_anomaly : bool
            Whether to include anomaly detection

        Returns:
        --------
        predictions : np.ndarray
            Binary predictions (0=healthy, 1=clogging)
        """
        if threshold is None:
            threshold = self.optimal_threshold

        proba = self.predict_proba(X, use_anomaly=use_anomaly)
        predictions = (proba[:, 1] >= threshold).astype(int)

        return predictions

    def predict_time_to_clog(self, X):
        """
        Predict continuous time-to-clog using regression models.

        Parameters:
        -----------
        X : np.ndarray
            Features to predict

        Returns:
        --------
        time_predictions : np.ndarray
            Predicted time until clogging (in steps)
        """
        if not self.is_fitted or len(self.regression_models) == 0:
            raise RuntimeError("No regression models fitted.")

        # Get predictions from each regressor
        predictions = []

        for model_name, model in self.regression_models.items():
            # Reverse transform to original scale
            X_original = self.scaler.inverse_transform(X)
            pred = model.predict_time_to_clog(X_original)
            predictions.append(pred)

        # Average predictions
        time_predictions = np.mean(predictions, axis=0)

        return time_predictions

    def predict_risk_level(self, X, use_anomaly=True, return_details=False):
        """
        Predict 4-level risk scores instead of binary classification.

        Parameters:
        -----------
        X : np.ndarray
            Features to predict
        use_anomaly : bool
            Whether to include anomaly detection scores
        return_details : bool
            If True, return detailed breakdown by model component

        Returns:
        --------
        results : dict
            Dictionary containing:
            - 'risk_levels': Array of integers [0, 1, 2, 3] for [LOW, MODERATE, HIGH, CRITICAL]
            - 'risk_scores': Continuous probability scores [0-1]
            - 'risk_labels': Human-readable labels ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
            - 'risk_descriptions': Detailed descriptions for each prediction
            - 'recommended_actions': Recommended actions for each prediction
            - 'details' (if return_details=True): Breakdown by model component
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get continuous probability scores
        proba = self.predict_proba(X, use_anomaly=use_anomaly)
        risk_scores = proba[:, 1]  # Probability of clogging

        # Get risk level configuration
        risk_config = self.config.get('risk_levels', {})
        if not risk_config.get('enabled', True):
            raise RuntimeError("Risk level prediction is disabled in config.")

        thresholds = risk_config.get('thresholds', {
            'low': 0.25, 'moderate': 0.50, 'high': 0.75, 'critical': 1.00
        })
        labels = risk_config.get('labels', {
            0: 'LOW', 1: 'MODERATE', 2: 'HIGH', 3: 'CRITICAL'
        })
        descriptions = risk_config.get('descriptions', {
            0: 'Filter operating normally. No signs of clogging detected.',
            1: 'Minor degradation detected. Filter performance within acceptable range.',
            2: 'Significant degradation detected. Filter approaching critical threshold.',
            3: 'Severe clogging detected. Filter performance critically impaired.'
        })
        actions = risk_config.get('actions', {
            0: 'Normal operation - Continue routine monitoring',
            1: 'Early warning - Increase monitoring frequency',
            2: 'Action recommended - Schedule maintenance within 24-48 hours',
            3: 'Immediate action required - Urgent maintenance needed NOW'
        })

        # Map continuous scores to discrete risk levels
        risk_levels = np.zeros(len(risk_scores), dtype=int)
        risk_levels[risk_scores >= thresholds['low']] = 1
        risk_levels[risk_scores >= thresholds['moderate']] = 2
        risk_levels[risk_scores >= thresholds['high']] = 3

        # Get human-readable labels
        risk_labels = np.array([labels[level] for level in risk_levels])
        risk_descriptions = np.array([descriptions[level] for level in risk_levels])
        recommended_actions = np.array([actions[level] for level in risk_levels])

        # Prepare results
        results = {
            'risk_levels': risk_levels,
            'risk_scores': risk_scores,
            'risk_labels': risk_labels,
            'risk_descriptions': risk_descriptions,
            'recommended_actions': recommended_actions
        }

        # Add detailed breakdown if requested
        if return_details:
            details = {}

            # Classification component
            proba_no_anomaly = self.predict_proba(X, use_anomaly=False)
            details['classification'] = proba_no_anomaly[:, 1]

            # Anomaly detection component
            if use_anomaly and self.anomaly_detector is not None:
                X_original = self.scaler.inverse_transform(X)
                anomaly_results = self.anomaly_detector.predict_anomaly_scores(X_original)
                details['anomaly'] = anomaly_results['ensemble_score']

            # Regression component (if available)
            if len(self.regression_models) > 0:
                X_original = self.scaler.inverse_transform(X)
                horizon = self.config.get('forecast_horizon_steps', 120)

                reg_risks = []
                for model in self.regression_models.values():
                    risk = model.predict_risk_score(X_original, horizon=horizon)
                    reg_risks.append(risk)

                details['regression'] = np.mean(reg_risks, axis=0)

            results['details'] = details

        return results

    def predict_risk_scores(self, X, use_all_models=True):
        """
        Predict comprehensive risk scores from all models.

        Parameters:
        -----------
        X : np.ndarray
            Features to predict
        use_all_models : bool
            Whether to use all available models

        Returns:
        --------
        risk_scores : dict
            Dictionary with risk scores from each model type
        """
        risk_scores = {}

        # Classification probability
        proba = self.predict_proba(X, use_anomaly=False)
        risk_scores['classification'] = proba[:, 1]

        # Anomaly detection
        if self.anomaly_detector is not None:
            X_original = self.scaler.inverse_transform(X)
            anomaly_results = self.anomaly_detector.predict_anomaly_scores(X_original)
            risk_scores['anomaly'] = anomaly_results['ensemble_score']

        # Regression-based risk
        if len(self.regression_models) > 0:
            X_original = self.scaler.inverse_transform(X)
            horizon = self.config.get('forecast_horizon_steps', 120)

            reg_risks = []
            for model in self.regression_models.values():
                risk = model.predict_risk_score(X_original, horizon=horizon)
                reg_risks.append(risk)

            risk_scores['regression'] = np.mean(reg_risks, axis=0)

        # Ensemble risk (weighted average)
        if use_all_models:
            scores_list = []
            weights_list = []

            if 'classification' in risk_scores:
                scores_list.append(risk_scores['classification'])
                weights_list.append(0.4)

            if 'anomaly' in risk_scores:
                scores_list.append(risk_scores['anomaly'])
                weights_list.append(0.3)

            if 'regression' in risk_scores:
                scores_list.append(risk_scores['regression'])
                weights_list.append(0.3)

            if len(scores_list) > 0:
                risk_scores['ensemble'] = np.average(scores_list, axis=0, weights=weights_list)

        return risk_scores

    def _compute_class_weights(self, y):
        """Compute balanced class weights."""
        from sklearn.utils.class_weight import compute_sample_weight
        return compute_sample_weight('balanced', y)

    def _create_classification_model(self, model_type):
        """Create classification model of specified type."""
        if model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1
            )

        elif model_type == 'xgb':
            import xgboost as xgb
            # Scale pos weight for imbalance
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )

        elif model_type == 'lgbm':
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=-1
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Wrap with calibration for better probability estimates
        calibrated_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')

        return calibrated_model
