"""
Anomaly detection module for filter clogging prediction.
Perfect for severely imbalanced data where clogging events are rare.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from config import CONFIG


class AnomalyDetectionModule:
    """
    Anomaly detection ensemble for filter degradation detection.

    Uses unsupervised learning on healthy data to detect abnormal behavior
    that may indicate impending clogging.

    Methods:
    --------
    - Isolation Forest: Tree-based anomaly detection
    - Local Outlier Factor (LOF): Density-based detection
    - One-Class SVM: Margin-based detection
    """

    def __init__(self, config=CONFIG):
        """
        Initialize anomaly detection ensemble.

        Parameters:
        -----------
        config : dict
            Configuration dictionary with anomaly detection settings
        """
        self.config = config
        ad_config = config.get('anomaly_detection', {})

        self.contamination = ad_config.get('contamination', 0.01)
        self.n_estimators = ad_config.get('n_estimators', 200)
        self.lof_neighbors = ad_config.get('lof_neighbors', 20)
        self.ensemble_weights = ad_config.get('ensemble_weights', [0.5, 0.3, 0.2])

        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.lof = None
        self.ocsvm = None
        self.is_fitted = False

    def fit(self, X_healthy, verbose=True):
        """
        Train anomaly detectors on healthy (non-clogged) data only.

        Parameters:
        -----------
        X_healthy : array-like, shape (n_samples, n_features)
            Training data from healthy filter operation
        verbose : bool
            Print training progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"ANOMALY DETECTION TRAINING")
            print(f"{'='*60}")
            print(f"Training on {len(X_healthy)} healthy samples")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_healthy)

        # 1. Isolation Forest
        if verbose:
            print("\n1. Training Isolation Forest...")
        self.isolation_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X_scaled)

        # 2. Local Outlier Factor (LOF)
        if verbose:
            print("2. Training Local Outlier Factor...")
        self.lof = LocalOutlierFactor(
            n_neighbors=self.lof_neighbors,
            contamination=self.contamination,
            novelty=True,  # Enable predict for new data
            n_jobs=-1
        )
        self.lof.fit(X_scaled)

        # 3. One-Class SVM
        if verbose:
            print("3. Training One-Class SVM...")
        self.ocsvm = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=self.contamination
        )
        self.ocsvm.fit(X_scaled)

        self.is_fitted = True

        if verbose:
            print(f"\n[OK] Anomaly detection ensemble trained successfully")
            print(f"{'='*60}\n")

    def predict_anomaly_scores(self, X):
        """
        Compute anomaly scores for new samples.

        Higher scores indicate higher likelihood of being anomalous (degraded).

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to score

        Returns:
        --------
        results : dict
            Dictionary with:
            - 'ensemble_score': Weighted ensemble anomaly scores (0-1)
            - 'is_anomaly': Binary predictions (1=anomaly, 0=normal)
            - 'if_score': Isolation Forest scores
            - 'lof_score': LOF scores
            - 'ocsvm_score': One-Class SVM scores
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict_anomaly_scores()")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get scores from each detector (higher = more anomalous)
        if_scores = -self.isolation_forest.decision_function(X_scaled)
        lof_scores = -self.lof.decision_function(X_scaled)
        ocsvm_scores = -self.ocsvm.decision_function(X_scaled)

        # Normalize scores to [0, 1] range
        if_scores = self._normalize_scores(if_scores)
        lof_scores = self._normalize_scores(lof_scores)
        ocsvm_scores = self._normalize_scores(ocsvm_scores)

        # Weighted ensemble
        ensemble_scores = np.average(
            [if_scores, lof_scores, ocsvm_scores],
            axis=0,
            weights=self.ensemble_weights
        )

        # Binary predictions (ensemble score > 0.5 = anomaly)
        is_anomaly = (ensemble_scores > 0.5).astype(int)

        results = {
            'ensemble_score': ensemble_scores,
            'is_anomaly': is_anomaly,
            'if_score': if_scores,
            'lof_score': lof_scores,
            'ocsvm_score': ocsvm_scores,
        }

        return results

    def _normalize_scores(self, scores):
        """
        Normalize scores to [0, 1] range using min-max scaling.

        Parameters:
        -----------
        scores : np.ndarray
            Raw anomaly scores

        Returns:
        --------
        normalized : np.ndarray
            Scores normalized to [0, 1]
        """
        min_score = np.min(scores)
        max_score = np.max(scores)

        if max_score - min_score < 1e-8:
            return np.ones_like(scores) * 0.5

        normalized = (scores - min_score) / (max_score - min_score)
        return normalized

    def get_feature_importance(self, feature_names):
        """
        Get feature importance from Isolation Forest.

        Parameters:
        -----------
        feature_names : list
            Names of features

        Returns:
        --------
        importance_df : pd.DataFrame
            Feature importance sorted by importance
        """
        if not self.is_fitted or self.isolation_forest is None:
            raise RuntimeError("Must fit Isolation Forest first")

        import pandas as pd

        # Get feature importances from trees
        importances = np.zeros(len(feature_names))
        for tree in self.isolation_forest.estimators_:
            importances += tree.feature_importances_

        importances /= len(self.isolation_forest.estimators_)

        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df
