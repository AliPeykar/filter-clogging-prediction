"""
Survival analysis models for filter clogging prediction.
Handles censored data naturally - no SMOTE needed.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import CONFIG, RANDOM_SEED


class CoxPredictionModel:
    """
    Cox Proportional Hazards model with robust convergence handling.

    Features:
    - Automatic data cleaning (NaN/Inf removal)
    - Collinearity detection and removal
    - Progressive penalty fallback
    - Graceful degradation if convergence fails
    """

    def __init__(self):
        """Initialize Cox model wrapper."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.removed_features = []
        self.is_fitted = False

    def fit(self, X_train, duration_train, event_train, feature_names=None, verbose=True):
        """
        Train Cox Proportional Hazards model with robust error handling.

        Parameters:
        -----------
        X_train : array-like
            Training features
        duration_train : array-like
            Time to event (or censoring)
        event_train : array-like
            Event indicator (1=event occurred, 0=censored)
        feature_names : list, optional
            Feature names for interpretability
        verbose : bool
            Print training progress
        """
        try:
            from lifelines import CoxPHFitter
        except ImportError:
            if verbose:
                print("[!] lifelines not installed. Skipping Cox model.")
            self.model = None
            return self

        if verbose:
            print(f"\n{'='*60}")
            print(f"COX PROPORTIONAL HAZARDS TRAINING")
            print(f"{'='*60}")

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # Step 1: Clean data before scaling
        X_train_clean = X_train.copy()
        duration_train_clean = duration_train.copy()
        event_train_clean = event_train.copy()

        # Remove NaN and Inf values
        valid_mask = (
            ~np.isnan(X_train_clean).any(axis=1) &
            ~np.isinf(X_train_clean).any(axis=1) &
            ~np.isnan(duration_train_clean) &
            (duration_train_clean > 0)
        )

        n_removed = len(X_train_clean) - valid_mask.sum()
        if n_removed > 0 and verbose:
            print(f"[!] Removed {n_removed} samples with NaN/Inf values")

        X_train_clean = X_train_clean[valid_mask]
        duration_train_clean = duration_train_clean[valid_mask]
        event_train_clean = event_train_clean[valid_mask]

        if len(X_train_clean) == 0:
            if verbose:
                print("[X] No valid samples after cleaning. Skipping Cox model.")
            self.model = None
            return self

        # Step 2: Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_clean)

        # Handle any NaNs that appear after scaling
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 3: Create dataframe for lifelines
        train_df = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        train_df['duration'] = duration_train_clean
        train_df['event'] = event_train_clean

        # Step 3.5: Validate survival data has sufficient variation
        n_events = event_train_clean.sum()
        n_censored = len(event_train_clean) - n_events
        unique_durations = len(np.unique(duration_train_clean))

        if verbose:
            print(f"Survival data summary:")
            print(f"  Events: {n_events}, Censored: {n_censored}")
            print(f"  Unique duration values: {unique_durations}")
            print(f"  Duration range: [{duration_train_clean.min():.1f}, {duration_train_clean.max():.1f}]")

        # Check if we have admissible pairs (events at different times)
        if n_events < 2:
            if verbose:
                print(f"[!] Insufficient events ({n_events}) for Cox model. Need at least 2.")
                print(f"  Skipping Cox model.")
            self.model = None
            return self

        # Check if all events happen at the same time (no variation)
        event_durations = duration_train_clean[event_train_clean == 1]
        if len(np.unique(event_durations)) < 2:
            if verbose:
                print(f"[!] All events occur at same time (duration={event_durations[0]:.1f})")
                print(f"  No admissible pairs for Cox model. Skipping.")
            self.model = None
            return self

        # Step 4: Remove highly correlated features (collinearity)
        corr_matrix = train_df.drop(columns=['duration', 'event']).corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        high_corr_cols = [
            col for col in upper_triangle.columns
            if any(upper_triangle[col] > 0.95)
        ]

        if len(high_corr_cols) > 0:
            if verbose:
                print(f"Removing {len(high_corr_cols)} highly correlated features (r>0.95)")
            train_df = train_df.drop(columns=high_corr_cols)
            self.removed_features = high_corr_cols

        # Step 5: Progressive penalty fallback
        penalties = [0.1, 1.0, 5.0, 10.0]

        for attempt, penalizer in enumerate(penalties, 1):
            try:
                if verbose:
                    print(f"\nAttempt {attempt}/4: Training with penalizer={penalizer}")

                self.model = CoxPHFitter(
                    penalizer=penalizer,
                    l1_ratio=0.0  # Pure L2 regularization
                )

                self.model.fit(
                    train_df,
                    duration_col='duration',
                    event_col='event',
                    show_progress=False
                )

                if verbose:
                    print(f"[OK] Cox model converged successfully")
                    print(f"  Concordance index: {self.model.concordance_index_:.3f}")

                self.is_fitted = True
                return self

            except Exception as e:
                if verbose:
                    print(f"[X] Attempt {attempt} failed: {str(e)[:100]}")

                if attempt == len(penalties):
                    # Final attempt failed - graceful degradation
                    if verbose:
                        print(f"\n[!] Cox model failed to converge after {len(penalties)} attempts")
                        print(f"  Skipping Cox model. Will use Random Survival Forest instead.")
                    self.model = None
                    return self

        return self

    def predict_risk(self, X_test):
        """
        Predict risk scores for new samples.

        Parameters:
        -----------
        X_test : array-like
            Test features

        Returns:
        --------
        risk_scores : np.ndarray
            Risk scores (higher = more risk)
        """
        if self.model is None:
            raise RuntimeError("Cox model not fitted or failed to converge")

        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Create dataframe
        feature_cols = [f for f in self.feature_names if f not in self.removed_features]
        test_df = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        test_df = test_df[feature_cols]

        # Predict risk
        risk_scores = self.model.predict_partial_hazard(test_df).values

        return risk_scores

    def predict_survival(self, X_test, times):
        """
        Predict survival probabilities at specified times.

        Parameters:
        -----------
        X_test : array-like
            Test features
        times : array-like
            Time points for survival prediction

        Returns:
        --------
        survival_probs : np.ndarray, shape (n_samples, n_times)
            Survival probabilities
        """
        if self.model is None:
            raise RuntimeError("Cox model not fitted or failed to converge")

        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Create dataframe
        feature_cols = [f for f in self.feature_names if f not in self.removed_features]
        test_df = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        test_df = test_df[feature_cols]

        # Predict survival function
        survival_probs = self.model.predict_survival_function(test_df, times=times).T.values

        return survival_probs


class RandomSurvivalForestModel:
    """
    Random Survival Forest model wrapper.

    More robust than Cox for non-proportional hazards and non-linear relationships.
    """

    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=10):
        """
        Initialize Random Survival Forest.

        Parameters:
        -----------
        n_estimators : int
            Number of trees
        max_depth : int
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split node
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train, duration_train, event_train, verbose=True):
        """
        Train Random Survival Forest.

        Parameters:
        -----------
        X_train : array-like
            Training features
        duration_train : array-like
            Time to event (or censoring)
        event_train : array-like
            Event indicator (1=event occurred, 0=censored)
        verbose : bool
            Print training progress
        """
        try:
            from sksurv.ensemble import RandomSurvivalForest
        except ImportError:
            if verbose:
                print("[!] scikit-survival not installed. Skipping RSF model.")
            self.model = None
            return self

        if verbose:
            print(f"\n{'='*60}")
            print(f"RANDOM SURVIVAL FOREST TRAINING")
            print(f"{'='*60}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Create structured array for sksurv
        y_train = np.array(
            [(bool(e), t) for e, t in zip(event_train, duration_train)],
            dtype=[('event', bool), ('time', float)]
        )

        # Train model
        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        if verbose:
            print(f"[OK] Random Survival Forest trained successfully")
            print(f"  Trees: {self.n_estimators}")
            print(f"  Concordance index: {self.model.score(X_train_scaled, y_train):.3f}")

        self.is_fitted = True
        return self

    def predict_risk(self, X_test):
        """
        Predict risk scores (cumulative hazard).

        Parameters:
        -----------
        X_test : array-like
            Test features

        Returns:
        --------
        risk_scores : np.ndarray
            Risk scores (higher = more risk)
        """
        if self.model is None:
            raise RuntimeError("RSF model not fitted")

        X_test_scaled = self.scaler.transform(X_test)
        risk_scores = self.model.predict(X_test_scaled)

        return risk_scores

    def predict_survival(self, X_test, times):
        """
        Predict survival probabilities at specified times.

        Parameters:
        -----------
        X_test : array-like
            Test features
        times : array-like
            Time points for survival prediction

        Returns:
        --------
        survival_probs : np.ndarray, shape (n_samples, n_times)
            Survival probabilities
        """
        if self.model is None:
            raise RuntimeError("RSF model not fitted")

        X_test_scaled = self.scaler.transform(X_test)

        # Get survival functions
        survival_funcs = self.model.predict_survival_function(X_test_scaled)

        # Interpolate at specified times
        survival_probs = np.array([
            np.interp(times, fn.x, fn.y) for fn in survival_funcs
        ])

        return survival_probs
