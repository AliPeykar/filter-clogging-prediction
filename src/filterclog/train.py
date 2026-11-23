"""Model training: XGBoost (primary, tuned) and RandomForest (fixed baseline).

Corrections vs legacy ``ProductionModelTrainer``:

* XGBoost early-stops on the *validation* block (``eval_metric='aucpr'``);
  legacy had no early stopping at all.
* The RF baseline uses fixed, sensible defaults — it is a comparison point,
  not a tuned competitor (the legacy GA run accidentally promoted a tuned RF
  to headline model).
* No scaler: both models are tree ensembles and scale-invariant; the legacy
  RobustScaler added a fit/transform surface without benefit.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_xgb(
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    seed: int = 42,
    max_n_estimators: int = 2000,
    early_stopping_rounds: int = 50,
    fixed_n_estimators: int | None = None,
    monotone: tuple[int, ...] | None = None,
) -> XGBClassifier:
    """Train with early stopping on (X_val, y_val); if no validation block is
    available (strict-temporal S1), ``fixed_n_estimators`` — typically the
    best iteration found under S2 — must be supplied."""
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    if X_val is not None and len(X_val):
        # logloss, not aucpr: ranking metrics saturate on the separable
        # validation block and stop boosting after one round (see hpo.py)
        model = XGBClassifier(
            n_estimators=max_n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            monotone_constraints=monotone,
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        logger.info(
            "XGB trained: best_iteration=%d (early stopping), spw=%.2f",
            model.best_iteration, spw,
        )
    else:
        if fixed_n_estimators is None:
            raise ValueError("no validation block and no fixed_n_estimators supplied")
        model = XGBClassifier(
            n_estimators=fixed_n_estimators,
            eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            monotone_constraints=monotone,
            **params,
        )
        model.fit(X_train, y_train, verbose=False)
        logger.info("XGB trained without early stopping: n_estimators=%d", fixed_n_estimators)
    return model


def train_rf_baseline(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = 42
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
