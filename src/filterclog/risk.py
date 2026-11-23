"""Secondary target: six-level risk classification.

Critical rebuild of the legacy approach, which fabricated six classes by
quantile-binning a synthetic "combined risk" and then force-promoted clogged
rows. Here the levels are honest bands of the thesis rule's own score
(``validation_score``), with the top band aligned to the empirical clogged
threshold (score >= 11 in the stored data; the thesis text says >= 12 — a
documented discrepancy).

Hidden assumptions surfaced (and printed into every report):
* the target is the rule's score, so this experiment measures *rule
  emulation from causal features*, not independently observed risk;
* score levels are nearly monotone in time within the single lifecycle, so
  part of the measured skill is trend-following;
* bands are merged if underpopulated (<100 rows) to keep per-class estimates
  meaningful.

Ordinal-aware evaluation (quadratic-weighted kappa, level MAE, +/-1 accuracy)
replaces plain accuracy: confusing adjacent risk bands is a far smaller error
than jumping several bands, and QWK/MAE encode exactly that.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, recall_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

#: score -> level bands; L5 aligned with the empirical clogged rule (>= 11).
DEFAULT_BANDS = [(0, 1), (2, 3), (4, 5), (6, 8), (9, 10), (11, 20)]

RISK_CAVEATS = [
    "Risk levels are bands of the labeling rule's own score: this measures "
    "rule emulation from causal features, not independently observed risk.",
    "Score levels are nearly monotone in time within the single lifecycle; "
    "part of the measured skill is trend-following.",
    "Thesis text places the clogged decision at score >= 12; the stored data "
    "uses >= 11. Bands align with the data; the discrepancy is documented.",
]


def make_risk_target(
    score: pd.Series,
    bands: list[tuple[int, int]] | None = None,
    min_rows: int = 100,
    count_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Map the rule score to ordinal levels; merge underpopulated bands into
    their lower neighbour. Band occupancy is counted over ``count_idx``
    (pass the training indices so test composition cannot inform the target
    design, and so every retained level is guaranteed to be trainable)."""
    bands = list(bands or DEFAULT_BANDS)
    levels = np.full(len(score), -1, dtype=int)
    for lvl, (lo, hi) in enumerate(bands):
        levels[(score >= lo) & (score <= hi)] = lvl
    if (levels < 0).any():
        raise ValueError("score values outside all bands")
    counted = levels if count_idx is None else levels[count_idx]

    # merge sparse bands (keep at least 3 levels)
    while len(bands) > 3:
        counts = np.bincount(counted, minlength=len(bands))
        sparse = [i for i, c in enumerate(counts) if c < min_rows]
        if not sparse:
            break
        i = sparse[0]
        j = i - 1 if i > 0 else i + 1
        lo = min(bands[i][0], bands[j][0])
        hi = max(bands[i][1], bands[j][1])
        keep = min(i, j)
        bands = [b for k, b in enumerate(bands) if k not in (i, j)]
        bands.insert(keep, (lo, hi))
        levels = np.full(len(score), -1, dtype=int)
        for lvl, (blo, bhi) in enumerate(bands):
            levels[(score >= blo) & (score <= bhi)] = lvl
        counted = levels if count_idx is None else levels[count_idx]
        logger.info("merged sparse risk band; bands now %s", bands)
    return levels, bands


def evaluate_risk_model(
    X_train: np.ndarray,
    levels_train: np.ndarray,
    X_val: np.ndarray,
    levels_val: np.ndarray,
    X_test: np.ndarray,
    levels_test: np.ndarray,
    n_classes: int,
    seed: int = 42,
) -> dict:
    present = np.unique(levels_train)
    if len(present) < n_classes:
        raise ValueError(
            f"risk levels {sorted(set(range(n_classes)) - set(present))} have "
            "no training rows; pass count_idx=train indices to make_risk_target"
        )
    # class weights via sample_weight (thesis's class-weight methodology,
    # multiclass form)
    counts = np.bincount(levels_train, minlength=n_classes).astype(float)
    weights = len(levels_train) / (n_classes * np.clip(counts, 1, None))
    sw = weights[levels_train]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=1000,
        early_stopping_rounds=50,
        eval_metric="mlogloss",
        max_depth=6,
        learning_rate=0.05,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, levels_train, sample_weight=sw,
              eval_set=[(X_val, levels_val)], verbose=False)

    pred = model.predict(X_test)
    qwk = float(cohen_kappa_score(levels_test, pred, weights="quadratic"))
    mae = float(np.abs(levels_test - pred).mean())
    within1 = float((np.abs(levels_test - pred) <= 1).mean())
    per_level_recall = recall_score(
        levels_test, pred, labels=list(range(n_classes)), average=None, zero_division=0
    )
    result = {
        "n_classes": n_classes,
        "quadratic_weighted_kappa": qwk,
        "level_mae": mae,
        "within_one_accuracy": within1,
        "per_level_recall": [round(float(r), 4) for r in per_level_recall],
        "confusion_matrix": confusion_matrix(
            levels_test, pred, labels=list(range(n_classes))
        ).tolist(),
        "test_level_counts": np.bincount(levels_test, minlength=n_classes).tolist(),
        "caveats": RISK_CAVEATS,
    }
    logger.info("risk model: QWK=%.3f, MAE=%.3f, +/-1 acc=%.3f", qwk, mae, within1)
    return result
