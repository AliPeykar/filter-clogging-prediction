"""Probability calibration and decision-threshold selection on validation.

Legacy used raw probabilities at the implicit 0.5 threshold. Here:

* Platt (sigmoid) calibration is fitted on validation probabilities —
  isotonic is avoided because the 1,200-row validation block is thin and
  isotonic overfits small samples.
* The decision threshold is chosen on *calibrated* validation probabilities:
  primary rule max-F1; an operational threshold at FPR <= 1% (on the healthy
  validation block) is reported alongside for maintenance-planning use.
* The test set plays no role in either step.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger(__name__)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Platt scaling on the logit of the raw probability (the standard
    formulation operates on decision scores, not probabilities); moderate C
    keeps the fit stable on the nearly-separable validation block."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)

    def fit(self, proba_val: np.ndarray, y_val: np.ndarray) -> "PlattCalibrator":
        self._lr.fit(_logit(proba_val).reshape(-1, 1), y_val)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(_logit(proba).reshape(-1, 1))[:, 1]

    def coef(self) -> tuple[float, float]:
        """(slope, intercept) — persisted so a reuse run can reapply the
        exact calibration instead of thresholding raw probabilities."""
        return float(self._lr.coef_[0, 0]), float(self._lr.intercept_[0])

    @classmethod
    def from_coef(cls, slope: float, intercept: float) -> "PlattCalibrator":
        cal = cls()
        cal._lr.classes_ = np.array([0, 1])
        cal._lr.coef_ = np.array([[slope]])
        cal._lr.intercept_ = np.array([intercept])
        return cal


def select_threshold_max_f1(proba_val: np.ndarray, y_val: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.nanargmax(f1[:-1]))  # last point has no threshold
    thr = float(thresholds[best])
    logger.info("max-F1 threshold=%.4f (val F1=%.4f)", thr, f1[best])
    return thr


def select_threshold_fpr(proba_val: np.ndarray, y_val: np.ndarray, max_fpr: float = 0.01) -> float:
    """Smallest threshold whose FPR on the healthy validation rows is
    <= max_fpr (flags exactly floor(n * max_fpr) of them).

    Transfer caveat (also emitted in run reports): the validation healthy
    block is far more separable than the test onset boundary, so this
    threshold does NOT transfer to the held-out-transition regime — it is
    reported for completeness, not as an operational recommendation.
    """
    neg = proba_val[y_val == 0]
    if len(neg) == 0:
        raise ValueError("no healthy rows in validation block")
    allowed = int(np.floor(len(neg) * max_fpr))
    neg_desc = np.sort(neg)[::-1]
    thr = float(neg_desc[allowed]) + 1e-12
    logger.info("FPR<=%.1f%% threshold=%.4f (flags %d/%d healthy val rows)",
                100 * max_fpr, thr, allowed, len(neg))
    return thr
