"""Evaluation: point metrics, moving-block bootstrap CIs, event-level
detection delay, and paired model comparison.

Statistical honesty notes baked into the outputs:

* Consecutive rows are strongly autocorrelated, so an iid bootstrap would
  understate variance — a *moving-block* bootstrap (block length >= the
  residual autocorrelation range) is used instead. Its CIs quantify sampling
  noise within ONE filter lifecycle only; generalization across filters
  cannot be claimed from this dataset.
* Detection delay has no CI: the dataset contains exactly one onset event.
  Sensitivity across alarm rules and thresholds is reported instead.
* McNemar's test on paired errors is reported with the explicit caveat that
  autocorrelation inflates the effective sample size, making its p-value
  optimistic ("consistent with", not "proof of").
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ metrics
def point_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "ece": float(expected_calibration_error(y_true, proba)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        out["pr_auc"] = float(average_precision_score(y_true, proba))
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        out["pr_auc"] = None
        out["roc_auc"] = None
        out["note"] = "one-class evaluation window: ranking metrics undefined"
    return out


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(proba[mask].mean() - y_true[mask].mean())
    return ece


# ------------------------------------------------- moving-block bootstrap CI
def block_bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    block_len: int = 50,
    n_resamples: int = 1000,
    seed: int = 42,
    metrics: tuple[str, ...] = ("f1", "balanced_accuracy", "mcc", "pr_auc", "roc_auc"),
) -> dict:
    """95% percentile CIs from a moving-block bootstrap over the (time-ordered)
    evaluation window."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if block_len >= n:
        raise ValueError(f"block_len ({block_len}) must be < window length ({n})")
    n_blocks = int(np.ceil(n / block_len))
    starts_max = n - block_len
    samples: dict[str, list[float]] = {m: [] for m in metrics}

    for _ in range(n_resamples):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        yt, pp = y_true[idx], proba[idx]
        if len(np.unique(yt)) < 2:
            continue  # degenerate resample: ranking metrics undefined
        pred = (pp >= threshold).astype(int)
        vals = {
            "f1": f1_score(yt, pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(yt, pred),
            "mcc": matthews_corrcoef(yt, pred),
            "pr_auc": average_precision_score(yt, pp),
            "roc_auc": roc_auc_score(yt, pp),
        }
        for m in metrics:
            samples[m].append(float(vals[m]))

    out = {}
    for m in metrics:
        arr = np.array(samples[m])
        if len(arr) < 100:
            out[m] = {"ci_low": None, "ci_high": None, "n_valid_resamples": int(len(arr))}
            continue
        out[m] = {
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "n_valid_resamples": int(len(arr)),
        }
    out["_config"] = {"block_len": block_len, "n_resamples": n_resamples, "seed": seed}
    return out


def paired_bootstrap_diff(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    block_len: int = 50,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """CI on F1(A) - F1(B) using identical resampled blocks for both models."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    n_blocks = int(np.ceil(n / block_len))
    diffs = []
    for _ in range(n_resamples):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        fa = f1_score(yt, (proba_a[idx] >= threshold_a).astype(int), zero_division=0)
        fb = f1_score(yt, (proba_b[idx] >= threshold_b).astype(int), zero_division=0)
        diffs.append(fa - fb)
    arr = np.array(diffs)
    if len(arr) < 100:
        return {
            "metric": "f1_diff (A - B)",
            "mean": None, "ci_low": None, "ci_high": None,
            "n_valid_resamples": int(len(arr)),
        }
    return {
        "metric": "f1_diff (A - B)",
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "n_valid_resamples": int(len(arr)),
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar on paired errors. Caveat (also emitted in the result):
    autocorrelated rows inflate effective n -> optimistic p-value."""
    err_a = pred_a != y_true
    err_b = pred_b != y_true
    b = int((err_a & ~err_b).sum())  # A wrong, B right
    c = int((~err_a & err_b).sum())  # A right, B wrong
    n = b + c
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))
    return {
        "b_A_wrong_B_right": b,
        "c_A_right_B_wrong": c,
        "p_value": p,
        "caveat": (
            "rows are autocorrelated within one lifecycle; effective sample "
            "size is inflated, so this p-value is optimistic"
        ),
    }


# ------------------------------------------------------- detection delay
def detection_delay(
    proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    consecutive: int = 3,
    seconds_per_row: float | None = None,
) -> dict:
    """Event-level evaluation on a window containing the (single) onset.

    Alarm = first row where `consecutive` successive probabilities >= threshold.
    Delay = alarm row - true onset row (negative = early alarm).
    Also reports false-alarm episodes fully inside the healthy prefix.
    """
    onset_positions = np.flatnonzero(y_true == 1)
    if len(onset_positions) == 0:
        return {"note": "no positive rows in window; delay undefined"}
    onset = int(onset_positions[0])

    above = proba >= threshold
    # alarm is credited at DECISION time: the k-th consecutive exceedance,
    # since an operator cannot declare it before the k-th confirmation
    alarm_row = None
    run = 0
    for i, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run >= consecutive:
            alarm_row = i
            break

    # alarm flapping: maximal sustained runs (>= k consecutive) that CLEAR
    # again before the true onset. The first such run is the alarm itself; if
    # it clears pre-onset it is counted, because an operator would see the
    # alarm raised and retracted. A run still active at onset is not counted.
    cleared = 0
    run = 0
    for i in range(onset):
        run = run + 1 if above[i] else 0
        if run >= consecutive and (i + 1 == onset or not above[i + 1]):
            if i + 1 < onset:  # run terminates strictly before onset
                cleared += 1
            run = 0

    out = {
        "onset_row_in_window": onset,
        "alarm_decision_row_in_window": alarm_row,
        "delay_rows": None if alarm_row is None else int(alarm_row - onset),
        "consecutive_rule": consecutive,
        "alarm_convention": "delay measured at the k-th confirmation row",
        "cleared_alarm_episodes_before_onset": cleared,
        "n_events": 1,
        "note": "single onset event: no CI on delay; see sensitivity table",
    }
    if seconds_per_row and alarm_row is not None:
        out["delay_minutes"] = round((alarm_row - onset) * seconds_per_row / 60.0, 2)
    return out


def anticipation_profile(
    proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    windows: tuple[int, ...] = (0, 60, 120, 300, 600),
    seconds_per_row: float | None = None,
) -> list[dict]:
    """Anticipation-aware pointwise metrics (range-based evaluation in the
    spirit of Tatbul et al., NeurIPS 2018).

    Pointwise scoring counts a flag raised shortly *before* the labeling
    rule's onset as a false positive, penalizing exactly the early warning a
    predictive-maintenance system is built for. For each anticipation window
    W, healthy rows within W rows before the onset are excluded from FP
    counting (treated as acceptable early warning), and precision/F1 are
    recomputed. W=0 reproduces the strict pointwise metrics; the full curve
    is reported so no single W has to be defended.
    """
    onset_positions = np.flatnonzero(y_true == 1)
    if len(onset_positions) == 0:
        return [{"note": "no positive rows; anticipation profile undefined"}]
    onset = int(onset_positions[0])
    pred = (proba >= threshold).astype(int)

    rows = []
    for w in windows:
        keep = np.ones(len(y_true), dtype=bool)
        excl_lo = max(onset - w, 0)
        # exclude only healthy rows inside the anticipation band
        band = np.arange(excl_lo, onset)
        keep[band[y_true[band] == 0]] = False
        yt, pd_ = y_true[keep], pred[keep]
        entry = {
            "anticipation_rows": int(w),
            "n_excluded": int((~keep).sum()),
            "precision": float(precision_score(yt, pd_, zero_division=0)),
            "recall": float(recall_score(yt, pd_, zero_division=0)),
            "f1": float(f1_score(yt, pd_, zero_division=0)),
            "mcc": float(matthews_corrcoef(yt, pd_)) if len(np.unique(yt)) == 2 else None,
        }
        if seconds_per_row:
            entry["anticipation_minutes"] = round(w * seconds_per_row / 60.0, 1)
        rows.append(entry)
    return rows


def delay_sensitivity(
    proba: np.ndarray,
    y_true: np.ndarray,
    thresholds: dict[str, float],
    consecutive_values: tuple[int, ...] = (1, 3, 5),
    seconds_per_row: float | None = None,
) -> list[dict]:
    rows = []
    for name, thr in thresholds.items():
        for k in consecutive_values:
            d = detection_delay(proba, y_true, thr, consecutive=k, seconds_per_row=seconds_per_row)
            rows.append({"threshold_rule": name, "threshold": float(thr), "consecutive": k, **d})
    return rows
