"""Unit tests for the statistics and CV machinery (detection delay, threshold
selectors, segment folds, calibrator round-trip)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filterclog.calibration import (
    PlattCalibrator,
    select_threshold_fpr,
    select_threshold_max_f1,
)
from filterclog.cv import expanding_window_folds
from filterclog.evaluation import anticipation_profile, detection_delay


# ------------------------------------------------------------ detection delay
def test_delay_alarm_at_decision_row():
    # onset at 10; probabilities exceed threshold from row 6 onward
    y = np.array([0] * 10 + [1] * 10)
    p = np.array([0.1] * 6 + [0.9] * 14)
    d = detection_delay(p, y, threshold=0.5, consecutive=3)
    # first 3-consecutive confirmation completes at row 8
    assert d["alarm_decision_row_in_window"] == 8
    assert d["delay_rows"] == -2
    # the alarm run stays active through onset -> no cleared episode
    assert d["cleared_alarm_episodes_before_onset"] == 0


def test_delay_counts_cleared_episodes():
    # a sustained run at rows 2-5 that clears, then a run active at onset
    y = np.array([0] * 15 + [1] * 5)
    p = np.zeros(20)
    p[2:6] = 0.9   # cleared episode (>=3 consecutive, then retracts)
    p[10:] = 0.9   # run still active at onset
    d = detection_delay(p, y, threshold=0.5, consecutive=3)
    assert d["alarm_decision_row_in_window"] == 4  # first confirmation is in the early run
    assert d["cleared_alarm_episodes_before_onset"] == 1
    # early run too short to count as an episode
    p2 = np.zeros(20)
    p2[2:4] = 0.9  # only 2 consecutive -> no episode
    p2[10:] = 0.9
    d2 = detection_delay(p2, y, threshold=0.5, consecutive=3)
    assert d2["alarm_decision_row_in_window"] == 12
    assert d2["cleared_alarm_episodes_before_onset"] == 0


def test_delay_no_alarm():
    y = np.array([0] * 5 + [1] * 5)
    p = np.full(10, 0.1)
    d = detection_delay(p, y, threshold=0.5, consecutive=3)
    assert d["alarm_decision_row_in_window"] is None
    assert d["delay_rows"] is None


# ------------------------------------------------------ anticipation profile
def test_anticipation_excludes_only_pre_onset_band():
    y = np.array([0] * 10 + [1] * 10)
    p = np.zeros(20)
    p[7:] = 0.9  # flags 3 healthy rows immediately before onset + all clogged
    prof = anticipation_profile(p, y, threshold=0.5, windows=(0, 3))
    strict, w3 = prof[0], prof[1]
    assert strict["precision"] < 1.0
    assert w3["n_excluded"] == 3
    assert w3["precision"] == 1.0
    assert w3["recall"] == strict["recall"] == 1.0


# ---------------------------------------------------------------- thresholds
def test_fpr_threshold_flags_at_most_budget():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.4, 600), rng.uniform(0.6, 1.0, 100)])
    y = np.array([0] * 600 + [1] * 100)
    thr = select_threshold_fpr(p, y, max_fpr=0.01)
    fpr = (p[y == 0] >= thr).mean()
    assert fpr <= 0.01
    assert fpr == pytest.approx(6 / 600)  # exactly floor(600*0.01) flagged


def test_max_f1_threshold_perfect_separation():
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1])
    thr = select_threshold_max_f1(p, y)
    pred = (p >= thr).astype(int)
    assert (pred == y).all()


# ---------------------------------------------------------------- calibrator
def test_platt_round_trip_via_coef():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.99, 500)
    y = (rng.uniform(size=500) < p).astype(int)
    cal = PlattCalibrator().fit(p, y)
    rebuilt = PlattCalibrator.from_coef(*cal.coef())
    np.testing.assert_allclose(cal.transform(p), rebuilt.transform(p), rtol=1e-10)


# ------------------------------------------------------------- segment folds
def test_expanding_folds_two_segments():
    # two contiguous segments: 0..999 (healthy) and 2000..2499 (clogged)
    idx = np.concatenate([np.arange(1000), np.arange(2000, 2500)])
    y = np.zeros(2500, dtype=int)
    y[2000:] = 1
    folds = expanding_window_folds(idx, y, n_splits=4, gap=20)
    assert len(folds) >= 2
    for fit_idx, val_idx in folds:
        # both classes in every fit set, positives in every val block
        assert len(np.unique(y[fit_idx])) == 2
        assert (y[val_idx] == 1).sum() > 0
        # no overlap, and within each segment fit strictly precedes val
        assert len(np.intersect1d(fit_idx, val_idx)) == 0
        for seg_lo, seg_hi in ((0, 999), (2000, 2499)):
            f = fit_idx[(fit_idx >= seg_lo) & (fit_idx <= seg_hi)]
            v = val_idx[(val_idx >= seg_lo) & (val_idx <= seg_hi)]
            if len(f) and len(v):
                assert f.max() < v.min()
                assert v.min() - f.max() - 1 >= 20  # purge gap inside segment
