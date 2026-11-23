# Verification Checklist — executed 2026-07-02

All items executed against the corrected pipeline; commands and observed
results recorded.

| # | Check | Method | Result |
|---|---|---|---|
| 1 | Causality (no future data in any feature) | `tests/test_causality.py`: truncation equivalence — `transform(df[:t+1]).iloc[-1] == transform(df).iloc[t]` at random t; single code path for train/predict; warm-up rows NaN not zero-filled | **PASS** (pytest, 4 tests) |
| 2 | Split integrity | `tests/test_splits.py`: no index overlap, purge gaps ≥ 60 rows, S2 composition (train 8,339 @ 14.4% pos / val 1,200 balanced / test 1,740 = 1,082 healthy + 658 clogged with onset inside), S1 one-class test documented | **PASS** (pytest, 4 tests) |
| 3 | Banned-feature guard | `tests/test_causality.py::test_banned_columns_raise`: pipeline raises if any label-constituent column enters the leakage-free design matrix | **PASS** |
| 4 | Target-shuffle audit | Block-permuted training labels (block=200, 5 replicates), best-run hyperparameters, evaluated on true test labels. Expectation: skill collapses to base rate (0.378) | **PASS** — PR-AUC 0.268 ± 0.009, ROC-AUC 0.285 ± 0.037 (true-label run: 0.927 / 0.969) |
| 5 | Circularity ablation | `configs/ablation_thesis_features.yaml` (adds `validation_score`, `adaptive_cusum`, `cusum_threshold`, precomputed EMAs) | **PASS** — reproduces the legacy pathology: ROC-AUC 1.0000, F1 0.9992, PR-AUC 1.0000 (vs 0.969 / 0.706 / 0.927 leakage-free) |
| 6 | Reproducibility | `python run_experiment.py --config configs/main.yaml` run twice; full `metrics.json` compared | **PASS** — bit-identical |
| 7 | Sampling interval | `verify_sampling_interval()` archived in every `metrics.json`; ~3.6 s/row derived from the thesis's 11.5 h experiment duration; time column confirmed to be a bare row counter | **PASS** (recorded, assumption documented) |
| 8 | Sanity | Leakage-free headline metrics strictly < 1.0; sustained alarm (3-consecutive, credited at the k-th confirmation row) fires 598 rows = ~35.8 min *before* the true onset; 5 sustained episodes inside the pre-onset anticipation band raised and cleared again before onset (alarm flapping — these lie at/after the first alarm, not earlier in the window); label-transition facts (row 8,943, 77.6/22.4) asserted in `tests/test_target.py` | **PASS** |

Full unit-test suite: `python -m pytest tests/` → **21 passed**.

## Independent code review (executed)

A reviewer agent audited the full package against the plan. Verdict: core
machinery (causal features, split geometry, leakage guards, val-only tuning,
reproducibility) correct. All findings were fixed and all runs regenerated:

* **Critical 1** — S1 reuse applied S2's calibrated-space thresholds to raw
  probabilities. Fixed: Platt coefficients persisted in `metrics.json` and
  reapplied in the reuse path (verified in the regenerated strict-temporal
  run).
* **Critical 2** — "false-alarm episodes" counter contradicted its spec and
  was misquoted in this checklist. Fixed: renamed/rescoped to
  `cleared_alarm_episodes_before_onset` (sustained runs that raise and clear
  again pre-onset), unit-tested.
* Important — alarm now credited at the k-th confirmation row (decision
  time); FPR≤1% threshold carries an explicit non-transfer caveat in the
  delay-sensitivity output; per-horizon figure filenames (a variable-shadowing
  bug also saved one figure under a curve-label name); risk-band occupancy
  counted on training rows only, with a trainability guard; Platt refit on
  logits with bounded C; statistics/CV/calibration unit tests added (+8).
* Minor — FPR selector off-by-one, bootstrap edge guards, `pct_change`
  future-warning, hardcoded onset row in figures, strict-split config
  validation, docstring fixes.

## Headline results (S2 held-out-transition, leakage-free, calibrated, val-selected threshold)

| Metric | XGBoost (primary) | RF baseline |
|---|---|---|
| ROC-AUC | 0.969 | 0.976 |
| PR-AUC | 0.927 | 0.969 |
| F1 (pointwise) | 0.706 | 0.944 |
| F1 (anticipation window 600 rows ≈ 36 min) | 0.999 | 0.944 |
| Recall | 0.998 | 0.894 |
| Detection delay | −35.8 min (early) | late (misses first ~70 clogged rows) |

Interpretation for the paper: the two models make *complementary* errors —
XGBoost anticipates the rule's onset by ~36 minutes (all of its pointwise
false positives lie in one contiguous pre-onset band), while the RF baseline
mirrors the rule more literally but detects late. Under pointwise scoring RF
looks better; under anticipation-aware scoring XGBoost dominates. Both views
are reported.

## Standing limitations (must appear in the paper)

1. Single filter lifecycle; cross-filter generalization untested.
2. Labels come from the thesis's deterministic rule, not independent ground
   truth; supervised results measure causal-feature emulation/anticipation of
   that rule.
3. S2's clogged training segment post-dates the test window (interpolation
   across a held-out transition, not pure forecasting).
4. Validation block reused for HPO, early stopping, calibration, and
   threshold; detection delay has n_events = 1 (no CI).
5. Thesis text says clogged ⟺ score ≥ 12; stored data uses ≥ 11.
