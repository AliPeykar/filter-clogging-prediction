"""Paper analysis A9: residual leakage routes in the corrected pipeline.

Review round 3, comment M5. The corrected pipeline (run_experiment.py,
configs/main.yaml) has two routes by which signal from the test window, or
from after it, reaches the training features:

  (a) normalization constants (means, SDs, maximum flow, baseline
      resistance) are fitted on the training set, which includes Train-B
      (samples 10,321-11,519, after the test window);
  (b) recursive features (EMAs, one-sided CUSUM) are computed once over the
      whole series, so Val-B and Train-B carry state accumulated through the
      test window.

A third point concerns feature 40, dp_cusum_pos, which never resets and so
accumulates with position in the series.

Variants (all else as in the main configuration: seed 42, 75 TPE trials,
monotone constraints, Platt scaling, max-F1 threshold on validation):

  V0 main       - control; must reproduce the main configuration on this machine
  V1 norm_A     - normalization constants fitted on Train-A only
  V2 restart    - recursive and rolling features restarted after the test
                  window: samples 1-9,600 are computed as one block (as in
                  deployment), Val-B from sample 9,601 and Train-B from
                  sample 10,261, i.e. from the start of their 60-sample purge
                  gaps, which serve as warm-up and are never scored
  V3 no_cusum   - dp_cusum_pos removed (39 features)
  V4 all        - V1 + V2 + V3

Run from paper/:  python 03_analysis/a9_residual_leakage.py
Needs the pipeline package (07_repository/src, or src/ at the repository
root). Output: 03_analysis/out/a9_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
for cand in (PAPER.parent / "src", PAPER / "07_repository" / "src"):
    if (cand / "filterclog").exists():
        sys.path.insert(0, str(cand))
        DATA = cand.parent / "Comprehensive_Filter_Analysis.xlsx"
        break

from filterclog.calibration import PlattCalibrator, select_threshold_max_f1  # noqa: E402
from filterclog.config import SplitConfig  # noqa: E402
from filterclog.data import load_dataset  # noqa: E402
from filterclog.evaluation import point_metrics  # noqa: E402
from filterclog.features import CausalFeatureEngineer  # noqa: E402
from filterclog.hpo import optimize_xgb  # noqa: E402
from filterclog.splitting import make_splits  # noqa: E402
from filterclog.target import make_binary_target  # noqa: E402
from filterclog.train import train_rf_baseline, train_xgb  # noqa: E402

SEED = 42
ONSET, STEP = 8943, 8340
TEST = (7861, 9600)
# 0-based starts of the blocks for V2: 1-9,600 | 9,601-10,260 | 10,261-11,519
RESTART_BLOCKS = [(0, 9600), (9600, 10260), (10260, 11519)]


def features(df, eng, restart: bool) -> pd.DataFrame:
    if not restart:
        return eng.transform(df)
    parts = [eng.transform(df.iloc[a:b]) for a, b in RESTART_BLOCKS]
    return pd.concat(parts)


def events(pred: np.ndarray, rows: np.ndarray) -> dict:
    """First flag and false positives relative to the rule onset."""
    flagged = rows[pred == 1]
    first = int(flagged[0]) - ONSET if len(flagged) else None
    fp_rows = rows[(pred == 1) & (rows < ONSET)]
    return {"first_flag_minus_onset": first,
            "false_positives": int(len(fp_rows)),
            "false_positives_before_step": int((fp_rows < STEP).sum())}


def run_variant(name, df, y, split, norm_rows, restart, drop_cusum) -> dict:
    eng = CausalFeatureEngineer("leakage_free", (5, 15, 30), (10, 50), 0.5)
    eng.fit(df.iloc[norm_rows])
    Xf = features(df, eng, restart)
    if drop_cusum:
        Xf = Xf.drop(columns=["dp_cusum_pos"])
    names = list(Xf.columns)
    X = Xf.to_numpy(dtype=np.float32)
    mono = eng.monotone_constraints(names)
    warm = eng.warmup_rows

    def usable(idx):
        # as run_experiment.py: drop warm-up rows only; remaining NaN (e.g.
        # dp_rate where dp is zero) are handled by the models
        return idx[idx >= warm]

    tr, va, te = usable(split.train), usable(split.val), usable(split.test)
    nan_rows = [int(np.isnan(X[i]).any(axis=1).sum()) for i in (tr, va, te)]
    hpo = optimize_xgb(X, y, tr, va, n_trials=75, n_cv_splits=4, cv_gap=60, seed=SEED,
                       max_n_estimators=2000, early_stopping_rounds=50, monotone=mono)
    model = train_xgb(hpo["best_params"], X[tr], y[tr], X[va], y[va], seed=SEED,
                      max_n_estimators=2000, early_stopping_rounds=50, monotone=mono)
    pv, pt = model.predict_proba(X[va])[:, 1], model.predict_proba(X[te])[:, 1]
    cal = PlattCalibrator().fit(pv, y[va])
    thr = select_threshold_max_f1(cal.transform(pv), y[va])
    m_x = point_metrics(y[te], cal.transform(pt), thr)

    rf = train_rf_baseline(X[tr], y[tr], SEED)
    pv_rf, pt_rf = rf.predict_proba(X[va])[:, 1], rf.predict_proba(X[te])[:, 1]
    cal_rf = PlattCalibrator().fit(pv_rf, y[va])
    thr_rf = select_threshold_max_f1(cal_rf.transform(pv_rf), y[va])
    m_r = point_metrics(y[te], cal_rf.transform(pt_rf), thr_rf)

    rows = te + 1
    keep = ("roc_auc", "pr_auc", "precision", "recall", "f1", "confusion_matrix")
    out = {
        "n_features": len(names),
        "n_train_val_test": [len(tr), len(va), len(te)],
        "rows_with_nan_train_val_test": nan_rows,
        "xgb": {k: m_x[k] for k in keep} | events((cal.transform(pt) >= thr).astype(int), rows)
               | {"n_trees": int(getattr(model, "best_iteration", 0)) + 1},
        "rf": {k: m_r[k] for k in keep} | events((cal_rf.transform(pt_rf) >= thr_rf).astype(int), rows),
    }
    for mdl in ("xgb", "rf"):
        for k in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
            out[mdl][k] = round(out[mdl][k], 4)
    print(name, json.dumps(out), flush=True)
    return out


def main() -> None:
    df = load_dataset(DATA, "Comprehensive_Analysis")
    y = make_binary_target(df)
    split = make_splits(len(df), SplitConfig(scheme="held_out_transition"))
    train_a = np.arange(0, 7140)
    variants = {
        "V0_main": (split.train, False, False),
        "V1_norm_trainA": (train_a, False, False),
        "V2_restart_after_test": (split.train, True, False),
        "V3_no_cusum": (split.train, False, True),
        "V4_all": (train_a, True, True),
    }
    res = {"data": DATA.name, "seed": SEED, "python": sys.version.split()[0],
           "restart_blocks_1based": [[a + 1, b] for a, b in RESTART_BLOCKS]}
    for name, (norm_rows, restart, drop) in variants.items():
        res[name] = run_variant(name, df, y, split, norm_rows, restart, drop)
    (HERE / "out" / "a9_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
