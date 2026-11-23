"""Paper analysis A1: label structure, split support, and simple baselines.

Read-only with respect to the project: imports the final ``filterclog``
package and the dataset, reads the re-run main result, writes only to
``paper/03_analysis/out/``.

Usage:  python a1_baselines_label_structure.py <main_run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filterclog.calibration import PlattCalibrator, select_threshold_max_f1  # noqa: E402
from filterclog.config import load_config  # noqa: E402
from filterclog.data import load_dataset, validate_dataset  # noqa: E402
from filterclog.evaluation import (  # noqa: E402
    anticipation_profile, block_bootstrap_ci, detection_delay,
    expected_calibration_error, point_metrics,
)
from filterclog.features import CausalFeatureEngineer  # noqa: E402
from filterclog.splitting import make_splits  # noqa: E402
from filterclog.target import make_binary_target  # noqa: E402
from filterclog.train import train_rf_baseline  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
SEED, BLOCK, NBOOT = 42, 50, 1000


def boot_indices(n: int, block: int, n_resamples: int, seed: int):
    """Same moving-block resampling scheme as evaluation.block_bootstrap_ci."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    for _ in range(n_resamples):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        yield np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def auc_ci_and_diff(y, score_a, score_b=None):
    """Block-bootstrap 95% CI of ROC-AUC(a) and, if given, of AUC(a)-AUC(b)."""
    a, d = [], []
    for idx in boot_indices(len(y), BLOCK, NBOOT, SEED):
        if len(np.unique(y[idx])) < 2:
            continue
        va = roc_auc_score(y[idx], score_a[idx])
        a.append(va)
        if score_b is not None:
            d.append(va - roc_auc_score(y[idx], score_b[idx]))
    out = {"roc_auc_ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
    if score_b is not None:
        out["auc_diff_mean"] = float(np.mean(d))
        out["auc_diff_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        out["auc_diff_frac_positive"] = float(np.mean(np.array(d) > 0))
    return out


def thr_metrics(y, score, thr):
    pred = (score >= thr).astype(int)
    onset = int(np.flatnonzero(y == 1)[0])
    pos = np.flatnonzero(pred == 1)
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "first_flag_minus_onset_rows": None if len(pos) == 0 else int(pos[0] - onset),
    }


def main(run_dir: str) -> None:
    run = Path(run_dir).resolve()
    cfg = load_config(run / "config.yaml")
    with open(run / "metrics.json", encoding="utf-8") as fh:
        metrics = json.load(fh)

    df = load_dataset(ROOT / cfg.data_path, cfg.sheet_name)
    validate_dataset(df)
    y = make_binary_target(df)
    dp = df["dp"].to_numpy(float)
    q = df["flowrate"].to_numpy(float)
    score = df["validation_score"].to_numpy()
    res: dict = {"source_run": str(run.relative_to(ROOT))}

    # ------------------------------------------------ 1. label structure
    ch = np.flatnonzero(np.diff(score) != 0) + 1  # 0-based index of new value
    steps = [
        {"row": int(c + 1), "score_from": int(score[c - 1]), "score_to": int(score[c]),
         "dp_before": float(dp[c - 1]), "dp_at": float(dp[c]),
         "dp_changed": bool(dp[c] != dp[c - 1])}
        for c in ch
    ]
    pd.DataFrame(steps).to_csv(OUT / "score_steps.csv", index=False)
    onset_row = int(np.flatnonzero(y == 1)[0] + 1)  # 1-based
    # last large single-row dp increase before the label onset
    d_dp = np.diff(dp)
    pre = np.arange(7860, onset_row - 1)  # test-window rows before onset
    j = pre[np.argmax(d_dp[pre - 1])]  # 0-based index of the row after the jump
    res["label_structure"] = {
        "n_rows": int(len(df)),
        "n_clogged": int(y.sum()),
        "onset_row_1based": onset_row,
        "n_score_steps": int(len(ch)),
        "score_values_when_clogged": sorted(int(v) for v in np.unique(score[y == 1])),
        "score_max_when_healthy": int(score[y == 0].max()),
        "rows_with_score_ge_12": int((score >= 12).sum()),
        "dp_at_onset_minus1_and_onset": [float(dp[onset_row - 2]), float(dp[onset_row - 1])],
        "score_at_onset_minus1_and_onset": [int(score[onset_row - 2]), int(score[onset_row - 1])],
        "steps_without_dp_change": [s for s in steps if not s["dp_changed"] and s["row"] >= 4000],
        "largest_pre_onset_dp_jump_in_test_window": {
            "row_1based": int(j + 1), "dp_before": float(dp[j - 1]), "dp_after": float(dp[j]),
            "rows_from_jump_to_label_onset": int(onset_row - (j + 1)),
        },
        "dp_between_jump_and_onset_min_max": [float(dp[j:onset_row - 1].min()),
                                              float(dp[j:onset_row - 1].max())],
        "dp_before_jump_in_test_window_min_max": [float(dp[7860:j].min()), float(dp[7860:j].max())],
    }

    # ------------------------------------------------ 2. split support
    split = make_splits(len(df), cfg.split)
    seg_rows = {k: np.arange(lo - 1, hi) for k, (lo, hi) in split.segments.items()}
    support = {}
    for k, idx in seg_rows.items():
        support[k] = {
            "rows_1based": [int(idx[0] + 1), int(idx[-1] + 1)], "n": int(len(idx)),
            "n_clogged": int(y[idx].sum()),
            "dp_min": float(dp[idx].min()), "dp_max": float(dp[idx].max()),
            "q_median": float(np.median(q[idx])),
            "score_min": int(score[idx].min()), "score_max": int(score[idx].max()),
        }
    fit_idx = np.concatenate([split.train, split.val])
    lo_gap = float(dp[fit_idx][y[fit_idx] == 0].max())
    hi_gap = float(dp[fit_idx][y[fit_idx] == 1].min())
    te = split.test
    in_gap = (dp[te] > lo_gap) & (dp[te] < hi_gap)
    res["split_support"] = {
        "segments": support,
        "dp_gap_unobserved_in_train_and_val": [lo_gap, hi_gap],
        "train_only_gap": [float(dp[split.train][y[split.train] == 0].max()),
                           float(dp[split.train][y[split.train] == 1].min())],
        "n_train_val_rows_inside_gap": int(((dp[fit_idx] > lo_gap) & (dp[fit_idx] < hi_gap)).sum()),
        "test_rows_inside_gap": int(in_gap.sum()),
        "test_rows_inside_gap_frac": float(in_gap.mean()),
        "test_clogged_inside_gap": int((in_gap & (y[te] == 1)).sum()),
        "test_healthy_inside_gap": int((in_gap & (y[te] == 0)).sum()),
        "test_dp_healthy_min_max": [float(dp[te][y[te] == 0].min()), float(dp[te][y[te] == 0].max())],
        "test_dp_clogged_min_max": [float(dp[te][y[te] == 1].min()), float(dp[te][y[te] == 1].max())],
    }

    # --------------------------------------- 3. models from the main run
    eng = CausalFeatureEngineer(cfg.feature_set, cfg.rolling_windows, cfg.ema_spans, cfg.cusum_k)
    eng.fit(df.iloc[split.train])
    Xf = eng.transform(df)
    assert list(Xf.columns) == metrics["feature_names"]
    X = Xf.to_numpy(dtype=np.float32)
    warm = eng.warmup_rows
    tr, va = split.train[split.train >= warm], split.val
    yt, yv = y[te], y[va]
    res["effective_sizes"] = {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                              "warmup_rows_dropped": int(warm)}

    xgb = XGBClassifier()
    xgb.load_model(run / "xgb_h0.json")
    p_val_raw = xgb.predict_proba(X[va])[:, 1]
    p_te_raw = xgb.predict_proba(X[te])[:, 1]
    cal = PlattCalibrator().fit(p_val_raw, yv)
    p_te = cal.transform(p_te_raw)
    thr = select_threshold_max_f1(cal.transform(p_val_raw), yv)
    stored = metrics["horizons"]["h0"]["xgb"]
    chk = point_metrics(yt, p_te, thr)
    assert abs(chk["f1"] - stored["test_metrics"]["f1"]) < 1e-9, "reloaded XGB != stored run"
    assert abs(chk["roc_auc"] - stored["test_metrics"]["roc_auc"]) < 1e-9

    rf = train_rf_baseline(X[tr], y[tr], SEED)
    pr_val_raw, pr_te_raw = rf.predict_proba(X[va])[:, 1], rf.predict_proba(X[te])[:, 1]
    cal_rf = PlattCalibrator().fit(pr_val_raw, yv)
    pr_te = cal_rf.transform(pr_te_raw)
    thr_rf = select_threshold_max_f1(cal_rf.transform(pr_val_raw), yv)
    chk_rf = point_metrics(yt, pr_te, thr_rf)
    assert abs(chk_rf["f1"] - metrics["horizons"]["h0"]["rf"]["test_metrics"]["f1"]) < 1e-9

    pred_x, pred_r = (p_te >= thr).astype(int), (pr_te >= thr_rf).astype(int)
    onset_w = int(np.flatnonzero(yt == 1)[0])
    fp_rows = te[(pred_x == 1) & (yt == 0)] + 1
    fn_rf = te[(pred_r == 0) & (yt == 1)] + 1
    res["model_behaviour"] = {
        "xgb": {
            "threshold": float(thr), "f1": chk["f1"], "roc_auc": chk["roc_auc"],
            "pr_auc": chk["pr_auc"],
            "n_fp": int(len(fp_rows)),
            "fp_rows_min_max_1based": [int(fp_rows.min()), int(fp_rows.max())],
            "fp_rows_after_dp_jump": int((fp_rows >= j + 1).sum()),
            "fp_rows_before_dp_jump": int((fp_rows < j + 1).sum()),
            "fp_dp_min_max": [float(dp[fp_rows - 1].min()), float(dp[fp_rows - 1].max())],
            "pred_pos_dp_min": float(dp[te][pred_x == 1].min()),
            "pred_neg_dp_max": float(dp[te][pred_x == 0].max()),
            "first_sustained_alarm_row_1based_k3": int(
                te[detection_delay(p_te, yt, thr, 3)["alarm_decision_row_in_window"]] + 1),
            "ece_raw_test": float(expected_calibration_error(yt, p_te_raw)),
            "ece_platt_test": float(expected_calibration_error(yt, p_te)),
            "brier_raw_test": float(np.mean((p_te_raw - yt) ** 2)),
            "brier_platt_test": float(np.mean((p_te - yt) ** 2)),
            "f1_at_raw_0p5": float(f1_score(yt, (p_te_raw >= 0.5).astype(int))),
        },
        "rf": {
            "threshold": float(thr_rf), "f1": chk_rf["f1"], "roc_auc": chk_rf["roc_auc"],
            "pr_auc": chk_rf["pr_auc"],
            "n_fn": int(len(fn_rf)),
            "fn_rows_min_max_1based": [int(fn_rf.min()), int(fn_rf.max())] if len(fn_rf) else None,
            "fn_dp_min_max": [float(dp[fn_rf - 1].min()), float(dp[fn_rf - 1].max())] if len(fn_rf) else None,
            "first_flag_minus_onset_rows": int(np.flatnonzero(pred_r == 1)[0] - onset_w),
            "pred_pos_dp_min": float(dp[te][pred_r == 1].min()),
            "pred_neg_dp_max": float(dp[te][pred_r == 0].max()),
            "ece_raw_test": float(expected_calibration_error(yt, pr_te_raw)),
            "ece_platt_test": float(expected_calibration_error(yt, pr_te)),
            "f1_at_raw_0p5": float(f1_score(yt, (pr_te_raw >= 0.5).astype(int))),
        },
    }

    # ------------------------------------------- 4. univariate baselines
    ema50 = pd.Series(dp).ewm(span=50, adjust=False).mean().to_numpy()
    rmean30 = pd.Series(dp).rolling(30, min_periods=30).mean().to_numpy()
    resist = dp / np.clip(q, 0.5, None)
    signals = {"dp": dp, "dp_rmean_30": rmean30, "dp_ema_50": ema50,
               "resistance_dp_over_q": resist, "neg_flowrate": -q}
    base = {}
    for name, s in signals.items():
        sv, st = s[va], s[te]
        hmax, cmin = float(sv[yv == 0].max()), float(sv[yv == 1].min())
        entry = {
            "roc_auc": float(roc_auc_score(yt, st)),
            "pr_auc": float(average_precision_score(yt, st)),
            "val_healthy_max": hmax, "val_clogged_min": cmin,
            "val_separable": bool(hmax < cmin),
        }
        entry.update(auc_ci_and_diff(yt, st, p_te))
        entry["auc_diff_is"] = "baseline minus XGBoost"
        rf_cmp = auc_ci_and_diff(yt, st, pr_te)
        entry["auc_diff_vs_rf_mean"] = rf_cmp["auc_diff_mean"]
        entry["auc_diff_vs_rf_ci"] = rf_cmp["auc_diff_ci"]
        if hmax < cmin:
            rules = {
                # the pipeline's own rule applied to a 1-D score: first max-F1
                # threshold on validation = smallest clogged validation value
                "val_max_f1_first": select_threshold_max_f1(sv, yv),
                "val_midpoint": 0.5 * (hmax + cmin),
                "val_healthy_max_plus": hmax + 1e-9,
            }
            entry["thresholds"] = {r: thr_metrics(yt, st, t) for r, t in rules.items()}
        base[name] = entry
    res["univariate_baselines"] = base

    # threshold sweep of raw dp across (and slightly beyond) the unobserved gap
    grid = np.round(np.arange(17.0, 30.01, 0.25), 2)
    sweep = [thr_metrics(yt, dp[te], t) for t in grid]
    pd.DataFrame(sweep).to_csv(OUT / "dp_threshold_sweep.csv", index=False)
    inside = [s for s in sweep if lo_gap < s["threshold"] < hi_gap]
    res["dp_threshold_sweep_inside_gap"] = {
        "n_thresholds": len(inside),
        "f1_min": min(s["f1"] for s in inside), "f1_max": max(s["f1"] for s in inside),
        "f1_argmax_threshold": max(inside, key=lambda s: s["f1"])["threshold"],
        "first_flag_minus_onset_min": min(s["first_flag_minus_onset_rows"] for s in inside),
        "first_flag_minus_onset_max": max(s["first_flag_minus_onset_rows"] for s in inside),
    }

    # ---------------------------------- 5. logistic-regression baselines
    def lr_baseline(cols):
        A = np.column_stack(cols)
        mu, sd = A[tr].mean(0), A[tr].std(0)
        Z = (A - mu) / sd
        m = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Z[tr], y[tr])
        pv, pt = m.predict_proba(Z[va])[:, 1], m.predict_proba(Z[te])[:, 1]
        c = PlattCalibrator().fit(pv, yv)
        t = select_threshold_max_f1(c.transform(pv), yv)
        pm = point_metrics(yt, c.transform(pt), t)
        out = {k: pm[k] for k in ("roc_auc", "pr_auc", "f1", "precision", "recall",
                                  "confusion_matrix", "threshold")}
        out["coef_standardised"] = [float(v) for v in m.coef_[0]]
        out["f1_at_raw_0p5"] = float(f1_score(yt, (pt >= 0.5).astype(int)))
        pos = np.flatnonzero(pt >= 0.5)
        out["first_flag_minus_onset_rows_raw_0p5"] = int(pos[0] - onset_w) if len(pos) else None
        out["boundary_dp_raw_0p5_single_feature"] = (
            float(mu[0] - sd[0] * m.intercept_[0] / m.coef_[0, 0]) if A.shape[1] == 1 else None)
        return out

    res["logistic_baselines"] = {
        "lr_dp_only": lr_baseline([dp]),
        "lr_dp_and_q": lr_baseline([dp, q]),
    }

    # -------------------------- 6. sensitivity of scores to onset placement
    # relabel the onset at the dp jump (row j+1) instead of the rule's onset
    y_alt = y.copy()
    y_alt[j:onset_row - 1] = 1
    yt_alt = y_alt[te]
    alt = {}
    for name, (p, t) in {"xgb": (p_te, thr), "rf": (pr_te, thr_rf)}.items():
        pred = (p >= t).astype(int)
        alt[name] = {
            "f1_rule_onset": float(f1_score(yt, pred)),
            "f1_onset_at_dp_jump": float(f1_score(yt_alt, pred)),
            "roc_auc_rule_onset": float(roc_auc_score(yt, p)),
            "roc_auc_onset_at_dp_jump": float(roc_auc_score(yt_alt, p)),
        }
    pred_mid = (dp[te] >= base["dp"]["thresholds"]["val_midpoint"]["threshold"]).astype(int)
    alt["dp_val_midpoint"] = {
        "f1_rule_onset": float(f1_score(yt, pred_mid)),
        "f1_onset_at_dp_jump": float(f1_score(yt_alt, pred_mid)),
        "roc_auc_rule_onset": float(roc_auc_score(yt, dp[te])),
        "roc_auc_onset_at_dp_jump": float(roc_auc_score(yt_alt, dp[te])),
    }
    res["onset_placement_sensitivity"] = {
        "alt_onset_row_1based": int(j + 1), "rule_onset_row_1based": onset_row,
        "n_rows_relabelled": int(onset_row - 1 - j), "scores": alt,
    }
    res["anticipation_profile_rf"] = anticipation_profile(pr_te, yt, thr_rf)
    res["anticipation_profile_xgb"] = anticipation_profile(p_te, yt, thr)
    res["xgb_block_bootstrap"] = block_bootstrap_ci(yt, p_te, thr, BLOCK, NBOOT, SEED)
    res["rf_block_bootstrap"] = block_bootstrap_ci(yt, pr_te, thr_rf, BLOCK, NBOOT, SEED)

    # per-row dump for figures
    pd.DataFrame({
        "row": te + 1, "y": yt, "dp": dp[te], "flowrate": q[te], "score": score[te],
        "p_xgb_raw": p_te_raw, "p_xgb": p_te, "p_rf_raw": pr_te_raw, "p_rf": pr_te,
    }).to_csv(OUT / "test_window_predictions.csv", index=False)

    with open(OUT / "a1_results.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
