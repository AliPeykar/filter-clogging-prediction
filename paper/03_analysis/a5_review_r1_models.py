"""Revision analyses for review round 1 (comments M2, M5, M9).

(1) Search-space fragility: per-trial validation log-loss versus test-window
    scores for the narrow (published) search space and for a wide space; a
    fixed untuned model on the corrected split.
(2) Decomposition of mechanism M2 by group of rule-derived columns.
(3) Strictly causal, label-free baselines calibrated on the healthy head only.

The test window is used here ONLY to describe what the audit asks about
(how selection on validation relates to test scores); nothing is selected on
it. Writes paper/03_analysis/out/a5_results.json and trial CSVs.
Usage: python a5_review_r1_models.py <main_run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filterclog.calibration import PlattCalibrator, select_threshold_max_f1  # noqa: E402
from filterclog.config import load_config  # noqa: E402
from filterclog.data import load_dataset  # noqa: E402
from filterclog.evaluation import detection_delay  # noqa: E402
from filterclog.features import CausalFeatureEngineer  # noqa: E402
from filterclog.hpo import _make_model, suggest_params  # noqa: E402
from filterclog.splitting import make_splits  # noqa: E402
from filterclog.target import make_binary_target  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
SEED = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)


def wide_params(trial: optuna.Trial) -> dict:
    """A deliberately unrestricted space (re-created for this audit; the
    original wide space of the project was not archived)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
    }


def score(model, X, y, va, te) -> dict:
    pv, pt = model.predict_proba(X[va])[:, 1], model.predict_proba(X[te])[:, 1]
    cal = PlattCalibrator().fit(pv, y[va])
    thr = select_threshold_max_f1(cal.transform(pv), y[va])
    pred = (cal.transform(pt) >= thr).astype(int)
    return {
        "val_logloss": float(log_loss(y[va], pv, labels=[0, 1])),
        "test_roc_auc": float(roc_auc_score(y[te], pt)),
        "test_f1": float(f1_score(y[te], pred, zero_division=0)),
        "test_precision": float(precision_score(y[te], pred, zero_division=0)),
        "test_recall": float(recall_score(y[te], pred, zero_division=0)),
    }


def search(space, name, X, y, tr, va, te, spw, mono, n_trials=75) -> dict:
    rows = []

    def objective(trial):
        params = space(trial)
        m = _make_model(params, spw, SEED, 2000, 50, mono)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        s = score(m, X, y, va, te)
        rows.append({"trial": trial.number, **params, "n_trees": int(m.best_iteration) + 1, **s})
        return s["val_logloss"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"search_trials_{name}.csv", index=False)
    best = d.loc[d["val_logloss"].idxmin()]
    rho = float(d[["val_logloss", "test_roc_auc"]].corr(method="spearman").iloc[0, 1])
    return {
        "n_trials": int(len(d)),
        "best_by_validation": {k: (float(v) if not isinstance(v, (int, np.integer)) else int(v))
                               for k, v in best.items()},
        "test_roc_auc_min_median_max": [float(d.test_roc_auc.min()), float(d.test_roc_auc.median()),
                                        float(d.test_roc_auc.max())],
        "test_f1_min_median_max": [float(d.test_f1.min()), float(d.test_f1.median()), float(d.test_f1.max())],
        "val_logloss_min_median_max": [float(d.val_logloss.min()), float(d.val_logloss.median()),
                                       float(d.val_logloss.max())],
        "spearman_val_logloss_vs_test_auc": rho,
        "frac_trials_test_auc_below_0p9": float((d.test_roc_auc < 0.9).mean()),
        "frac_trials_test_auc_ge_raw_dp_0p9927": float((d.test_roc_auc >= 0.9927).mean()),
        "n_trials_val_loss_below_narrow_best": None,
    }


def main(run_dir: str) -> None:
    run = Path(run_dir).resolve()
    cfg = load_config(run / "config.yaml")
    main_metrics = json.loads((run / "metrics.json").read_text())
    df = load_dataset(ROOT / cfg.data_path, cfg.sheet_name)
    y = make_binary_target(df)
    split = make_splits(len(df), cfg.split)
    eng = CausalFeatureEngineer("leakage_free", cfg.rolling_windows, cfg.ema_spans, cfg.cusum_k)
    eng.fit(df.iloc[split.train])
    Xf = eng.transform(df)
    names = list(Xf.columns)
    X = Xf.to_numpy(dtype=np.float32)
    warm = eng.warmup_rows
    tr, va, te = split.train[split.train >= warm], split.val, split.test
    spw = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
    mono = eng.monotone_constraints(names)
    res: dict = {}

    # ------------------------------------------------ (1) search-space fragility
    narrow = search(suggest_params, "narrow", X, y, tr, va, te, spw, mono)
    stored = main_metrics["horizons"]["h0"]["xgb"]
    assert abs(narrow["best_by_validation"]["test_roc_auc"] - stored["test_metrics"]["roc_auc"]) < 1e-6, \
        "narrow search did not reproduce the main run"
    wide = search(wide_params, "wide", X, y, tr, va, te, spw, mono)
    dn = pd.read_csv(OUT / "search_trials_narrow.csv")
    dw = pd.read_csv(OUT / "search_trials_wide.csv")
    wide["n_trials_val_loss_below_narrow_best"] = int((dw.val_logloss < dn.val_logloss.min()).sum())
    wide["test_auc_of_trials_with_val_loss_below_narrow_best"] = [
        float(v) for v in sorted(dw.loc[dw.val_logloss < dn.val_logloss.min(), "test_roc_auc"])]
    res["search_narrow"], res["search_wide"] = narrow, wide

    fixed = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, scale_pos_weight=spw,
                          random_state=SEED, n_jobs=2, tree_method="hist", eval_metric="logloss")
    fixed.fit(X[tr], y[tr])
    res["fixed_untuned_on_corrected_split"] = score(fixed, X, y, va, te)
    fixed_m = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, scale_pos_weight=spw,
                            random_state=SEED, n_jobs=2, tree_method="hist", eval_metric="logloss",
                            monotone_constraints=mono)
    fixed_m.fit(X[tr], y[tr])
    res["fixed_untuned_with_monotone"] = score(fixed_m, X, y, va, te)

    # ------------------------------------------------ (2) M2 by column group
    rule = df[["validation_score", "cusum_threshold", "adaptive_cusum", "short_ema", "long_ema"]].astype(float)
    ema_cols = rule[["short_ema", "long_ema"]].copy()
    ema_cols["rule_ema_spread"] = rule["short_ema"] - rule["long_ema"]
    ema_cols["rule_ema_ratio"] = rule["short_ema"] / (rule["long_ema"] + 1e-9)
    groups = {
        "none (40 signal-only features)": None,
        "+ rule score": rule[["validation_score"]],
        "+ rule CUSUM statistic and threshold": rule[["adaptive_cusum", "cusum_threshold"]],
        "+ rule EMAs, their difference and ratio": ema_cols,
        "+ rule CUSUM and EMAs (no score)": pd.concat([rule[["adaptive_cusum", "cusum_threshold"]], ema_cols], axis=1),
        "+ all rule-derived columns": pd.concat([rule[["validation_score", "adaptive_cusum", "cusum_threshold"]], ema_cols], axis=1),
    }
    best_params = stored["best_params"]
    dec = {}
    for gname, extra in groups.items():
        Xg = X if extra is None else np.hstack([X, extra.to_numpy(dtype=np.float32)])
        mono_g = mono if extra is None else tuple(mono) + (0,) * extra.shape[1]
        m_t = _make_model(best_params, spw, SEED, 2000, 50, mono_g)
        m_t.fit(Xg[tr], y[tr], eval_set=[(Xg[va], y[va])], verbose=False)
        m_f = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, scale_pos_weight=spw,
                            random_state=SEED, n_jobs=2, tree_method="hist", eval_metric="logloss")
        m_f.fit(Xg[tr], y[tr])
        dec[gname] = {"n_features": int(Xg.shape[1]),
                      "main_hyperparameters": score(m_t, Xg, y, va, te),
                      "fixed_untuned": score(m_f, Xg, y, va, te)}
    res["m2_by_column_group"] = dec

    # ------------------------------------------------ (3) causal label-free baselines
    dp, q = df["dp"].to_numpy(float), df["flowrate"].to_numpy(float)
    R = dp / np.clip(q, 0.5, None)
    healthy_head = np.arange(0, 7800)  # Train-A, the purge gap and Val-A: all before the test window
    yt = y[te]
    onset_w = int(np.flatnonzero(yt == 1)[0])
    base = {}

    def event_stats(flags_te):
        pos = np.flatnonzero(flags_te)
        d3 = detection_delay(flags_te.astype(float), yt, 0.5, consecutive=3)
        return {
            "n_flagged": int(flags_te.sum()),
            "precision": float(precision_score(yt, flags_te, zero_division=0)),
            "recall": float(recall_score(yt, flags_te, zero_division=0)),
            "f1": float(f1_score(yt, flags_te, zero_division=0)),
            "first_flag_minus_onset": None if len(pos) == 0 else int(pos[0] - onset_w),
            "sustained_alarm_minus_onset": d3.get("delay_rows"),
            "cleared_alarms_before_onset": d3.get("cleared_alarm_episodes_before_onset"),
        }

    # B1: healthy-envelope limit = largest value seen before the test window
    for nm, sig in (("dp", dp), ("dp_over_q", R)):
        lim = float(sig[healthy_head].max())
        f = (sig[te] > lim).astype(int)
        base[f"envelope_{nm}"] = {"limit": lim, "calibrated_on_rows": [1, 7800], **event_stats(f)}

    # B2: one-sided CUSUM on the one-step EWMA forecast error, k = 0.5 sigma,
    # decision limit = largest CUSUM value over the healthy head (no alarm there)
    for nm, sig in (("dp", dp), ("dp_over_q", R)):
        s = pd.Series(sig)
        fc = s.ewm(span=50, adjust=False).mean().shift(1)
        e = (s - fc).to_numpy()
        e[0] = 0.0
        cal_idx = healthy_head[healthy_head >= 400]  # after the initial zero-dP stretch
        sd = float(np.std(e[cal_idx]))
        z = e / sd
        S = np.zeros(len(z))
        acc = 0.0
        for i, v in enumerate(z):
            acc = max(0.0, acc + v - 0.5)
            S[i] = acc
        h = float(S[cal_idx].max())
        f = (S[te] > h).astype(int)
        first_any = np.flatnonzero(S > h)
        base[f"cusum_forecast_error_{nm}"] = {
            "residual_sd_healthy": sd, "k": 0.5, "decision_limit_h": h,
            "calibrated_on_rows": [401, 7800],
            "first_alarm_row_whole_series": int(first_any[0] + 1) if len(first_any) else None,
            "cusum_at_step_row_8340": float(S[8339]), "cusum_max_test_before_step": float(S[7860:8339].max()),
            **event_stats(f)}
    res["causal_label_free_baselines"] = base

    (OUT / "a5_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
