"""Revision analyses for review round 1 (comments M3, M4, M6, M7, M8).

Read-only on the project; writes paper/03_analysis/out/a4_results.json and
CSV files. Usage: python a4_review_r1_checks.py <main_run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filterclog.calibration import PlattCalibrator, select_threshold_max_f1  # noqa: E402
from filterclog.config import load_config  # noqa: E402
from filterclog.data import load_dataset  # noqa: E402
from filterclog.features import CausalFeatureEngineer  # noqa: E402
from filterclog.splitting import make_splits  # noqa: E402
from filterclog.target import make_binary_target  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
SEED, NBOOT = 42, 1000
ONSET, STEP = 8943, 8340  # 1-based rows


def boot_idx(n, block, n_resamples, seed):
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    for _ in range(n_resamples):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        yield np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def pct(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def acf(x, lag):
    x = np.asarray(x, float) - np.mean(x)
    return float((x[:-lag] * x[lag:]).sum() / (x * x).sum())


def main(run_dir: str) -> None:
    run = Path(run_dir).resolve()
    cfg = load_config(run / "config.yaml")
    metrics = json.loads((run / "metrics.json").read_text())
    df = load_dataset(ROOT / cfg.data_path, cfg.sheet_name)
    y = make_binary_target(df)
    dp, q = df["dp"].to_numpy(float), df["flowrate"].to_numpy(float)
    score = df["validation_score"].to_numpy()
    res: dict = {}

    split = make_splits(len(df), cfg.split)
    seg = {k: np.arange(lo - 1, hi) for k, (lo, hi) in split.segments.items()}
    te = split.test
    yt = y[te]
    rows_te = te + 1
    R = dp / np.clip(q, 0.5, None)

    # ------------------------------------------------ M3: two-variable support
    sup = {}
    for k, idx in seg.items():
        sup[k] = {
            "q_min": float(q[idx].min()), "q_p01": float(np.percentile(q[idx], 1)),
            "q_median": float(np.median(q[idx])), "q_p99": float(np.percentile(q[idx], 99)),
            "q_max": float(q[idx].max()),
            "frac_q_le_6p6": float((q[idx] <= 6.6).mean()),
            "frac_q_le_6p68": float((q[idx] < 6.68).mean()),
            "r_min": float(R[idx].min()), "r_max": float(R[idx].max()),
        }
    fit = np.concatenate([split.train, split.val])
    h, c = fit[y[fit] == 0], fit[y[fit] == 1]
    q_h_min, q_c_max = float(q[h].min()), float(q[c].max())
    q_c_p99 = float(np.percentile(q[c], 99))
    r_h_max, r_c_min = float(R[h].max()), float(R[c].min())
    pre = te[(rows_te < STEP)]
    mid = te[(rows_te >= STEP) & (rows_te < ONSET)]
    post = te[rows_te >= ONSET]
    res["support_2d"] = {
        "segments": sup,
        "healthy_fit_q_min": q_h_min, "clogged_fit_q_max": q_c_max, "clogged_fit_q_p99": q_c_p99,
        "healthy_fit_q_p01": float(np.percentile(q[h], 1)),
        "clogged_fit_frac_q_le_6p6": float((q[c] <= 6.6).mean()),
        "healthy_fit_frac_q_le_6p6": float((q[h] <= 6.6).mean()),
        "healthy_fit_r_max": r_h_max, "clogged_fit_r_min": r_c_min,
        "test_pre_step": {"n": int(len(pre)), "frac_q_le_6p6": float((q[pre] <= 6.6).mean()),
                          "q_median": float(np.median(q[pre])), "frac_q_below_healthy_min": float((q[pre] < q_h_min).mean())},
        "test_step_to_onset": {"n": int(len(mid)), "frac_q_le_6p6": float((q[mid] <= 6.6).mean()),
                               "q_median": float(np.median(q[mid])), "frac_q_below_healthy_min": float((q[mid] < q_h_min).mean()),
                               "r_min": float(R[mid].min()), "r_max": float(R[mid].max())},
        "test_post_onset": {"n": int(len(post)), "frac_q_le_6p6": float((q[post] <= 6.6).mean()),
                            "q_median": float(np.median(q[post])), "frac_q_below_healthy_min": float((q[post] < q_h_min).mean())},
    }
    # joint support: a test sample is "inside" if its (Q, dP) pair lies inside the
    # bounding box of healthy fit samples or inside that of clogged fit samples
    def in_box(idx, ref):
        return ((q[idx] >= q[ref].min()) & (q[idx] <= q[ref].max())
                & (dp[idx] >= dp[ref].min()) & (dp[idx] <= dp[ref].max()))
    inside = in_box(te, h) | in_box(te, c)
    res["support_2d"]["test_outside_both_boxes"] = {
        "n": int((~inside).sum()), "frac": float((~inside).mean()),
        "n_healthy": int(((~inside) & (yt == 0)).sum()), "n_clogged": int(((~inside) & (yt == 1)).sum())}
    dp_gap = (dp[te] > dp[h].max()) & (dp[te] < dp[c].min())
    q_gap = (q[te] < q_h_min) & (q[te] > q_c_p99)
    res["support_2d"]["test_in_dp_gap"] = int(dp_gap.sum())
    res["support_2d"]["test_q_below_healthy_fit_min"] = int((q[te] < q_h_min).sum())
    res["support_2d"]["test_in_dp_gap_and_q_below_healthy_min"] = int((dp_gap & (q[te] < q_h_min)).sum())

    # events in the test window
    def event(r):
        i = r - 1
        return {"row": r, "dp": [float(dp[i - 1]), float(dp[i])], "q": [float(q[i - 1]), float(q[i])],
                "r_change_pct": float(100 * (R[i] / R[i - 1] - 1))}
    d = np.diff(dp)
    big = [int(i + 2) for i in np.flatnonzero(np.abs(d[te[0]:te[-1]]) >= 1.0) + te[0]]
    res["test_window_events_abs_dp_jump_ge_1kpa"] = [event(r) for r in big]
    res["step_event"] = event(STEP)

    # ----------------------------------------- model flags and what explains them
    eng = CausalFeatureEngineer(cfg.feature_set, cfg.rolling_windows, cfg.ema_spans, cfg.cusum_k)
    eng.fit(df.iloc[split.train])
    Xf = eng.transform(df)
    X = Xf.to_numpy(dtype=np.float32)
    va = split.val
    xgb = XGBClassifier()
    xgb.load_model(run / "xgb_h0.json")
    pv_raw, pt_raw = xgb.predict_proba(X[va])[:, 1], xgb.predict_proba(X[te])[:, 1]
    cal = PlattCalibrator().fit(pv_raw, y[va])
    pt = cal.transform(pt_raw)
    thr = select_threshold_max_f1(cal.transform(pv_raw), y[va])
    pred = (pt >= thr).astype(int)
    assert abs(f1_score(yt, pred) - metrics["horizons"]["h0"]["xgb"]["test_metrics"]["f1"]) < 1e-9
    slope, intercept = cal.coef()
    raw_thr = float(1 / (1 + np.exp((intercept - np.log(thr / (1 - thr))) / slope)))
    res["platt"] = {
        "slope": slope, "intercept": intercept, "threshold_calibrated": float(thr),
        "threshold_raw_equivalent": raw_thr,
        "val_raw_max_healthy": float(pv_raw[y[va] == 0].max()),
        "val_raw_min_clogged": float(pv_raw[y[va] == 1].min()),
        "val_separable_by_model": bool(pv_raw[y[va] == 0].max() < pv_raw[y[va] == 1].min()),
        "val_brier_raw": float(np.mean((pv_raw - y[va]) ** 2)),
        "val_brier_platt": float(np.mean((cal.transform(pv_raw) - y[va]) ** 2)),
        "scale_pos_weight": float((y[split.train] == 0).sum() / (y[split.train] == 1).sum()),
        "test_raw_p_max": float(pt_raw.max()),
        "test_raw_p_clogged_median": float(np.median(pt_raw[yt == 1])),
    }

    # agreement of the XGBoost flags with single-variable rules
    def agree(rule):
        return {"agreement": float((rule == pred).mean()),
                "f1_vs_xgb_flags": float(f1_score(pred, rule)),
                "f1_vs_rule_label": float(f1_score(yt, rule)),
                "n_flag": int(rule.sum())}
    qm5 = Xf["flowrate_rmean_5"].to_numpy()[te]
    qm15 = Xf["flowrate_rmean_15"].to_numpy()[te]
    cands = {}
    for th in (18.25, 19.0, 20.0):
        cands[f"dp>={th}"] = agree((dp[te] >= th).astype(int))
    for th in (6.5, 6.55, 6.6, 6.65, 6.7, 6.75):
        cands[f"q<={th}"] = agree((q[te] <= th).astype(int))
        cands[f"q_rmean15<={th}"] = agree((qm15 <= th).astype(int))
    for th in (2.7, 2.8, 2.9, 3.0):
        cands[f"dp/q>={th}"] = agree((R[te] >= th).astype(int))
    res["xgb_flag_agreement_with_simple_rules"] = cands
    best = max(cands.items(), key=lambda kv: kv[1]["agreement"])
    res["xgb_flag_best_simple_rule"] = {"rule": best[0], **best[1]}

    # Q-threshold sweep scored against the rule label
    qs = np.round(np.arange(5.8, 7.21, 0.05), 2)
    sweep = []
    onset_w = int(np.flatnonzero(yt == 1)[0])
    for th in qs:
        p = (q[te] <= th).astype(int)
        pos = np.flatnonzero(p == 1)
        sweep.append({"q_threshold": float(th), "precision": float(precision_score(yt, p, zero_division=0)),
                      "recall": float(recall_score(yt, p, zero_division=0)),
                      "f1": float(f1_score(yt, p, zero_division=0)),
                      "first_flag_minus_onset": None if len(pos) == 0 else int(pos[0] - onset_w)})
    pd.DataFrame(sweep).to_csv(OUT / "q_threshold_sweep.csv", index=False)
    res["q_threshold_sweep"] = {"f1_min": min(s["f1"] for s in sweep), "f1_max": max(s["f1"] for s in sweep),
                                "argmax": max(sweep, key=lambda s: s["f1"])}
    res["single_signal_auc"] = {"neg_q": float(roc_auc_score(yt, -q[te])),
                                "neg_q_rmean15": float(roc_auc_score(yt, -qm15)),
                                "dp": float(roc_auc_score(yt, dp[te]))}

    # false positives: flag rates by Q, and SHAP restricted to them
    fp = (pred == 1) & (yt == 0)
    midmask = (rows_te >= STEP + 1) & (rows_te < ONSET)
    res["xgb_flags_step_to_onset"] = {
        "n_samples": int(midmask.sum()), "n_flagged": int(pred[midmask].sum()),
        "flag_rate_q_le_6p6": float(pred[midmask & (q[te] <= 6.6)].mean()),
        "flag_rate_q_gt_6p6": float(pred[midmask & (q[te] > 6.6)].mean()),
        "n_q_gt_6p6": int((midmask & (q[te] > 6.6)).sum()),
        "unflagged_rows": [int(r) for r in rows_te[midmask & (pred == 0)]][:60],
    }
    import shap
    expl = shap.TreeExplainer(xgb)
    sv = expl.shap_values(X[te])
    names = list(Xf.columns)

    def top(mask, k=8):
        m = np.abs(sv[mask]).mean(0)
        o = np.argsort(m)[::-1][:k]
        return [{"feature": names[i], "mean_abs_shap": float(m[i]), "mean_shap": float(sv[mask][:, i].mean())} for i in o]
    res["shap_false_positives"] = top(fp)
    res["shap_test_window"] = top(np.ones(len(te), bool))
    # contrast: mean SHAP shift from the pre-step healthy samples to the false positives
    pre_mask = rows_te < STEP
    shift = sv[fp].mean(0) - sv[pre_mask].mean(0)
    o = np.argsort(np.abs(shift))[::-1][:8]
    res["shap_shift_pre_step_to_false_positives"] = [
        {"feature": names[i], "shift": float(shift[i])} for i in o]
    q_feats = [i for i, n in enumerate(names) if n.startswith("flow") and "interaction" not in n]
    dp_feats = [i for i, n in enumerate(names) if n.startswith(("dp", "ema", "resistance"))]
    inter = [i for i, n in enumerate(names) if "interaction" in n]
    res["shap_shift_by_group"] = {"q_features": float(shift[q_feats].sum()), "dp_features": float(shift[dp_feats].sum()),
                                  "q_x_dp_interaction": float(shift[inter].sum()), "total": float(shift.sum())}
    cus = Xf["dp_cusum_pos"].to_numpy()
    nz = np.flatnonzero(cus[te] > 0)
    res["cusum_feature_test"] = {"frac_zero": float((cus[te] == 0).mean()),
                                 "first_nonzero_row": int(rows_te[nz[0]]) if len(nz) else None}

    # ---------------------------------------------------- M4: label audit
    ch = np.flatnonzero(np.diff(score) != 0) + 1
    same_dp = [int(c + 1) for c in ch if dp[c] == dp[c - 1]]
    cols = ["dp", "flowrate", "short_ema", "long_ema", "adaptive_cusum", "cusum_threshold"]

    def window_stats(r, w=5):
        i = r - 1
        out = {}
        for col in cols:
            v = df[col].to_numpy(float)[i - w:i + w + 1]
            out[col] = {"min": float(v.min()), "max": float(v.max()), "max_abs_step": float(np.abs(np.diff(v)).max())}
        return out
    lab = {
        "n_score_steps": int(len(ch)), "n_steps_with_dp_unchanged": int(len(same_dp)),
        "steps_with_dp_unchanged_from_row_4000": [r for r in same_dp if r >= 4000],
        "window_pm5_at_7001": window_stats(7001), "window_pm5_at_7501": window_stats(7501),
        "window_pm5_at_8943": window_stats(8943),
        "n_score_ge_12": int((score >= 12).sum()),
        "rows_score_ge_12": [int((np.flatnonzero(score >= 12) + 1).min()), int((np.flatnonzero(score >= 12) + 1).max())],
        "n_score_ge_10": int((score >= 10).sum()), "first_row_score_ge_10": int(np.flatnonzero(score >= 10)[0] + 1),
    }
    ac, ct = df["adaptive_cusum"].to_numpy(float), df["cusum_threshold"].to_numpy(float)
    exc = ac > ct
    lab["cusum_exceeds_threshold"] = {"n": int(exc.sum()), "first_row": int(np.flatnonzero(exc)[0] + 1),
                                      "continuous_from_row": int(len(df) - np.argmin(exc[::-1]) + 1) if not exc.all() else 1}
    lab["rule_column_auc_test"] = {col: float(roc_auc_score(yt, df[col].to_numpy(float)[te])) for col in
                                   ["adaptive_cusum", "short_ema", "long_ema", "validation_score"]}
    # persistence hypothesis: first row t at which `sig >= theta` has held for N consecutive samples
    def first_persist(sig, theta, n):
        run = 0
        for i, v in enumerate(sig):
            run = run + 1 if v >= theta else 0
            if run >= n:
                return i + 1
        return None
    hyp = []
    se, le = df["short_ema"].to_numpy(float), df["long_ema"].to_numpy(float)
    for name, sig in (("dp", dp), ("short_ema", se), ("long_ema", le)):
        for n in (500, 550, 590, 595, 600, 601, 602, 603, 604, 605, 610, 650):
            for theta in np.round(np.arange(18.1, 20.6, 0.1), 1):
                r = first_persist(sig, theta, n)
                if r == ONSET:
                    hyp.append({"signal": name, "theta": float(theta), "n": n})
    lab["persistence_hypotheses_matching_onset_exactly"] = hyp
    lab["first_row_dp_ge_18p5_for_600"] = first_persist(dp, 18.5, 600)
    lab["first_row_dp_ge_19_for_600"] = first_persist(dp, 19.0, 600)
    lab["first_row_dp_ge_19_for_604"] = first_persist(dp, 19.0, 604)
    res["label_audit"] = lab

    # ---------------------------------------------------- M6: block length
    rf_csv = pd.read_csv(OUT / "test_window_predictions.csv")
    p_rf = rf_csv["p_rf"].to_numpy()
    thr_rf = metrics["horizons"]["h0"]["rf"]["test_metrics"]["threshold"]
    pred_rf = (p_rf >= thr_rf).astype(int)
    err_x, err_r = (pred != yt).astype(float), (pred_rf != yt).astype(float)
    res["autocorrelation"] = {
        "dp_test_lag1": acf(dp[te], 1), "dp_test_lag50": acf(dp[te], 50), "dp_test_lag200": acf(dp[te], 200),
        "xgb_error_lag50": acf(err_x, 50), "xgb_error_lag200": acf(err_x, 200), "xgb_error_lag400": acf(err_x, 400),
        "rf_error_lag50": acf(err_r, 50), "label_lag50": acf(yt.astype(float), 50), "label_lag400": acf(yt.astype(float), 400),
    }
    bl = {}
    for block in (50, 100, 200, 400):
        f1x, f1r, aucx, aucr, aucd, d_dx, d_dr, d_f1 = [], [], [], [], [], [], [], []
        for idx in boot_idx(len(te), block, NBOOT, SEED):
            if len(np.unique(yt[idx])) < 2:
                continue
            ax, ar, ad = (roc_auc_score(yt[idx], pt[idx]), roc_auc_score(yt[idx], p_rf[idx]),
                          roc_auc_score(yt[idx], dp[te][idx]))
            fx, fr = f1_score(yt[idx], pred[idx], zero_division=0), f1_score(yt[idx], pred_rf[idx], zero_division=0)
            f1x.append(fx); f1r.append(fr); aucx.append(ax); aucr.append(ar); aucd.append(ad)
            d_dx.append(ad - ax); d_dr.append(ad - ar); d_f1.append(fx - fr)
        bl[str(block)] = {
            "n_valid": len(f1x), "xgb_f1": pct(f1x), "rf_f1": pct(f1r), "xgb_auc": pct(aucx), "rf_auc": pct(aucr),
            "dp_auc": pct(aucd), "auc_dp_minus_xgb": pct(d_dx), "auc_dp_minus_xgb_frac_pos": float(np.mean(np.array(d_dx) > 0)),
            "auc_dp_minus_rf": pct(d_dr), "auc_dp_minus_rf_frac_pos": float(np.mean(np.array(d_dr) > 0)),
            "f1_xgb_minus_rf": pct(d_f1), "f1_xgb_minus_rf_frac_neg": float(np.mean(np.array(d_f1) < 0)),
        }
    res["block_length_sensitivity"] = bl

    # ---------------------------------------------------- M8: quantisation
    u = np.unique(dp)
    steps = np.diff(u)
    res["dp_quantisation"] = {
        "n_unique": int(len(u)), "min_step": float(steps.min()), "median_step": float(np.median(steps)),
        "frac_steps_in_0p108_0p110": float(((steps > 0.1075) & (steps < 0.1105)).mean()),
        "step_8340_in_counts_of_0p1086": float((dp[STEP - 1] - dp[STEP - 2]) / 0.1086),
        "n_unique_q": int(len(np.unique(q))),
    }

    (OUT / "a4_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
