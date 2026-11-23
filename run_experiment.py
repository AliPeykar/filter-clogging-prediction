"""Single entry point for the corrected filter-clogging pipeline.

Usage:
    python run_experiment.py --config configs/main.yaml
    python run_experiment.py --config configs/strict_temporal.yaml \
        --reuse-run outputs/main_<stamp>   # S1 reuses S2's tuning

Every run archives its resolved config, metrics, figures, and (optionally)
SHAP artifacts under outputs/<name>_<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from filterclog.calibration import (
    PlattCalibrator,
    select_threshold_fpr,
    select_threshold_max_f1,
)
from filterclog.config import config_to_dict, load_config
from filterclog.data import load_dataset, validate_dataset, verify_sampling_interval
from filterclog.evaluation import (
    anticipation_profile,
    block_bootstrap_ci,
    delay_sensitivity,
    detection_delay,
    expected_calibration_error,
    mcnemar_test,
    paired_bootstrap_diff,
    point_metrics,
)
from filterclog.features import CausalFeatureEngineer
from filterclog.hpo import optimize_xgb
from filterclog.reporting import (
    STANDARD_CAVEATS,
    git_commit_hash,
    make_run_dir,
    save_json,
    write_report_md,
)
from filterclog.splitting import describe_split, make_splits
from filterclog.target import audit_labels, make_binary_target, make_horizon_target
from filterclog.train import train_rf_baseline, train_xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_experiment")


def run(config_path: str, reuse_run: str | None = None) -> Path:
    cfg = load_config(config_path)
    np.random.seed(cfg.seed)
    run_dir = make_run_dir(ROOT / cfg.output_root, cfg.name)
    shutil.copy(config_path, run_dir / "config.yaml")

    # ---------------------------------------------------------------- data
    df = load_dataset(ROOT / cfg.data_path, cfg.sheet_name)
    validate_dataset(df)
    sampling = verify_sampling_interval(df)
    label_audit = audit_labels(df, run_dir / "label_audit.json")
    y_base = make_binary_target(df)
    sec_per_row = sampling["derived_seconds_per_row"]

    # ---------------------------------------------------------------- split
    split = make_splits(len(df), cfg.split)
    split_desc = describe_split(split, y_base)
    logger.info("split: %s", split_desc)

    # -------------------------------------------------------------- features
    eng = CausalFeatureEngineer(
        cfg.feature_set, cfg.rolling_windows, cfg.ema_spans, cfg.cusum_k
    )
    eng.fit(df.iloc[split.train])
    X_frame = eng.transform(df)
    feature_names = list(X_frame.columns)
    X = X_frame.to_numpy(dtype=np.float32)
    warm = eng.warmup_rows
    monotone = eng.monotone_constraints(feature_names) if cfg.monotone_constraints else None

    def usable(idx: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Drop warm-up rows and undefined horizon labels (-1)."""
        idx = idx[idx >= warm]
        return idx[y[idx] >= 0]

    # S1 reuse: hyperparameters / iteration count / threshold from an S2 run
    reused = None
    if reuse_run:
        with open(Path(reuse_run) / "metrics.json", encoding="utf-8") as fh:
            reused = json.load(fh)
        logger.info("reusing tuning from %s", reuse_run)

    results: dict = {
        "config": config_to_dict(cfg),
        "git_commit": git_commit_hash(),
        "sampling": sampling,
        "label_audit": label_audit,
        "split": split_desc,
        "feature_names": feature_names,
        "horizons": {},
        "caveats": list(STANDARD_CAVEATS),
    }

    fig_dir = run_dir / "figures"
    transition_row = label_audit["transition_rows_1based"][0]
    if cfg.run_figures:
        from filterclog import figures
        from filterclog.physics import RESISTANCE_SI, fit_forchheimer

        figures.timeline_with_splits(df, split.segments, transition_row, fig_dir)
        figures.label_provenance(df, fig_dir)
        figures.eda_distributions(df, fig_dir)
        figures.correlation_heatmap(df, X_frame, fig_dir)
        # Per-phase physics. Forchheimer fits are computed only as archived
        # diagnostics: within any phase the dp variation is degradation-
        # driven, so the fit is confounded (R^2 < 0, B < 0) — evidence for
        # using the Darcy operating line dP = R*Q instead
        # (docs/physics_formulation.md).
        phases = [
            ("healthy", (0, 7140), "#009E73"),
            ("transition band", (7860, 9600), "#56B4E9"),
            ("clogged plateau", (10320, len(df)), "#D55E00"),
        ]
        phase_info, forch_diag = [], []
        for n, rows, c in phases:
            seg = df.iloc[rows[0]:rows[1]]
            r_med = float((seg["dp"] / seg["flowrate"].clip(lower=0.5)).median())
            phase_info.append({
                "name": n, "rows": rows, "color": c,
                "r_med_kpa_h_m3": r_med,
                "r_med_si": r_med * RESISTANCE_SI,
            })
            forch_diag.append(
                fit_forchheimer(seg["flowrate"].to_numpy(), seg["dp"].to_numpy())
                | {"phase": n, "rows": list(rows)}
            )
        figures.dp_q_physics(df, phase_info, fig_dir)
        results["phase_resistance_si"] = {
            p["name"]: p["r_med_si"] for p in phase_info
        }
        results["forchheimer_diagnostics"] = forch_diag

    horizon_fig_rows = []
    for h in cfg.horizons:
        h_key = f"h{h}"
        y_h = make_horizon_target(y_base, h)
        tr = usable(split.train, y_h)
        va = usable(split.val, y_h)
        te = usable(split.test, y_h)
        block: dict = {"h_rows": int(h)}

        # ------------------------------------------------------------- HPO
        has_val = len(va) > 0
        if reused is not None:
            if h_key not in reused["horizons"]:
                raise KeyError(
                    f"reuse run has no horizon {h_key!r}; available: "
                    f"{list(reused['horizons'])}"
                )
            src = reused["horizons"][h_key]["xgb"]
            best_params = src["best_params"]
            hpo_info = {"reused_from": reuse_run}
            fixed_iters = src["n_estimators_final"]
        else:
            hpo_res = optimize_xgb(
                X, y_h, tr, va,
                n_trials=cfg.hpo.n_trials,
                n_cv_splits=cfg.hpo.n_cv_splits,
                cv_gap=cfg.hpo.cv_gap,
                seed=cfg.seed,
                max_n_estimators=cfg.hpo.max_n_estimators,
                early_stopping_rounds=cfg.hpo.early_stopping_rounds,
                timeout_s=cfg.hpo.timeout_s,
                monotone=monotone,
            )
            best_params = hpo_res["best_params"]
            hpo_info = hpo_res
            fixed_iters = None

        # ----------------------------------------------------------- train
        model = train_xgb(
            best_params, X[tr], y_h[tr],
            X[va] if has_val else None, y_h[va] if has_val else None,
            seed=cfg.seed,
            max_n_estimators=cfg.hpo.max_n_estimators,
            early_stopping_rounds=cfg.hpo.early_stopping_rounds,
            fixed_n_estimators=fixed_iters,
            monotone=monotone,
        )
        model.save_model(run_dir / f"xgb_{h_key}.json")

        proba_val_raw = model.predict_proba(X[va])[:, 1] if has_val else None
        proba_test_raw = model.predict_proba(X[te])[:, 1]

        # ----------------------------------------- calibration + threshold
        if has_val and cfg.calibration == "sigmoid":
            cal = PlattCalibrator().fit(proba_val_raw, y_h[va])
            proba_val = cal.transform(proba_val_raw)
            proba_test = cal.transform(proba_test_raw)
            cal_info = {
                "method": "platt_sigmoid",
                "platt_coef": list(cal.coef()),
                "val_brier_raw": float(np.mean((proba_val_raw - y_h[va]) ** 2)),
                "val_brier_calibrated": float(np.mean((proba_val - y_h[va]) ** 2)),
            }
        elif reused is not None and reused["horizons"][h_key]["xgb"]["calibration"].get("platt_coef"):
            # reapply the S2 run's exact calibration so the reused thresholds
            # operate in the probability space they were selected in
            src_cal = reused["horizons"][h_key]["xgb"]["calibration"]
            cal = PlattCalibrator.from_coef(*src_cal["platt_coef"])
            proba_val = proba_val_raw
            proba_test = cal.transform(proba_test_raw)
            cal_info = {"method": "platt_sigmoid", "reused_coef": src_cal["platt_coef"]}
        else:
            proba_val, proba_test = proba_val_raw, proba_test_raw
            cal_info = {"method": "none"}

        if has_val:
            thr_f1 = select_threshold_max_f1(proba_val, y_h[va])
            thr_fpr = select_threshold_fpr(proba_val, y_h[va], 0.01)
        elif reused is not None:
            src = reused["horizons"][h_key]["xgb"]
            thr_f1 = src["threshold_max_f1"]
            thr_fpr = src["threshold_fpr1pct"]
        else:
            raise RuntimeError("no validation block and no reused thresholds")
        thr = thr_f1 if cfg.threshold_rule == "max_f1" else thr_fpr

        # ------------------------------------------------------- evaluation
        test_metrics = point_metrics(y_h[te], proba_test, thr)
        boot = block_bootstrap_ci(
            y_h[te], proba_test, thr,
            block_len=cfg.bootstrap_blocks,
            n_resamples=cfg.bootstrap_resamples,
            seed=cfg.seed,
        )
        delay = detection_delay(
            proba_test, y_h[te], thr,
            consecutive=cfg.alarm_consecutive,
            seconds_per_row=sec_per_row,
        )
        sensitivity = delay_sensitivity(
            proba_test, y_h[te],
            {"max_f1": thr_f1, "fpr_1pct": thr_fpr},
            seconds_per_row=sec_per_row,
        )
        sensitivity.append({
            "caveat": (
                "the fpr_1pct threshold is selected on the highly separable "
                "validation healthy block and does not transfer to the "
                "held-out-transition regime; reported for completeness only"
            )
        })
        anticipation = anticipation_profile(
            proba_test, y_h[te], thr, seconds_per_row=sec_per_row
        )

        n_est_final = int(fixed_iters or (int(getattr(model, "best_iteration", 0) or 0) + 1))
        block["xgb"] = {
            "best_params": best_params,
            "hpo": hpo_info,
            "n_estimators_final": n_est_final,
            "calibration": cal_info,
            "threshold_max_f1": float(thr_f1),
            "threshold_fpr1pct": float(thr_fpr),
            "test_metrics": test_metrics,
            "bootstrap_ci": boot,
            "detection_delay": delay,
            "delay_sensitivity": sensitivity,
            "anticipation_profile": anticipation,
        }

        # --------------------------------------------------------- baseline
        if has_val:
            rf = train_rf_baseline(X[tr], y_h[tr], cfg.seed)
            proba_test_rf_raw = rf.predict_proba(X[te])[:, 1]
            proba_val_rf = rf.predict_proba(X[va])[:, 1]
            cal_rf = PlattCalibrator().fit(proba_val_rf, y_h[va])
            proba_test_rf = cal_rf.transform(proba_test_rf_raw)
            thr_rf = select_threshold_max_f1(cal_rf.transform(proba_val_rf), y_h[va])
            block["rf"] = {
                "test_metrics": point_metrics(y_h[te], proba_test_rf, thr_rf),
                "anticipation_profile": anticipation_profile(
                    proba_test_rf, y_h[te], thr_rf, seconds_per_row=sec_per_row
                ),
            }
            block["comparison"] = {
                "paired_block_bootstrap_f1_diff": paired_bootstrap_diff(
                    y_h[te], proba_test, proba_test_rf, thr, thr_rf,
                    block_len=cfg.bootstrap_blocks,
                    n_resamples=cfg.bootstrap_resamples,
                    seed=cfg.seed,
                ),
                "mcnemar": mcnemar_test(
                    y_h[te],
                    (proba_test >= thr).astype(int),
                    (proba_test_rf >= thr_rf).astype(int),
                ),
            }

        # ----------------------------------------------------------- figures
        if cfg.run_figures:
            from filterclog import figures

            curves = {f"XGB ({cfg.feature_set})": (y_h[te], proba_test)}
            if "rf" in block:
                curves["RF baseline"] = (y_h[te], proba_test_rf)
                figures.anticipation_curves(
                    {"XGBoost": anticipation, "RF baseline": block["rf"]["anticipation_profile"]},
                    fig_dir, name=f"anticipation_curves_{h_key}",
                )
            figures.pr_roc_curves(curves, fig_dir, name=f"pr_roc_curves_{h_key}")
            if has_val and cfg.calibration == "sigmoid" and len(np.unique(y_h[te])) == 2:
                figures.calibration_plot(
                    y_h[te], proba_test_raw, proba_test,
                    expected_calibration_error(y_h[te], proba_test_raw),
                    expected_calibration_error(y_h[te], proba_test),
                    fig_dir, name=f"calibration_{h_key}",
                )
            figures.probability_trajectory(
                proba_test, y_h[te], thr,
                delay.get("onset_row_in_window", 0),
                delay.get("alarm_decision_row_in_window"),
                int(te.min()) + 1,
                fig_dir,
                name=f"prob_trajectory_{h_key}",
            )
            figures.residual_analysis(
                y_h[te], proba_test, int(te.min()) + 1, fig_dir,
                name=f"residual_analysis_{h_key}",
            )
            if reused is None:
                figures.hpo_history(hpo_info["trial_values"], fig_dir)
                diag = hpo_info.get("cv_diagnostic", {})
                if diag.get("fold_train_sizes"):
                    figures.cv_and_learning_curve(
                        diag["fold_train_sizes"], diag["fold_logloss"],
                        diag["fold_pr_auc"], fig_dir,
                    )
            figures.confusion_matrix_plot(
                np.array(test_metrics["confusion_matrix"]),
                ["Not Clogged", "Clogged"], fig_dir,
                name=f"confusion_{h_key}",
            )

        if h > 0 and delay.get("delay_rows") is not None:
            horizon_fig_rows.append({
                "h_rows": int(h),
                "pr_auc": test_metrics["pr_auc"],
                "recall": test_metrics["recall"],
                "precision": test_metrics["precision"],
                "effective_warning_min": round(
                    (h - delay["delay_rows"]) * sec_per_row / 60.0, 2
                ),
            })

        results["horizons"][h_key] = block

    if cfg.run_figures and len(horizon_fig_rows) >= 2:
        from filterclog import figures

        figures.horizon_summary(horizon_fig_rows, fig_dir)

    # ----------------------------------------------------------------- SHAP
    if cfg.run_shap and "h0" in results["horizons"]:
        from filterclog.explain import compute_and_persist_shap
        from xgboost import XGBClassifier

        model0 = XGBClassifier()
        model0.load_model(run_dir / "xgb_h0.json")
        te0 = usable(split.test, y_base)
        ranking = compute_and_persist_shap(model0, X_frame.iloc[te0], run_dir / "shap")
        results["shap_top10"] = ranking.head(10).to_dict("records")

    # ----------------------------------------------------------- risk levels
    if cfg.run_risk_levels:
        from filterclog.risk import RISK_CAVEATS, evaluate_risk_model, make_risk_target

        tr0_count = usable(split.train, y_base)
        levels, bands = make_risk_target(df["validation_score"], count_idx=tr0_count)
        tr0 = tr0_count
        va0 = usable(split.val, y_base)
        te0 = usable(split.test, y_base)
        risk = evaluate_risk_model(
            X[tr0], levels[tr0], X[va0], levels[va0], X[te0], levels[te0],
            n_classes=len(bands), seed=cfg.seed,
        )
        risk["bands_score_ranges"] = [list(b) for b in bands]
        results["risk_levels"] = risk
        results["caveats"].extend(RISK_CAVEATS)
        if cfg.run_figures:
            from filterclog import figures

            figures.confusion_matrix_plot(
                np.array(risk["confusion_matrix"]),
                [f"L{i} ({lo}-{hi})" for i, (lo, hi) in enumerate(bands)],
                fig_dir, name="confusion_risk_levels",
            )

    # ---------------------------------------------------------------- output
    save_json(results, run_dir / "metrics.json")
    write_report_md(results, run_dir / "report.md")
    from filterclog.reporting import export_tables

    export_tables(results, run_dir / "tables")
    logger.info("run complete: %s", run_dir)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reuse-run", default=None,
                        help="path to a previous run dir whose tuning is reused (S1)")
    args = parser.parse_args()
    run(args.config, args.reuse_run)
