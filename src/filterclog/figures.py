"""Publication figures (300 dpi PNG + PDF)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)

# publication style: serif math-compatible fonts, sizes for two-column print,
# colorblind-safe Okabe-Ito palette
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

#: consistent model colors across every figure (Okabe-Ito)
MODEL_COLORS = {"xgb": "#0072B2", "rf": "#E69F00"}
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]

SEG_COLORS = {
    "train_a": "#c6dbef", "train_b": "#c6dbef", "train": "#c6dbef",
    "val_a": "#fdd0a2", "val_b": "#fdd0a2",
    "test": "#c7e9c0",
}


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("figure %s saved", name)


def timeline_with_splits(df: pd.DataFrame, segments: dict, transition_row: int, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    rows = np.arange(1, len(df) + 1)
    ax1.plot(rows, df["dp"], lw=0.8, color="#d62728", label="dp")
    ax1.set_ylabel("Differential pressure (dp)", color="#d62728")
    ax2 = ax1.twinx()
    ax2.plot(rows, df["flowrate"], lw=0.6, color="#1f77b4", alpha=0.7, label="flowrate")
    ax2.set_ylabel("Flowrate", color="#1f77b4")
    for name, (lo, hi) in segments.items():
        ax1.axvspan(lo, hi, color=SEG_COLORS.get(name, "#eeeeee"), alpha=0.35)
        ax1.text((lo + hi) / 2, ax1.get_ylim()[1] * 0.95, name, ha="center", fontsize=7)
    ax1.axvline(transition_row, color="k", ls="--", lw=1.2,
                label=f"onset (row {transition_row:,})")
    ax1.set_xlabel("Row (time order)")
    ax1.set_title("Sensor timeline, label onset, and split segments")
    ax1.legend(loc="upper left", fontsize=8)
    _save(fig, out_dir, "timeline_splits")


def label_provenance(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    rows = np.arange(1, len(df) + 1)
    ax.plot(rows, df["validation_score"], lw=0.8, color="#6a3d9a", label="validation_score (rule score)")
    y = (df["filter_status"] == "Clogged").astype(int)
    ax.fill_between(rows, 0, y * df["validation_score"].max(), alpha=0.15, color="#e31a1c",
                    label="label = Clogged")
    ax.axhline(11, color="k", ls=":", lw=1, label="empirical threshold (score >= 11)")
    ax.set_xlabel("Row (time order)")
    ax.set_ylabel("Rule score")
    ax.set_title("Label provenance: filter_status is a threshold on the rule score")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "label_provenance")


def pr_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], out_dir: Path,
                  name: str = "pr_roc_curves") -> None:
    """curves: name -> (y_true, proba). Skips one-class windows."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, (yt, pp) in curves.items():
        if len(np.unique(yt)) < 2:
            continue
        p, r, _ = precision_recall_curve(yt, pp)
        ax1.plot(r, p, label=label, lw=1.2)
        fpr, tpr, _ = roc_curve(yt, pp)
        ax2.plot(fpr, tpr, label=label, lw=1.2)
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision"); ax1.set_title("Precision-Recall (test)")
    ax2.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.set_title("ROC (test)")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)
    _save(fig, out_dir, name)


def calibration_plot(y_true: np.ndarray, proba_raw: np.ndarray, proba_cal: np.ndarray,
                     ece_raw: float, ece_cal: float, out_dir: Path,
                     name: str = "calibration") -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, pp, ece in (("raw", proba_raw, ece_raw), ("Platt-calibrated", proba_cal, ece_cal)):
        frac, mean_pred = calibration_curve(y_true, pp, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", label=f"{label} (ECE={ece:.3f})", lw=1.2, ms=4)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability curve (test)")
    ax.legend(fontsize=8)
    _save(fig, out_dir, name)


def probability_trajectory(proba: np.ndarray, y_true: np.ndarray, threshold: float,
                           onset_in_window: int, alarm_in_window: int | None,
                           window_start_row: int, out_dir: Path, name: str = "prob_trajectory") -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    rows = window_start_row + np.arange(len(proba))
    ax.plot(rows, proba, lw=0.8, color="#1f77b4", label="P(clogged), calibrated")
    ax.axhline(threshold, color="#ff7f0e", ls="--", lw=1, label=f"threshold={threshold:.3f}")
    ax.axvline(window_start_row + onset_in_window, color="k", ls="--", lw=1.2, label="true onset")
    if alarm_in_window is not None:
        ax.axvline(window_start_row + alarm_in_window, color="#d62728", ls="-", lw=1.2,
                   label="sustained alarm")
        delay = alarm_in_window - onset_in_window
        ax.set_title(f"Test-window probability trajectory (delay = {delay:+d} rows)")
    else:
        ax.set_title("Test-window probability trajectory (no sustained alarm)")
    ax.fill_between(rows, 0, y_true * 1.0, alpha=0.08, color="#e31a1c")
    ax.set_xlabel("Row (time order)"); ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="center left")
    _save(fig, out_dir, name)


def confusion_matrix_plot(cm: np.ndarray, class_names: list[str], out_dir: Path,
                          name: str = "confusion_matrix") -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, fraction=0.046)
    _save(fig, out_dir, name)


def ablation_comparison(metrics_by_run: dict[str, dict], out_dir: Path,
                        ci_by_run: dict[str, dict] | None = None) -> None:
    """Grouped bars with optional 95% block-bootstrap CI whiskers.

    metrics_by_run: run label -> test_metrics dict.
    ci_by_run: run label -> bootstrap_ci dict (metric -> {ci_low, ci_high}).
    """
    keys = ["balanced_accuracy", "f1", "mcc", "pr_auc", "roc_auc"]
    runs = list(metrics_by_run.keys())
    x = np.arange(len(keys))
    width = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, run in enumerate(runs):
        vals = [metrics_by_run[run].get(k) or 0 for k in keys]
        err = None
        if ci_by_run and run in ci_by_run:
            lo, hi = [], []
            for k, v in zip(keys, vals):
                c = ci_by_run[run].get(k) or {}
                if c.get("ci_low") is not None:
                    lo.append(max(v - c["ci_low"], 0))
                    hi.append(max(c["ci_high"] - v, 0))
                else:
                    lo.append(0.0)
                    hi.append(0.0)
            err = np.array([lo, hi])
        ax.bar(x + i * width, vals, width, label=run,
               color=OKABE_ITO[i % len(OKABE_ITO)], yerr=err, capsize=2,
               error_kw={"lw": 0.8})
    ax.set_xticks(x + width * (len(runs) - 1) / 2, keys, rotation=20)
    ax.set_ylim(0, 1.08)
    ax.set_title("Model / feature-set comparison (test; whiskers = 95% block-bootstrap CI)")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "ablation_comparison")


def eda_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Class-conditional histograms of the physical quantities."""
    resistance = df["dp"] / df["flowrate"].clip(lower=0.5)
    panels = [
        ("dp", df["dp"], r"$\Delta P$ [kPa]"),
        ("flowrate", df["flowrate"], r"$Q$ [m$^3$/h]"),
        ("resistance", resistance, r"$R=\Delta P/Q$ [kPa·h/m$^3$]"),
    ]
    clogged = (df["filter_status"] == "Clogged").to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (key, series, label) in zip(axes, panels):
        bins = np.histogram_bin_edges(series, bins=60)
        ax.hist(series[~clogged], bins=bins, alpha=0.65, label="Not Clogged",
                color="#009E73", density=True)
        ax.hist(series[clogged], bins=bins, alpha=0.65, label="Clogged",
                color="#D55E00", density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        if key == "flowrate":
            ax.set_yscale("log")  # heavy right tail from clean-filter excursions
    axes[0].legend()
    fig.suptitle("Class-conditional distributions of the measured quantities", y=1.02)
    _save(fig, out_dir, "eda_distributions")


def correlation_heatmap(df: pd.DataFrame, X: pd.DataFrame, out_dir: Path,
                        max_features: int = 18) -> None:
    """Spearman correlations: raw signals + a representative engineered
    subset (monotone-invariant, matching tree models)."""
    raw = df[["dp", "flowrate", "short_ema", "long_ema", "adaptive_cusum",
              "validation_score"]]
    keep = [c for c in (
        "resistance", "resistance_ratio", "resistance_slope_30", "dp_rmean_30",
        "dp_slope_30", "dp_madz_30", "dp_cusum_pos", "ema_spread",
        "flowrate_rmean_30", "flow_efficiency", "flow_dp_interaction",
    ) if c in X.columns][: max_features - raw.shape[1]]
    M = pd.concat([raw.reset_index(drop=True), X[keep].reset_index(drop=True)], axis=1)
    corr = M.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            if abs(v) >= 0.95 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, fraction=0.046, label="Spearman ρ")
    ax.set_title("Correlation structure (|ρ| ≥ 0.95 annotated)\n"
                 "left block: raw/precomputed columns incl. banned ones")
    _save(fig, out_dir, "correlation_heatmap")


def dp_q_physics(df: pd.DataFrame, phases: list[dict], out_dir: Path) -> None:
    """Delta-P vs Q scatter per lifecycle phase with Darcy operating lines.

    Within any phase the dp variation is degradation-driven (time), not
    Q-driven, so a Forchheimer fit is confounded (verified: R^2 < 0, B < 0 —
    diagnostics archived in metrics.json). The identifiable per-phase physics
    is the Darcy operating line Delta-P = R_med * Q, whose slope is the
    median specific flow resistance — rising across phases as permeability
    decays.

    phases: [{name, rows:(lo,hi), color, r_med_kpa_h_m3, r_med_si}].
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for pf in phases:
        lo, hi = pf["rows"]
        seg = df.iloc[lo:hi]
        ax.scatter(seg["flowrate"], seg["dp"], s=3, alpha=0.25, color=pf["color"],
                   label=f"{pf['name']} (rows {lo + 1}–{hi})", rasterized=True)
        qg = np.linspace(0, seg["flowrate"].max() * 1.02, 50)
        ax.plot(qg, pf["r_med_kpa_h_m3"] * qg, color=pf["color"], lw=1.6,
                label=(rf"  $\Delta P = \bar{{R}}Q$, "
                       rf"$\bar{{R}}$={pf['r_med_si']:.2e} Pa·s/m$^3$"))
    ax.set_xlabel(r"$Q$ [m$^3$/h]")
    ax.set_ylabel(r"$\Delta P$ [kPa]")
    ax.set_title("Operating lines per lifecycle phase: slope = specific flow\n"
                 r"resistance $\bar{R} \propto 1/k_1$ (Darcy permeability decay)")
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, out_dir, "dp_q_physics")


def hpo_history(trial_values: list[float], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    trials = np.arange(1, len(trial_values) + 1)
    vals = np.asarray(trial_values, float)
    ax.scatter(trials, vals, s=12, color="#56B4E9", label="trial")
    ax.plot(trials, np.minimum.accumulate(vals), color="#0072B2", lw=1.4,
            label="running best")
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("validation logloss")
    ax.set_title("Hyperparameter optimization history (constrained space)")
    ax.legend()
    _save(fig, out_dir, "hpo_history")


def cv_and_learning_curve(fold_sizes: list[int], fold_logloss: list[float],
                          fold_pr_auc: list[float], out_dir: Path) -> None:
    """Per-fold CV diagnostic doubling as an expanding-window learning curve
    (each fold's fit set is a strictly larger temporal prefix)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    folds = np.arange(1, len(fold_sizes) + 1)
    ax1.bar(folds - 0.18, fold_logloss, width=0.36, color="#0072B2", label="logloss")
    ax1b = ax1.twinx()
    ax1b.bar(folds + 0.18, fold_pr_auc, width=0.36, color="#E69F00", label="PR-AUC")
    ax1b.set_ylim(0, 1.05)
    ax1.set_xlabel("expanding-window fold")
    ax1.set_ylabel("logloss", color="#0072B2")
    ax1b.set_ylabel("PR-AUC", color="#E69F00")
    ax1.set_title("CV diagnostic per fold (in-support regime)")
    ax2.plot(fold_sizes, fold_logloss, "o-", color="#0072B2")
    ax2.set_xlabel("training rows (expanding window)")
    ax2.set_ylabel("fold validation logloss")
    ax2.set_title("Learning curve (time-series form)")
    _save(fig, out_dir, "cv_learning_curve")


def residual_analysis(y_true: np.ndarray, proba: np.ndarray, window_start_row: int,
                      out_dir: Path, name: str = "residual_analysis") -> None:
    """Probability residuals (p − y): where in time is probability mass
    misplaced, and how are errors distributed per class."""
    resid = proba - y_true
    rows = window_start_row + np.arange(len(resid))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    for cls, color, label in ((0, "#009E73", "Not Clogged"), (1, "#D55E00", "Clogged")):
        m = y_true == cls
        ax1.scatter(rows[m], resid[m], s=4, alpha=0.4, color=color, label=label,
                    rasterized=True)
        ax2.hist(resid[m], bins=40, orientation="horizontal", alpha=0.6, color=color)
    onset = np.flatnonzero(y_true == 1)
    if len(onset):
        ax1.axvline(window_start_row + onset[0], color="k", ls="--", lw=1)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("row (time order)")
    ax1.set_ylabel(r"probability residual $\hat{p} - y$")
    ax1.set_title("Residuals across the test window (dashed: onset)")
    ax1.legend()
    ax2.set_xlabel("count")
    ax2.set_title("residual distribution")
    _save(fig, out_dir, name)


def anticipation_curves(profiles: dict[str, list[dict]], out_dir: Path,
                        name: str = "anticipation_curves") -> None:
    """F1/precision vs anticipation window W for each model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for label, prof in profiles.items():
        ws = [p["anticipation_rows"] for p in prof]
        ax1.plot(ws, [p["f1"] for p in prof], "o-", label=label)
        ax2.plot(ws, [p["precision"] for p in prof], "o-", label=label)
    for ax, title in ((ax1, "F1"), (ax2, "Precision")):
        ax.set_xlabel("Anticipation window W (rows before onset excluded)")
        ax.set_title(f"{title} vs anticipation window")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    _save(fig, out_dir, name)


def horizon_summary(per_h: list[dict], out_dir: Path) -> None:
    """PR-AUC / recall / effective warning time vs horizon H."""
    hs = [d["h_rows"] for d in per_h]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(hs, [d["pr_auc"] for d in per_h], "o-", label="PR-AUC")
    ax1.plot(hs, [d["recall"] for d in per_h], "s-", label="Recall")
    ax1.plot(hs, [d["precision"] for d in per_h], "^-", label="Precision")
    ax1.set_xlabel("Horizon H (rows)"); ax1.set_ylim(0, 1.05); ax1.legend(fontsize=8)
    ax1.set_title("Early-warning performance vs horizon")
    ax2.plot(hs, [d.get("effective_warning_min") for d in per_h], "o-", color="#2ca02c")
    ax2.set_xlabel("Horizon H (rows)"); ax2.set_ylabel("Effective warning time (min)")
    ax2.set_title("Effective warning time = H·Δt − delay")
    _save(fig, out_dir, "horizon_summary")
