"""Manuscript figures (data figures only). Reads the dataset and the JSON/CSV
outputs of analyses A1-A3 and the re-run directories; writes PNG (600 dpi)
and PDF to paper/04_manuscript/figures/.

Usage: python make_figures.py [fig2 fig3 ...]   (default: all available)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
FIG = ROOT / "paper" / "04_manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- style: one system for every figure -----------------------------------
MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM  # single / double column width
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e4e4e4"
C = {  # colour follows the entity in every figure (Okabe-Ito, validated)
    "xgb": "#0072B2", "rf": "#E69F00", "dp": "#009E73", "rule": "#D55E00",
}
NAME = {"xgb": "XGBoost", "rf": "Random Forest", "dp": "ΔP threshold", "rule": "Rule label"}
SEG = {"train": "#dfe7f1", "val": "#f6e3c8", "test": "#d9ecd9"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42,
    "lines.linewidth": 1.0,
})

ONSET, JUMP = 8943, 8340  # rule onset row; ΔP step row (both verified in A1)
SEGMENTS = {"train_a": (1, 7140), "val_a": (7201, 7800), "test": (7861, 9600),
            "val_b": (9661, 10260), "train_b": (10321, 11519)}


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def panel_label(ax, s: str) -> None:
    ax.text(-0.01, 1.04, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            ha="right", va="bottom")


def load_data() -> pd.DataFrame:
    return pd.read_excel(ROOT / "Comprehensive_Filter_Analysis.xlsx")


def a1() -> dict:
    return json.loads((OUT / "a1_results.json").read_text())


# ---------------------------------------------------------------- Fig. 1
def fig1() -> None:
    """Block schematic of the rig, in the flow order given in the thesis."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(W2, 58 * MM))
    ax.set_xlim(0, 190)
    ax.set_ylim(0, 58)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=MUTED, fs=5.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, col=INK, lw=0.9, style="-|>"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color=col, lw=lw,
                                    shrinkA=0, shrinkB=0, mutation_scale=7))

    y, h = 30, 13
    items = [("Compressed air\nregulator\n1 bar", 22), ("Needle\nvalve", 13),
             ("Expansion\nto DN50", 17), ("Flow\nconditioner", 19),
             ("Orifice plate\nd = 6.9 mm\ntaps D, D/2", 20), ("Powder\ninjection\nport", 18),
             ("Filter housing\ncartridge\nsection", 24), ("Rotameter", 17)]
    gap = (190 - sum(w for _, w in items)) / (len(items) - 1 + 1.2)
    x = 0.0
    centres = {}
    for i, (t, w) in enumerate(items):
        fc = "#eef3f8" if i in (4, 6) else "white"
        box(x, y, w, h, t, fc=fc)
        centres[i] = (x, w)
        if i < len(items) - 1:
            arrow(x + w, y + h / 2, x + w + gap, y + h / 2)
        x += w + gap
    arrow(x - gap, y + h / 2, x - gap + 1.2 * gap, y + h / 2)
    ax.text(189.5, y - 1.5, "outlet", ha="right", va="top", fontsize=5.7, color=MUTED)

    # sensors and logger (signal lines dashed)
    for i, lab in [(4, "ΔP sensor\nMPX5100DP"), (6, "ΔP sensor\nMPX5100DP")]:
        cx = centres[i][0] + centres[i][1] / 2
        box(cx - 11, 9, 22, 10, lab)
        for dx in (-4, 4):
            ax.plot([cx + dx, cx + dx], [19, y], color=MUTED, lw=0.6)
        ax.plot([cx, cx], [9, 4.5], color=MUTED, lw=0.6, ls=(0, (2, 2)))
    cx5 = centres[4][0] + centres[4][1] / 2
    cx7 = centres[6][0] + centres[6][1] / 2
    ax.plot([cx5, cx7], [4.5, 4.5], color=MUTED, lw=0.6, ls=(0, (2, 2)))
    box(52, 0.5, 44, 8, "Arduino UNO R3  →  PC logger")
    ax.plot([96, cx5], [4.5, 4.5], color=MUTED, lw=0.6, ls=(0, (2, 2)))

    # powder feed
    cx6 = centres[5][0] + centres[5][1] / 2
    box(cx6 - 21, 48, 42, 9, "Piston syringe + check valve\nblack powder")
    arrow(cx6, 48, cx6, y + h)

    ax.text(0, 47.5, "air flow  →", fontsize=5.7, color=MUTED)
    ax.plot([0, 6], [2.5, 2.5], color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.text(7.5, 2.5, "signal", fontsize=5.7, color=MUTED, va="center")
    ax.plot([24, 30], [2.5, 2.5], color=MUTED, lw=0.6)
    ax.text(31.5, 2.5, "pressure taps", fontsize=5.7, color=MUTED, va="center")
    save(fig, "fig1_rig_schematic")


# ---------------------------------------------------------------- Fig. 2
def fig2() -> None:
    """Run overview: signals, rule score, split."""
    df = load_data()
    rows = np.arange(1, len(df) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(W2, 105 * MM), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 0.8, 1.0], "hspace": 0.16})
    for ax in axes:
        for k, (lo, hi) in SEGMENTS.items():
            ax.axvspan(lo, hi, color=SEG[k.split("_")[0]], lw=0, zorder=0)
        ax.axvline(ONSET, color=C["rule"], lw=0.9, ls=(0, (4, 2)), zorder=3)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=1)
        ax.set_axisbelow(True)

    ax = axes[0]
    ax.plot(rows, df["dp"], color=INK, lw=0.7, zorder=4)
    ax.set_ylabel("ΔP (kPa)")
    ax.set_ylim(0, 66)
    ax.text(ONSET - 90, 59, "rule onset\nsample 8,943", color=C["rule"], ha="right",
            va="top", fontsize=7)
    y_top = 1.03
    for k, lab in [("train_a", "Train-A"), ("val_a", "Val-A"), ("test", "Test"),
                   ("val_b", "Val-B"), ("train_b", "Train-B")]:
        lo, hi = SEGMENTS[k]
        ax.text((lo + hi) / 2, y_top, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color=INK)
    panel_label(ax, "a")

    ax = axes[1]
    q = df["flowrate"].to_numpy()
    ax.plot(rows, np.clip(q, None, 11), color=INK, lw=0.6, zorder=4)
    spikes = rows[q > 11]
    ax.plot(spikes, np.full(len(spikes), 10.9), ls="none", marker="^", ms=2.2,
            color=MUTED, zorder=5)
    ax.set_ylim(4.8, 11.3)
    ax.set_ylabel("Q (m³ h⁻¹)")
    ax.text(0.999, 1.03, f"▲ {len(spikes)} samples above the axis limit (max {q.max():.1f})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=MUTED)
    panel_label(ax, "b")

    ax = axes[2]
    ax.step(rows, df["validation_score"], where="post", color=INK, lw=0.8, zorder=4)
    ax.axhline(11, color=C["rule"], lw=0.8, ls=(0, (1, 1.5)), zorder=3)
    ax.text(150, 11.3, "label threshold in the stored data: score ≥ 11", color=C["rule"],
            fontsize=6.8, va="bottom")
    for r in (7001, 7501):
        ax.annotate("", xy=(r, df["validation_score"].iloc[r - 1] + 0.15), xytext=(6150, 13.4),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.text(6150, 13.5, "score +1 at samples 7,001 and 7,501, no rule input changes", fontsize=6.5,
            color=MUTED, ha="right", va="bottom")
    ax.set_ylim(-0.5, 15.5)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_ylabel("Rule score (0–20)")
    ax.set_xlabel("Sample index")
    ax.set_xlim(1, len(df))
    panel_label(ax, "c")

    handles = [Patch(color=SEG["train"], label="training"),
               Patch(color=SEG["val"], label="validation"),
               Patch(color=SEG["test"], label="test"),
               Patch(color="white", ec=MUTED, lw=0.5, label="purged (60 samples)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.035),
               handlelength=1.2, columnspacing=1.6)
    save(fig, "fig2_run_overview")


# ---------------------------------------------------------------- Fig. 3
def fig3() -> None:
    """F1 under each evaluation design (same data throughout)."""
    a3 = json.loads((OUT / "a3_results.json").read_text())
    r = a1()
    runs = ROOT / "paper" / "01_reproduction" / "runs"

    def run_metric(prefix: str) -> dict:
        d = sorted(runs.glob(prefix + "_2026*"))[-1]
        return json.loads((d / "metrics.json").read_text())["horizons"]["h0"]["xgb"]["test_metrics"]

    abl, strict = run_metric("ablation_thesis_features"), run_metric("strict_temporal")
    rows = [
        ("Temporal 80/20 hold-out\n(test set is one class)", a3["legacy_80_20_temporal"]["f1"], "leak"),
        ("Past→future split, added here\n(test set is one class)", strict["f1"], "leak"),
        ("Rule-derived columns as predictors\n(held-out-transition split)", abl["f1"], "leak"),
        ("Random stratified 80/20 hold-out", a3["random_stratified_80_20_holdout"]["f1"], "leak"),
        ("Shuffled stratified 5-fold", a3["shuffled_stratified_5fold"]["mean_f1"], "leak"),
        ("Corrected pipeline, Random Forest", r["model_behaviour"]["rf"]["f1"], "rf"),
        ("Corrected pipeline, XGBoost", r["model_behaviour"]["xgb"]["f1"], "xgb"),
        ("ΔP threshold at mid-gap, no model", r["univariate_baselines"]["dp"]["thresholds"]["val_midpoint"]["f1"], "dp"),
    ]
    col = {"leak": INK, "rf": C["rf"], "xgb": C["xgb"], "dp": C["dp"]}
    fig, ax = plt.subplots(figsize=(W2 * 0.78, 74 * MM))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v, k) in zip(ys, rows):
        ax.plot([0.5, v], [y, y], color=GRID, lw=1.0, zorder=1)
        ax.plot(v, y, "o", ms=5.5, color=col[k], mec="white", mew=0.8, zorder=3)
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", ha="left", fontsize=7)
    ax.axhline(2.5, color=MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.text(0.505, 2.62, "leakage mechanisms of the legacy pipeline, each in isolation ↑",
            fontsize=6.5, color=MUTED, va="bottom")
    ax.text(0.505, 2.38, "evaluation on the held-out-transition split ↓", fontsize=6.5, color=MUTED, va="top")
    ax.set_yticks(ys)
    ax.set_yticklabels([r_[0] for r_ in rows])
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.5, 1.07)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("F1 for the clogged class on each design's test data")
    save(fig, "fig3_evaluation_designs")


# ---------------------------------------------------------------- Fig. 4
def fig4() -> None:
    """Test window: both signals, and where each detector flags, relative to
    the step at sample 8,340 and the rule onset."""
    t = pd.read_csv(OUT / "test_window_predictions.csv")
    r = a1()
    r4 = json.loads((OUT / "a4_results.json").read_text())
    thr_x = r["model_behaviour"]["xgb"]["threshold"]
    thr_r = r["model_behaviour"]["rf"]["threshold"]
    thr_dp = r["univariate_baselines"]["dp"]["thresholds"]["val_midpoint"]["threshold"]
    lo, hi = r["split_support"]["dp_gap_unobserved_in_train_and_val"]
    q_min_h = r4["support_2d"]["healthy_fit_q_min"]
    x = t["row"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(W2, 128 * MM), sharex=True,
                             gridspec_kw={"height_ratios": [1.45, 0.8, 0.95], "hspace": 0.13})
    for a in axes:
        for row, col in [(JUMP, MUTED), (ONSET, C["rule"])]:
            a.axvline(row, color=col, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax = axes[0]
    ax.axhspan(lo, hi, color="#eeeeee", lw=0, zorder=0)
    ax.text(7885, hi - 0.3, "shaded: ΔP band with no training or\nvalidation samples "
            f"({lo:.1f}–{hi:.1f} kPa)", fontsize=6.8, color=MUTED, va="top", ha="left")
    ax.plot(x, t["dp"], color=INK, lw=0.8, zorder=3)
    ax.axhline(thr_dp, color=C["dp"], lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.text(9590, thr_dp + 0.3, f"mid-gap threshold {thr_dp:.1f} kPa", color=INK,
            fontsize=6.8, ha="right", va="bottom")
    ax.text(JUMP - 12, 31.3, "step, sample 8,340", ha="right", va="top", fontsize=7, color=MUTED)
    ax.text(ONSET + 12, 15.2, "rule onset\nsample 8,943", ha="left", va="bottom",
            fontsize=7, color=C["rule"])
    ax.annotate("", xy=(ONSET, 16.6), xytext=(JUMP, 16.6),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.6))
    ax.text((JUMP + ONSET) / 2, 16.85, "603 samples", ha="center", va="bottom",
            fontsize=6.8, color=MUTED)
    for row, yy in [(8000, 15.6), (8115, 14.25)]:
        ax.annotate("ΔP drop", xy=(row, t["dp"].iloc[row - x[0] + 1]), xytext=(row - 70, yy),
                    fontsize=6.5, color=MUTED, ha="center", va="top",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.set_ylabel("ΔP (kPa)")
    ax.set_ylim(13.2, 31.5)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    ax = axes[1]
    qq = t["flowrate"].to_numpy()
    ax.axhspan(5.9, q_min_h, color="#eeeeee", lw=0, zorder=0)
    ax.plot(x, np.clip(qq, None, 7.35), color=INK, lw=0.7, zorder=3)
    sp = x[qq > 7.35]
    ax.plot(sp, np.full(len(sp), 7.33), ls="none", marker="^", ms=2.2, color=MUTED, zorder=4)
    ax.text(7885, 6.03, f"shaded: Q below every healthy training and validation sample (< {q_min_h:.2f} m³ h⁻¹)",
            fontsize=6.8, color=MUTED, va="bottom", ha="left")
    ax.set_ylim(5.95, 7.4)
    ax.set_ylabel("Q (m³ h⁻¹)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.999, 1.03, f"▲ {len(sp)} spikes above the axis limit", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color=MUTED)
    panel_label(ax, "b")

    ax = axes[2]
    lanes = [
        ("rule", t["y"].to_numpy() == 1, "Rule label"),
        ("dp", t["dp"].to_numpy() >= thr_dp, "ΔP ≥ mid-gap"),
        ("rf", t["p_rf"].to_numpy() >= thr_r, "Random Forest"),
        ("xgb", t["p_xgb"].to_numpy() >= thr_x, "XGBoost"),
    ]
    for i, (key, flag, lab) in enumerate(lanes):
        d = np.diff(np.concatenate([[0], flag.astype(int), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        for s_, e_ in zip(starts, ends):
            ax.barh(i, x[e_ - 1] - x[s_] + 1, left=x[s_], height=0.56, color=C[key], lw=0)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([l[2] for l in lanes])
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.set_xlabel("Sample index (test window)")
    ax.set_xlim(x[0], x[-1])
    ax.spines["left"].set_visible(False)
    panel_label(ax, "c")
    ax.text(0.999, 1.02, "bars: samples flagged as clogged", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.8, color=MUTED)
    save(fig, "fig4_test_window")


# ---------------------------------------------------------------- Fig. 5
def fig5() -> None:
    """(a) joint (Q, ΔP) support of fitting data and test window; (b), (c) test
    F1 of single-variable thresholds on ΔP and on Q."""
    df = load_data()
    r = a1()
    r4 = json.loads((OUT / "a4_results.json").read_text())
    lo, hi = r["split_support"]["dp_gap_unobserved_in_train_and_val"]
    q_min_h = r4["support_2d"]["healthy_fit_q_min"]
    dp, q = df["dp"].to_numpy(), df["flowrate"].to_numpy()
    y = (df["filter_status"] == "Clogged").to_numpy()
    rows = np.arange(1, len(df) + 1)
    fit = np.zeros(len(df), bool)
    for k in ("train_a", "val_a", "val_b", "train_b"):
        a, b = SEGMENTS[k]
        fit[a - 1:b] = True
    a, b = SEGMENTS["test"]
    test = (rows >= a) & (rows <= b)
    sw = pd.read_csv(OUT / "dp_threshold_sweep.csv")
    qs = pd.read_csv(OUT / "q_threshold_sweep.csv")
    xg, rf = r["model_behaviour"]["xgb"]["f1"], r["model_behaviour"]["rf"]["f1"]

    fig = plt.figure(figsize=(W2, 78 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], wspace=0.3, hspace=0.62)
    ax = fig.add_subplot(gs[:, 0])
    ax.axhspan(lo, hi, color="#eeeeee", lw=0, zorder=0)
    ax.axvline(q_min_h, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    kw = dict(lw=0, rasterized=True)
    m = fit & ~y
    ax.scatter(q[m][::6], dp[m][::6], s=3, color="#b9b9b9", zorder=2, label="training/validation, healthy", **kw)
    m = fit & y
    ax.scatter(q[m][::3], dp[m][::3], s=3, color="#eab08a", zorder=2, label="training/validation, clogged", **kw)
    groups = [(test & (rows < JUMP), "o", "none", "test, before the step"),
              (test & (rows >= JUMP) & (rows < ONSET), "s", INK, "test, step to rule onset"),
              (test & (rows >= ONSET), "x", INK, "test, after the rule onset")]
    for m, mk, fc, lab in groups:
        if mk == "x":
            ax.scatter(q[m][::3], dp[m][::3], s=7, marker=mk, color=INK, lw=0.5, zorder=4, label=lab,
                       rasterized=True)
        else:
            ax.scatter(q[m][::3], dp[m][::3], s=6, marker=mk, facecolors=fc, edgecolors=INK, lw=0.4,
                       zorder=4, label=lab, rasterized=True)
    ax.set_xlim(5.1, 7.6)
    ax.set_ylim(0, 64)
    ax.set_xlabel("Q (m³ h⁻¹), spikes above 7.6 not shown")
    ax.set_ylabel("ΔP (kPa)")
    ax.text(q_min_h + 0.02, 62, f"healthy training\nminimum Q = {q_min_h:.2f}", fontsize=6.3,
            color=MUTED, va="top", ha="left")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.17), ncol=2, handletextpad=0.2,
              columnspacing=1.0, borderaxespad=0, fontsize=6.3, markerscale=1.6)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax.axvspan(lo, hi, color="#eeeeee", lw=0, zorder=0)
    ax.plot(sw["threshold"], sw["f1"], color=C["dp"], lw=1.4, zorder=3)
    for val, key in [(xg, "xgb"), (rf, "rf")]:
        ax.axhline(val, color=C[key], lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.text(24.0, xg - 0.02, f"XGBoost {xg:.3f}", ha="center", va="top", fontsize=6.3)
    ax.text(17.15, rf - 0.02, f"Random Forest {rf:.3f}", ha="left", va="top", fontsize=6.3)
    ax.set_xlabel("Threshold on ΔP (kPa): flag if ΔP ≥ threshold", fontsize=7)
    ax.set_ylabel("Test F1")
    ax.set_xlim(17, 30)
    ax.set_ylim(0.5, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    ax.axvspan(5.8, q_min_h, color="#eeeeee", lw=0, zorder=0)
    ax.plot(qs["q_threshold"], qs["f1"], color=INK, lw=1.4, zorder=3)
    for val, key in [(xg, "xgb"), (rf, "rf")]:
        ax.axhline(val, color=C[key], lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.set_xlabel("Threshold on Q (m³ h⁻¹): flag if Q ≤ threshold", fontsize=7)
    ax.set_ylabel("Test F1")
    ax.set_xlim(5.8, 7.2)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel_label(ax, "c")
    save(fig, "fig5_support_and_threshold_sweep")


# ---------------------------------------------------------------- Fig. 6
def fig6() -> None:
    """ROC curves on the test window."""
    t = pd.read_csv(OUT / "test_window_predictions.csv")
    r = a1()
    y = t["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(W1, 80 * MM))
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, zorder=1)
    items = [("dp", t["dp"], r["univariate_baselines"]["dp"]["roc_auc"], "Raw ΔP (no model)"),
             ("xgb", t["p_xgb"], r["model_behaviour"]["xgb"]["roc_auc"], "XGBoost"),
             ("rf", t["p_rf"], r["model_behaviour"]["rf"]["roc_auc"], "Random Forest")]
    for key, s, auc, lab in items:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=C[key], lw=1.4, label=f"{lab}, AUC {auc:.3f}", zorder=3)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", handlelength=1.6)
    save(fig, "fig6_roc")


# ---------------------------------------------------------------- Fig. S2
def figS2() -> None:
    """Validation loss versus test ROC-AUC, one point per Optuna trial."""
    dn = pd.read_csv(OUT / "search_trials_narrow.csv")
    dw = pd.read_csv(OUT / "search_trials_wide.csv")
    fig, ax = plt.subplots(figsize=(W1 * 1.25, 72 * MM))
    ax.scatter(dw["val_logloss"], dw["test_roc_auc"], s=14, marker="o", facecolors="none",
               edgecolors=INK, lw=0.6, label="wide search space (75 trials)", zorder=3)
    ax.scatter(dn["val_logloss"], dn["test_roc_auc"], s=12, marker="s", color=C["xgb"], lw=0,
               label="narrow search space (75 trials)", zorder=4)
    b = dn.loc[dn["val_logloss"].idxmin()]
    ax.annotate("selected in the main run", xy=(b["val_logloss"], b["test_roc_auc"]),
                xytext=(0.075, 0.90), fontsize=6.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.axhline(0.9927, color=C["dp"], lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.text(0.099, 0.9945, "raw ΔP, no model", fontsize=6.5, ha="right", va="bottom")
    ax.set_xlabel("Validation log-loss (the tuning objective)")
    ax.set_ylabel("Test-window ROC-AUC")
    ax.set_ylim(0.72, 1.01)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.5), handletextpad=0.3)
    save(fig, "figS2_validation_vs_test")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "figS2": figS2}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
