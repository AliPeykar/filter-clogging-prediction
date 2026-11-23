# Visualization & Reporting Plan

Audit of existing output figures, critique, and the complete publication set.
Conventions adopted from established open-source practice — Optuna's
`optuna.visualization` (optimization-history semantics), the SHAP repository
(beeswarm + dependence as the canonical pair), scikit-learn's model-evaluation
examples (reliability curves, PR/ROC presentation) — re-implemented in plain
matplotlib to keep dependencies pinned and styling uniform. All figures are
written as 300-dpi PNG **and** vector PDF per run under
`outputs/<run>/figures/`.

## 1. Audit of pre-existing material

**Legacy `legacy/visualizations/` (frozen with the legacy system):** static PNGs
(`sensor_analysis`, `confusion_matrices`, `model_performance`,
`feature_importance`, `hyperparameter_optimization`, `shap_summary`,
`risk_distribution_analysis`). Not publication-usable: they visualize the
leakage-contaminated runs (perfect scores → vacuous confusion matrices and
importances dominated by label-constituent features), carry no split/onset
annotations, and are not reproducible from any archived config. Superseded
wholesale.

**Corrected pipeline (before this plan):** timeline with split shading,
label-provenance exhibit, PR/ROC overlays, reliability curve, probability
trajectory with alarm annotation, binary + risk confusion matrices,
anticipation curves, horizon summary, SHAP beeswarm + dependence. Sound core,
but missing: dataset-exploration panels (distributions, correlations),
the physics figure (ΔP–Q with Forchheimer fits), HPO history, CV-fold view,
learning curve, residual/error analysis, cross-run comparison figure (the
function existed but was never invoked), and machine-readable summary tables.
This plan closes those gaps.

## 2. Complete figure set

Placement key: **T** = thesis, **P** = journal paper, **T+P** = both,
**S** = supplement/appendix.

### Dataset exploration & quality

| Figure | Why useful / scientific insight | Placement |
|---|---|---|
| `timeline_splits` — ΔP & Q vs time, onset line, shaded train/purge/val/test | Single-view of the experiment and the evaluation design; makes the held-out-transition logic auditable at a glance. Directly supports the split-design argument. | T+P |
| `label_provenance` — rule score vs time, clogged shading, empirical threshold (≥11) | Documents that the label is a deterministic rule output — the foundation of the leakage argument and of the honest task framing. | T+P |
| `eda_distributions` — histograms of ΔP, Q, R = ΔP/Q split by class | Shows class-conditional separation and the heavy right tail of Q (clean-filter excursions); justifies keeping "outliers" (they are physics, not noise) and motivates class-overlap discussion near the boundary. | T; S for P |
| `data_quality_panel` — missingness/duplicates/monotonicity summary (rendered from `label_audit.json` + validation) | Evidence of data integrity (0 missing, 0 duplicates, strict time order); replaces a missing-values figure that would be content-free here. | T (table form in P) |
| `correlation_heatmap` — Spearman among raw signals + key engineered features | Quantifies the near-duplication (ΔP ↔ EMAs r>0.99, CUSUM r≈0.98) that motivated banning precomputed columns, and shows engineered-feature redundancy structure. Spearman chosen: monotone-invariant, matches tree models. | T; S for P |

### Physics

| Figure | Why / insight | Placement |
|---|---|---|
| `dp_q_physics` — ΔP vs Q scatter colored by lifecycle phase, per-phase **Darcy operating lines** ΔP = R̄·Q (R̄ in Pa·s/m³) | The physics exhibit: rising operating-line slopes across phases show permeability decay directly, and the scatter shows the operating-line collapse at constant Q — *why* Forchheimer coefficients are unidentifiable here (fits archived as diagnostics: R² < 0, B < 0, confounded by within-phase degradation). Bridges thesis Ch. 3 physics to the ML feature set. | T+P |

### Model development

| Figure | Why / insight | Placement |
|---|---|---|
| `hpo_history` — val-logloss per Optuna trial + running best | Standard evidence the search converged; with the constrained space it also *shows* the flat landscape ("the space carries the regularization"). | T; S for P |
| `cv_fold_scores` — per-fold logloss / PR-AUC for the selected configuration | Documents the CV-regime mismatch honestly: in-support folds score near-ceiling while the transition-straddling test does not — the argument for val-based selection. | T; S for P |
| `learning_curve` — expanding-window training size vs fold validation score | Time-series-correct analogue of the classical learning curve (no shuffled subsets); shows whether performance is data-limited — relevant to the "collect more lifecycles" recommendation. | T; S for P |

### Evaluation (per horizon)

| Figure | Why / insight | Placement |
|---|---|---|
| `pr_roc_curves_h*` — XGB vs RF, PR + ROC | Threshold-free comparison; PR is the informative one at 38% prevalence. | T+P |
| `calibration_h*` — reliability curve raw vs Platt, ECE annotated | Shows probability quality and the documented calibration-transfer limitation under the covariate shift. | T; S for P |
| `prob_trajectory_h*` — calibrated P(clogged) across the test window, threshold, onset, sustained alarm, delay annotation | The single most important figure: converts abstract metrics into the operational story (alarm 35.8 min before onset). | T+P |
| `confusion_h*` | Standard; anchors the pointwise numbers. | T; S for P |
| `anticipation_curves_h0` — F1/precision vs anticipation window W, XGB vs RF | The key interpretive result: XGB's pointwise FPs are a contiguous pre-onset band (F1 0.706→0.999 by W=600) while RF errors are late detections. Justifies range-based evaluation (Tatbul et al., NeurIPS 2018). | T+P |
| `residual_analysis_h0` — probability residual (p − y) vs time + histogram by class | Classification analogue of residual plots: localizes *where* probability mass is misplaced (boundary band), complements the trajectory figure; reveals any systematic bias. | T; S for P |
| `horizon_summary` — PR-AUC/recall/precision and effective warning time vs H | Shows predictability decay with lead time; supports the "clogging time prediction" thesis objective. | T+P |
| `confusion_risk_levels` + risk metrics table | Secondary 6-level task with ordinal metrics (QWK, level-MAE, ±1 accuracy). | T; S for P |

### Interpretation

| Figure | Why / insight | Placement |
|---|---|---|
| `shap_beeswarm` | Global attribution; verifies the model uses physics-consistent signals (flow collapse, resistance level) rather than artifacts. | T+P |
| `shap_dependence_<top3>` | Shape of the top effects; checks monotone-constraint compliance visually. | T; S for P |

### Comparison & tables (cross-run, `compare_runs.py`)

| Artifact | Why / insight | Placement |
|---|---|---|
| `ablation_comparison` — grouped bars with 95% block-bootstrap CIs: leakage-free vs thesis-full × XGB vs RF | The paper's central methodological exhibit: quantifies label-circularity inflation (ROC 0.969 → 1.000) with uncertainty. | T+P |
| `summary_tables` — headline metrics (+CIs), delay sensitivity, anticipation profile, risk metrics — CSV **and** LaTeX (`booktabs`) | Machine-readable + drop-in for both manuscripts; eliminates hand-transcription errors. | T+P |

### Deliberately omitted, with reasons

- **Missing-values matrix / imputation plots** — dataset has zero missing
  values and zero duplicates; a `data_quality_panel` states this with the
  checks that prove it. An empty missingness matrix would be filler.
- **Classical (shuffled-subset) learning curves** — invalid for
  autocorrelated single-series data; replaced by the expanding-window form.
- **Regression prediction-vs-actual / residual scatter** — task is
  classification; the analogues provided are the probability trajectory and
  probability-residual figures.
- **Forchheimer coefficient estimates (rolling or per-phase)** — rolling
  fits are ill-conditioned at constant flow, and per-phase global fits are
  confounded by within-phase degradation (both verified: R² < 0, B < 0;
  diagnostics archived per run). Replaced by Darcy operating lines in
  `dp_q_physics`.

## 3. Style conventions

Serif (STIX) fonts sized for two-column print, colorblind-safe Okabe–Ito
palette, no chartjunk, every axis labeled with units (ΔP [kPa], Q [m³/h],
R [Pa·s/m³] where SI matters), consistent model colors across all figures
(XGB blue #0072B2, RF orange #E69F00), onset always the same dashed black
line. Set globally in `figures.py` rcParams.
