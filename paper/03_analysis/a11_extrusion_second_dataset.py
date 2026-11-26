"""Paper analysis A11: checks of Table 4 applied to a second, public data set.

Data: University of Melbourne extrusion-filter RUL data set (Ferreira, Ko,
Li, Otto, Data in Brief 69 (2026) 113254; Zenodo 10.5281/zenodo.21987832,
CC BY 4.0), file "Extruder Parameters.zip" (md5 0f3701bb7eef6c5265cdb1613cfdfe77):
60 run-to-failure runs of a steel-mesh filter in a single-screw polymer
extruder, 1 Hz motor volts and amps and four barrel temperatures. Motor
current is the authors' proxy for the unmeasured filter pressure. The
Labels.csv named in the article is not in the Zenodo record, so failure mode
and lifetime per run are not available; recording stopped about 10 s after
the observed failure.

Target used here (no label file needed): a sample is "near end of life" if it
lies within the last H seconds of its run's recording (H = 30, 60, 120 s;
main H = 60 s).

Checks:
  C1 class composition: a per-run 80/20 time split; how many runs give a
     single-class test part.
  C3 resampling: Random Forest on causal features, scored on a shuffled
     80/20 sample split and with run-level 5-fold cross-validation.
  C4 signal baselines: motor current alone, and current minus its
     run-start level, as scores without a model, under the run-level folds.
  C5 support: share of held-out samples whose current lies outside the
     training range of the fold.
  C7 target placement: H varied; ranking of model and signal baselines.

Run from paper/: python 03_analysis/a11_extrusion_second_dataset.py
Output: 03_analysis/out/a11_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "08_extrusion" / "extracted" / "Extruder Parameters"
SEED = 42
FEATS = ["amp", "volt", "power", "amp_m10", "amp_m30", "amp_slope30", "amp_rise", "t_front", "t_mid"]


def load() -> pd.DataFrame:
    parts = []
    for f in sorted(DATA.glob("Run*_extruder.csv"), key=lambda p: int(p.stem[3:].split("_")[0])):
        run = int(f.stem[3:].split("_")[0])
        d = pd.read_csv(f)
        d = d.rename(columns={"Motor Volt(V)": "volt", "Motor Amp(A)": "amp",
                              "Front Temp(C)": "t_front", "Middle Temp(C)": "t_mid"})
        d["run"] = run
        d["t"] = np.arange(len(d))
        d["t_to_end"] = len(d) - 1 - d["t"]
        a = d["amp"]
        d["power"] = d["volt"] * d["amp"]
        d["amp_m10"] = a.rolling(10, min_periods=1).mean()
        d["amp_m30"] = a.rolling(30, min_periods=1).mean()
        d["amp_slope30"] = a.rolling(30, min_periods=2).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True).fillna(0.0)
        # current minus the run's own start level (first 10 s, causal after 10 s)
        start = a.expanding().mean().where(d["t"] < 10, a.iloc[:10].mean())
        d["amp_rise"] = a - start
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def scores(y, s):
    return round(float(roc_auc_score(y, s)), 3) if len(np.unique(y)) == 2 else None


def evaluate(df: pd.DataFrame, H: int) -> dict:
    y = (df["t_to_end"] < H).astype(int).to_numpy()
    X = df[FEATS].to_numpy()
    g = df["run"].to_numpy()
    out = {"H_s": H, "positive_share": round(float(y.mean()), 3)}

    # C3a shuffled sample split
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
    rf = RandomForestClassifier(300, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
    p = rf.predict_proba(X[te])[:, 1]
    out["random_split"] = {"rf_roc_auc": scores(y[te], p), "rf_f1": round(float(f1_score(y[te], p >= 0.5)), 3),
                           "test_rows_from_training_runs_pct": 100.0}

    # C3b run-level 5-fold cross-validation (pooled out-of-fold predictions)
    oof = np.zeros(len(y)); outside = []
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
        rf = RandomForestClassifier(300, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
        oof[te] = rf.predict_proba(X[te])[:, 1]
        a_tr, a_te = df["amp"].to_numpy()[tr], df["amp"].to_numpy()[te]
        outside.append(float(((a_te < a_tr.min()) | (a_te > a_tr.max())).mean()))
    per_run = [scores(y[g == r], oof[g == r]) for r in np.unique(g)]
    per_run = [v for v in per_run if v is not None]
    out["run_level_cv"] = {
        "rf_roc_auc": scores(y, oof), "rf_f1": round(float(f1_score(y, oof >= 0.5)), 3),
        "rf_roc_auc_within_run_median": round(float(np.median(per_run)), 3),
        "held_out_amp_outside_training_range_pct": round(100 * float(np.mean(outside)), 2),
    }
    # C4 signal baselines (no model)
    out["baselines_pooled_roc_auc"] = {"amp": scores(y, df["amp"]), "amp_m30": scores(y, df["amp_m30"]),
                                       "amp_rise": scores(y, df["amp_rise"]), "power": scores(y, df["power"])}
    within = lambda col: float(np.median([v for v in (scores(y[g == r], df[col].to_numpy()[g == r])
                                                       for r in np.unique(g)) if v is not None]))
    out["baselines_within_run_median_roc_auc"] = {c: round(within(c), 3) for c in ("amp", "amp_m30", "amp_rise")}
    return out


def main() -> None:
    df = load()
    runs = df.groupby("run").size()
    res = {"source": "Zenodo 10.5281/zenodo.21987832, Extruder Parameters.zip", "n_runs": int(len(runs)),
           "n_rows": int(len(df)), "run_length_s_min_median_max": [int(runs.min()), int(runs.median()), int(runs.max())],
           "features": FEATS}
    # C1 per-run 80/20 time split, H = 60 s
    single = 0
    for r, n in runs.items():
        cut = int(np.floor(0.8 * n))
        te_t = np.arange(cut, n); te_y = (n - 1 - te_t) < 60
        single += int(te_y.all() or (~te_y).all())
    res["per_run_80_20_time_split_single_class_test_runs_H60"] = single
    res["by_H"] = [evaluate(df, H) for H in (30, 60, 120)]
    (HERE / "out" / "a11_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
