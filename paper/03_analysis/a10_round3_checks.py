"""Paper analysis A10: checks requested in review round 3.

1. (M3) Quantization of the logged flow near the healthy minimum: distinct
   logged Q values before sample 9,681 and their spacing.
2. (M6) Number of bootstrap resamples discarded because they contain one
   class, per block length (from out/a4_results.json, which ran the
   bootstrap of Table S6).
3. (m14) Where the Random Forest false negatives lie in the test window.
4. (M7) Label sensitivity: the onset is placed at the first test sample at
   which DeltaP reaches a level T (clogged from there on), for T swept over
   the unobserved band in 0.25 kPa steps (18.25-29.00 kPa). Predictions and
   thresholds of the main run are unchanged.

Run from paper/: python 03_analysis/a10_round3_checks.py
Output: 03_analysis/out/a10_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PAPER = HERE.parent
DATA = next(p for p in (PAPER.parent / "Comprehensive_Filter_Analysis.xlsx",
                        PAPER / "07_repository" / "Comprehensive_Filter_Analysis.xlsx") if p.exists())
TH_XGB, TH_RF, MIDGAP = 0.9333256940020761, 0.962581617734156, 23.0   # main run
ONSET = 8943


def runs(rows: np.ndarray) -> list[list[int]]:
    """Contiguous runs of sample numbers as [first, last]."""
    if len(rows) == 0:
        return []
    cut = np.where(np.diff(rows) > 1)[0]
    starts = np.r_[rows[0], rows[cut + 1]]
    ends = np.r_[rows[cut], rows[-1]]
    return [[int(a), int(b)] for a, b in zip(starts, ends)]


def main() -> None:
    res = {}
    df = pd.read_excel(DATA, sheet_name="Comprehensive_Analysis")
    q = df.flowrate.to_numpy(float)

    # 1. flow quantization (entered 130 kPa stretch, before sample 9,681)
    qs = q[:9680]
    lv = np.unique(np.round(qs[(qs > 6.45) & (qs < 6.85)], 3))
    res["q_levels_6p45_6p85_before_9681"] = lv.tolist()
    main_lv = lv[np.isin(np.round(lv, 3), [6.487, 6.526, 6.565, 6.603, 6.642, 6.68, 6.717, 6.755, 6.792])]
    res["q_level_spacing_near_6p6"] = [round(float(x), 3) for x in np.diff(main_lv)]
    res["q_squared_step_per_count_median"] = round(float(np.median(np.diff(main_lv ** 2))), 3)

    # 2. bootstrap resamples with one class (Table S6 bootstrap, A4)
    a4 = json.loads((OUT / "a4_results.json").read_text(encoding="utf-8"))
    res["bootstrap_valid_resamples_of_1000"] = {k: v["n_valid"] for k, v in a4["block_length_sensitivity"].items()}

    # 3 and 4 on the test window of the main run
    tw = pd.read_csv(OUT / "test_window_predictions.csv")
    rows, y, dp = tw.row.to_numpy(), tw.y.to_numpy(), tw.dp.to_numpy()
    fx = (tw.p_xgb >= TH_XGB).to_numpy().astype(int)
    fr = (tw.p_rf >= TH_RF).to_numpy().astype(int)
    fm = (dp >= MIDGAP).astype(int)
    fn_rf = rows[(y == 1) & (fr == 0)]
    res["rf_false_negative_runs"] = runs(fn_rf)
    res["rf_false_negatives_dp_range"] = [float(dp[np.isin(rows, fn_rf)].min()), float(dp[np.isin(rows, fn_rf)].max())]

    sweep = []
    for T in np.round(np.arange(18.25, 29.0001, 0.25), 2):
        hit = np.where(dp >= T)[0]
        onset = int(rows[hit[0]])
        yl = (rows >= onset).astype(int)
        f = {k: round(float(f1_score(yl, p, zero_division=0)), 3) for k, p in (("xgb", fx), ("rf", fr), ("midgap", fm))}
        best = max(f, key=f.get)
        sweep.append({"T": float(T), "onset": onset, **f, "best": best})
    res["onset_at_dp_level_sweep"] = sweep
    for k in ("xgb", "rf", "midgap"):
        v = [s[k] for s in sweep]
        res[f"sweep_f1_range_{k}"] = [min(v), max(v)]
    res["sweep_best_counts"] = {k: sum(s["best"] == k for s in sweep) for k in ("xgb", "rf", "midgap")}
    res["sweep_best_by_T"] = runs_by_best(sweep)
    (OUT / "a10_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "onset_at_dp_level_sweep"}, indent=1))


def runs_by_best(sweep):
    out, cur = [], None
    for s in sweep:
        if cur and cur["best"] == s["best"]:
            cur["T_to"] = s["T"]; cur["onset_to"] = s["onset"]
        else:
            cur = {"best": s["best"], "T_from": s["T"], "T_to": s["T"], "onset_from": s["onset"], "onset_to": s["onset"]}
            out.append(cur)
    return out


if __name__ == "__main__":
    main()
