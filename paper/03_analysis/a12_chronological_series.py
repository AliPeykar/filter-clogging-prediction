"""Paper analysis A12: the corrected pipeline on a chronological series.

Review round 3, comment M4. The analysed data set joins seven raw logs, with
874 log rows removed (847 with Q above 10 m3/h, at the converter limit of the
orifice channel), a repeated block (samples 8,191-8,339 = 6,793-6,941) and 10
samples that match no log row. Here the series is rebuilt from the raw logs:

  * for each of the seven logs, every row between the first and the last row
    used in the data set, in time order (no repeated block, removed rows
    restored, the 1,777 unused rows at the starts and ends of the logs left
    out, as in the data set);
  * variant ALL keeps every such row; variant MEAS drops rows with Q above
    10 m3/h (flow above the measurable range of the orifice channel), a rule
    applied to the whole series.

Label (no sample-index gate, no look-ahead): clogged from the first row with
Q <= 10 m3/h and dP > 21.776 kPa (the rule's level, baseline + 5 SD) onward.
Segments: the held-out-transition split of the paper, mapped by wall-clock
time: each segment holds the rows logged between the first and the last
sample of the original segment; the rows in between are purge gaps.

Pipeline as in the main configuration (seed 42, 75 TPE trials, monotone
constraints, Platt scaling, max-F1 threshold), with modules from
07_repository/src. Baselines: raw dP as a score, the mid-gap dP threshold
from the validation set, and a healthy-envelope limit on dP set on Train-A
and Val-A.

Inputs: raw logs in <THESIS>/data (THESIS_DIR or ../Thesis next to paper/),
out/a7_sample_provenance.csv. Output: out/a12_results.json
Run from paper/: python 03_analysis/a12_chronological_series.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
OUT = HERE / "out"
THESIS = Path(os.environ.get("THESIS_DIR", PAPER.parent / "Thesis"))
LOGS = next((p for p in (THESIS / "data", PAPER.parent / "raw_logs") if (p / "data2.csv").exists()), THESIS / "data")
FILES = ["data2.csv"] + [f"data2_{i}.csv" for i in range(2, 10)]
for cand in (PAPER.parent / "src", PAPER / "07_repository" / "src"):
    if (cand / "filterclog").exists():
        sys.path.insert(0, str(cand))
        break

from filterclog.calibration import PlattCalibrator, select_threshold_max_f1  # noqa: E402
from filterclog.evaluation import point_metrics  # noqa: E402
from filterclog.features import CausalFeatureEngineer  # noqa: E402
from filterclog.hpo import optimize_xgb  # noqa: E402
from filterclog.train import train_rf_baseline, train_xgb  # noqa: E402

SEED, LEVEL, QMAX = 42, 21.776, 10.0
SEGMENTS = {"Train-A": (1, 7140), "Val-A": (7201, 7800), "Test": (7861, 9600),
            "Val-B": (9661, 10260), "Train-B": (10321, 11519)}


def load_log(path: Path) -> pd.DataFrame:
    """As in A7: PuTTY log -> rows of the last session, with wall-clock time."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    start = dt.datetime.strptime(re.search(r"PuTTY log (\d{4}\.\d\d\.\d\d \d\d:\d\d:\d\d)", txt).group(1),
                                 "%Y.%m.%d %H:%M:%S")
    rows = []
    for line in txt.splitlines():
        p = line.split(",")
        if len(p) == 10:
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                pass
    df = pd.DataFrame(rows, columns=["t", "adc1", "v1", "dp1", "Q", "mdot", "P1", "adc2", "v2", "dp2"])
    r = np.where(np.diff(df.t.to_numpy()) < 0)[0]
    if len(r):
        df = df.iloc[r[-1] + 1:]
    df = df.copy()
    df["file"] = path.name
    df["wall"] = [start + dt.timedelta(seconds=float(x)) for x in df.t]
    return df


def build_series() -> tuple[pd.DataFrame, dict]:
    F = pd.concat([load_log(LOGS / f) for f in FILES], ignore_index=True)
    prov = pd.read_csv(OUT / "a7_sample_provenance.csv")
    ok = prov[prov.log_row >= 0]
    # consistency with A7: stored dP equals the log's dP channel at the mapped row
    match = float((np.round(F.dp2.to_numpy()[ok.log_row], 3) == np.round(ok.dp_stored, 3)).mean())
    spans = ok.groupby("file").log_row.agg(["min", "max"])
    parts = [F.iloc[a:b + 1] for a, b in spans.sort_values("min").itertuples(index=False)]
    S = pd.concat(parts).sort_values("wall", kind="stable").reset_index(drop=True)
    S = S.rename(columns={"Q": "flowrate", "dp2": "dp"})
    used = set(ok.log_row.tolist())
    info = {"dp_match_share": round(match, 4), "rows": int(len(S)),
            "rows_not_in_dataset": int(sum(1 for i in np.concatenate([np.arange(a, b + 1) for a, b in
                                                                       spans.itertuples(index=False)]) if i not in used)),
            "rows_Q_over_10": int((S.flowrate > QMAX).sum())}
    # wall-clock time of each original segment boundary
    wall = pd.to_datetime(prov.wall_clock)
    seg_t = {k: (wall[a - 1], wall[b - 1]) for k, (a, b) in SEGMENTS.items()}
    return S, {"info": info, "seg_t": seg_t}


def run(S: pd.DataFrame, seg_t: dict, name: str) -> dict:
    S = S.reset_index(drop=True)
    w = pd.to_datetime(S.wall)
    seg = {k: np.where((w >= a) & (w <= b))[0] for k, (a, b) in seg_t.items()}
    meas = S.flowrate.to_numpy() <= QMAX
    dp = S.dp.to_numpy()
    first = int(np.argmax(meas & (dp > LEVEL)))
    y = (np.arange(len(S)) >= first).astype(int)

    eng = CausalFeatureEngineer("leakage_free", (5, 15, 30), (10, 50), 0.5)
    train = np.concatenate([seg["Train-A"], seg["Train-B"]])
    val = np.concatenate([seg["Val-A"], seg["Val-B"]])
    test = seg["Test"]
    eng.fit(S.iloc[train])
    Xf = eng.transform(S[["flowrate", "dp"]])
    X = Xf.to_numpy(dtype=np.float32)
    mono = eng.monotone_constraints(list(Xf.columns))
    warm = eng.warmup_rows
    tr, va, te = train[train >= warm], val[val >= warm], test

    hpo = optimize_xgb(X, y, tr, va, n_trials=75, n_cv_splits=4, cv_gap=60, seed=SEED,
                       max_n_estimators=2000, early_stopping_rounds=50, monotone=mono)
    model = train_xgb(hpo["best_params"], X[tr], y[tr], X[va], y[va], seed=SEED,
                      max_n_estimators=2000, early_stopping_rounds=50, monotone=mono)
    pv, pt = model.predict_proba(X[va])[:, 1], model.predict_proba(X[te])[:, 1]
    cal = PlattCalibrator().fit(pv, y[va])
    thr = select_threshold_max_f1(cal.transform(pv), y[va])
    fx = (cal.transform(pt) >= thr).astype(int)
    mx = point_metrics(y[te], cal.transform(pt), thr)

    rf = train_rf_baseline(X[tr], y[tr], SEED)
    pv_r, pt_r = rf.predict_proba(X[va])[:, 1], rf.predict_proba(X[te])[:, 1]
    cal_r = PlattCalibrator().fit(pv_r, y[va])
    thr_r = select_threshold_max_f1(cal_r.transform(pv_r), y[va])
    fr = (cal_r.transform(pt_r) >= thr_r).astype(int)
    mr = point_metrics(y[te], cal_r.transform(pt_r), thr_r)

    v_h, v_c = dp[val][y[val] == 0], dp[val][y[val] == 1]
    midgap = (v_h.max() + v_c.min()) / 2
    fm = (dp[te] >= midgap).astype(int)
    head = np.concatenate([seg["Train-A"], seg["Val-A"]])
    env = dp[head].max()
    fe = (dp[te] >= env).astype(int)

    wt = w.to_numpy()

    def ev(flags):
        idx = te[flags == 1]
        if len(idx) == 0:
            return {"first_flag_minus_onset_rows": None}
        f0 = int(idx[0])
        fp = int(((flags == 1) & (y[te] == 0)).sum())
        return {"first_flag_minus_onset_rows": f0 - first,
                "first_flag_minus_onset_min": round(float((wt[f0] - wt[first]) / np.timedelta64(1, "s") / 60), 1),
                "false_positives": fp,
                "first_flag_Q": round(float(S.flowrate.iat[f0]), 2), "first_flag_dp": round(float(dp[f0]), 3)}

    tp = lambda f: {"precision": round(float((f & y[te]).sum() / max(f.sum(), 1)), 3),
                    "recall": round(float((f & y[te]).sum() / max(y[te].sum(), 1)), 3),
                    "f1": round(float(2 * (f & y[te]).sum() / max(f.sum() + y[te].sum(), 1)), 3)}
    hi = S.flowrate.to_numpy()[te] > QMAX
    res = {
        "variant": name, "rows": int(len(S)),
        "segment_rows": {k: int(len(v)) for k, v in seg.items()},
        "onset_row_wall": [first, str(S.wall.iat[first])],
        "test_rows_pos_neg": [int(y[te].sum()), int((1 - y[te]).sum())],
        "test_rows_Q_over_10": int(hi.sum()),
        "train_val_dp_band": [round(float(dp[np.concatenate([tr, va])][y[np.concatenate([tr, va])] == 0].max()), 3),
                              round(float(dp[np.concatenate([tr, va])][y[np.concatenate([tr, va])] == 1].min()), 3)],
        "xgb": {"roc_auc": round(mx["roc_auc"], 3), "pr_auc": round(mx["pr_auc"], 3), **tp(fx), **ev(fx)},
        "rf": {"roc_auc": round(mr["roc_auc"], 3), **tp(fr), **ev(fr)},
        "raw_dp_roc_auc": round(float(roc_auc_score(y[te], dp[te])), 3),
        "midgap_threshold_kPa": round(float(midgap), 3), "midgap": {**tp(fm), **ev(fm)},
        "envelope_dp_kPa": round(float(env), 3), "envelope": {**tp(fe), **ev(fe)},
    }
    print(json.dumps(res), flush=True)
    return res


def main() -> None:
    S, meta = build_series()
    res = {"series": meta["info"], "level_kPa": LEVEL, "q_max_m3h": QMAX}
    res["ALL"] = run(S, meta["seg_t"], "ALL")
    res["MEAS"] = run(S[S.flowrate <= QMAX], meta["seg_t"], "MEAS")
    (OUT / "a12_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
