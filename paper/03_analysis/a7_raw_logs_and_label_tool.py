"""A7 - Raw logs, sample provenance and the labeling tool (added 2026-10-19).

Inputs (read only, outside paper/):
  <THESIS>/newboost/final2/Comprehensive_Filter_Analysis.xlsx   analysed data set
  <THESIS>/newboost/filter-analyzer-complete.html                labeling tool source
  <THESIS>/data/data2.csv, data2_2.csv ... data2_9.csv           PuTTY logs of the Arduino
  <THESIS>/data/newdata/data.csv                                 concatenated logs
  paper/03_analysis/out/test_window_predictions.csv              main-run test scores
In this repository the data set is at the root, the logs are in raw_logs/
(detected automatically) and the labeling tool is in labeling_tool/.
Elsewhere <THESIS> defaults to the folder "Thesis" next to paper/; override
with the environment variable THESIS_DIR.

Outputs: paper/03_analysis/out/a7_results.json, a7_sample_provenance.csv

Checks:
 1. Python port of the labeling tool (JavaScript in the HTML file) reproduces
    the stored short_ema, long_ema, adaptive_cusum, cusum_threshold and
    validation_score for all samples; finds the settings used.
 2. Decomposes the score into its seven criteria; tests the equivalent rule
    "sample > 7,500 and dP > baseline + 5 sd".
 3. Maps every stored sample to its raw log row: file, Arduino time, wall-clock
    time, entered pressure P1; finds splices, gaps, duplicated rows and
    samples that match no log row.
 4. Re-computes the logged Q from the orifice equation and the dP channels
    from the MPX5100 transfer function.
 5. Re-scores the main-run test predictions without the duplicated block.
"""
from __future__ import annotations
import datetime as dt, json, os, re
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

PAPER = Path(__file__).resolve().parents[1]
THESIS = Path(os.environ.get("THESIS_DIR", PAPER.parent / "Thesis"))
OUT = PAPER / "03_analysis" / "out"
XLSX = THESIS / "newboost" / "final2" / "Comprehensive_Filter_Analysis.xlsx"
LOGS = THESIS / "data"
if (PAPER.parent / "raw_logs").is_dir():  # public repository layout
    XLSX = PAPER.parent / "Comprehensive_Filter_Analysis.xlsx"
    LOGS = PAPER.parent / "raw_logs"
FILES = ["data2.csv"] + [f"data2_{i}.csv" for i in range(2, 10)]

D = pd.read_excel(XLSX)
n = len(D)
dp = D.dp.to_numpy(float)
tcol = D.time.to_numpy(float)
res: dict = {"inputs": {"dataset": str(XLSX.name), "n_samples": n}}


# ---------------------------------------------------------------- 1. tool port
def label_tool(sens="moderate", thr=12, time_thr=7500, ema_s=50, ema_l=200, base_frac=0.2):
    """Line-by-line port of analyzeData(), computeAdaptiveCUSUM() and
    performComprehensiveValidation() in filter-analyzer-complete.html."""
    bs = int(np.floor(n * base_frac)); b = dp[:bs]; sb = np.sort(b)
    q1 = sb[int(np.floor(len(sb) * .25))]; q3 = sb[int(np.floor(len(sb) * .75))]; iqr = q3 - q1
    cb = b[(b >= q1 - 1.5 * iqr) & (b <= q3 + 1.5 * iqr)]
    bm = cb.mean(); bsd = np.sqrt(((cb - bm) ** 2).mean())
    sa, la = 2 / (ema_s + 1), 2 / (ema_l + 1)
    S = np.empty(n); L = np.empty(n); S[0] = L[0] = dp[0]
    for i in range(1, n):
        S[i] = sa * dp[i] + (1 - sa) * S[i - 1]; L[i] = la * dp[i] + (1 - la) * L[i - 1]
    km, hm = {"conservative": (0.8, 5.0), "moderate": (0.5, 4.0), "aggressive": (0.3, 3.0)}[sens]
    lw = max(50, min(200, int(np.floor(n * 0.05))))
    C = np.zeros(n); H = np.zeros(n); trig = np.zeros(n, bool)
    for i in range(n):
        ld = dp[max(0, i - lw):i + 1]
        if len(ld) > 10:
            sd = np.sqrt(((ld - ld.mean()) ** 2).mean()); k = max(km * sd, 0.05); h = max(hm * sd, 2 * bsd, 0.5)
        else:
            k = km * bsd; h = hm * bsd
        H[i] = h
        cd = 0.7 * (dp[i] - bm) + 0.3 * (S[i] - L[i])
        C[i] = max(0.0, (C[i - 1] if i else 0.0) + cd - k); trig[i] = C[i] > h
    P = np.zeros((n, 7), int)
    for i in range(n):
        P[i, 0] = 4 if (trig[i] and C[i] > H[i] * 2.0) else 2 if (trig[i] and C[i] > H[i] * 1.2) else 0
        d = S[i] - L[i]
        P[i, 1] = 4 if (S[i] > L[i] and d > 4 * bsd) else 2 if (S[i] > L[i] and d > 2 * bsd) else 0
        P[i, 2] = 3 if dp[i] > bm + 5 * bsd else 2 if dp[i] > bm + 3 * bsd else 0
        P[i, 3] = 2 if tcol[i] > time_thr else 1 if tcol[i] > time_thr - 500 else 0
        if i >= 30:
            o = dp[i - 30:i - 15].sum() / 15; r = dp[i - 15:i + 1].sum() / 16
            if o > 0:
                P[i, 4] = 3 if r / o > 1.4 else 2 if r / o > 1.2 else 0
        if i >= 20:
            P[i, 5] = 2 if (dp[i - 20:i + 1] > bm + 2 * bsd).sum() >= 15 else 0
        if i < n - 30:  # uses the next 30 samples
            fa = dp[i:i + 30].mean(); ca = dp[max(0, i - 10):i + 1].sum() / min(11, i + 1)
            P[i, 6] = 2 if fa > ca * 1.15 else 0
    return dict(bm=bm, bsd=bsd, S=S, L=L, C=C, H=H, P=P, score=P.sum(1))


CRIT = ["cusum", "ema_divergence", "dp_elevation", "time_index", "rate_increase",
        "sustained_elevation", "look_ahead_30"]
st_score = D.validation_score.to_numpy()
search = []
for sens in ("conservative", "moderate", "aggressive"):
    for bf in (0.2, 0.25):
        r = label_tool(sens, base_frac=bf)
        search.append(dict(sensitivity=sens, baseline_fraction=bf,
                           score_identical=float((r["score"] == st_score).mean()),
                           cusum_max_abs_diff=float(np.abs(r["C"] - D.adaptive_cusum).max())))
        if sens == "moderate" and bf == 0.25:
            R = r
res["tool_settings_search"] = search
lab = (D.filter_status == "Clogged").to_numpy()
res["tool_reproduction"] = {
    "settings": {"cusum_sensitivity": "moderate", "baseline_fraction": 0.25, "time_threshold": 7500,
                 "ema_short": 50, "ema_long": 200, "score_threshold": 11},
    "short_ema_max_abs_diff": float(np.abs(R["S"] - D.short_ema).max()),
    "long_ema_max_abs_diff": float(np.abs(R["L"] - D.long_ema).max()),
    "cusum_max_abs_diff": float(np.abs(R["C"] - D.adaptive_cusum).max()),
    "cusum_threshold_max_abs_diff": float(np.abs(R["H"] - D.cusum_threshold).max()),
    "scores_identical": int((R["score"] == st_score).sum()),
    "labels_identical_at_11": bool(((R["score"] >= 11) == lab).all()),
    "clogged_at_threshold": {str(t): int((R["score"] >= t).sum()) for t in (10, 11, 12)},
    "baseline_mean_kPa": round(float(R["bm"]), 4), "baseline_sd_kPa": round(float(R["bsd"]), 4),
}

# ------------------------------------------------------- 2. label decomposition
P, sc = R["P"], R["score"]
T5 = R["bm"] + 5 * R["bsd"]
simple = (tcol > 7500) & (dp > T5)
onset = int(np.argmax(sc >= 11)) + 1
def first(mask): return int(np.argmax(mask)) + 1 if mask.any() else None
res["label_structure"] = {
    "dp_thresholds_kPa": {"mean+2sd": round(R["bm"] + 2 * R["bsd"], 3), "mean+3sd": round(R["bm"] + 3 * R["bsd"], 3),
                          "mean+5sd": round(T5, 3)},
    "rule_onset_sample": onset,
    "scores_7501_to_onset_minus_1": sorted(set(int(x) for x in sc[7500:onset - 1])),
    "criteria_at_samples": {str(s): dict(zip(CRIT, map(int, P[s - 1]))) for s in (7000, 7001, 7501, 8340, 8942, 8943, 8954, 9500)},
    "equivalent_rule": "clogged <=> sample > 7,500 and dP > mean + 5 sd",
    "equivalent_rule_disagreements": int((simple != lab).sum()),
    "disagreeing_samples": [int(x) for x in np.where(simple != lab)[0] + 1],
    "onset_without_look_ahead": first((sc - P[:, 6]) >= 11),
    "clogged_without_time_criterion": int(((sc - P[:, 3]) >= 11).sum()),
    "clogged_labels_that_need_time_criterion": int(((sc >= 11) & ((sc - P[:, 3]) < 11)).sum()),
    "clogged_labels_that_need_look_ahead": int(((sc >= 11) & ((sc - P[:, 6]) < 11)).sum()),
    "max_score_before_7001": int(sc[:7000].max()),
}


# ----------------------------------------------------------- 3. provenance
def load_log(path: Path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    start = dt.datetime.strptime(re.search(r"PuTTY log (\d{4}\.\d\d\.\d\d \d\d:\d\d:\d\d)", txt).group(1), "%Y.%m.%d %H:%M:%S")
    meta = defaultdict(list); rows = []
    for line in txt.splitlines():
        m = re.match(r"# (.+?): (.+)", line)
        if m: meta[m.group(1)].append(m.group(2).strip())
        p = line.split(",")
        if len(p) == 10:
            try: rows.append([float(x) for x in p])
            except ValueError: pass
    df = pd.DataFrame(rows, columns=["t", "adc1", "v1", "dp1", "Q", "mdot", "P1", "adc2", "v2", "dp2"])
    r = np.where(np.diff(df.t.to_numpy()) < 0)[0]          # rows left over from the previous session
    if len(r): df = df.iloc[r[-1] + 1:]
    df = df.copy(); df["file"] = path.name
    df["wall"] = [start + dt.timedelta(seconds=float(x)) for x in df.t]
    return start, dict(meta), df

logs, F = {}, []
for f in FILES:
    s, meta, df = load_log(LOGS / f); F.append(df)
    logs[f] = {"putty_start": str(s), "rows": len(df), "header": {k: v[-1] for k, v in meta.items()}}
F = pd.concat(F, ignore_index=True)
res["logs"] = logs

Rw = pd.read_csv(LOGS / "newdata" / "data.csv", header=None,
                 names=["t", "adc1", "v1", "dp1", "Q", "mdot", "P1", "adc2", "v2", "dp2"])
rd = Rw.dp2.round(3).to_numpy(); sd_ = D.dp.round(3).to_numpy()
K = 12; seeds = defaultdict(list)
for k in range(len(rd) - K): seeds[rd[k:k + K].tobytes()].append(k)
ridx = np.full(n, -1); i = 0
while i < n:                                       # greedy longest-match blocks on the dP sequence
    best = (0, None)
    for k in (seeds.get(sd_[i:i + K].tobytes(), []) if i + K <= n else []):
        a, b, path = i, k, []
        while a < n and b < len(rd):
            if rd[b] == sd_[a]: path.append(b); a += 1; b += 1; continue
            for s in (1, 2, 3):
                if b + s + 1 < len(rd) and a + 1 < n and rd[b + s] == sd_[a] and rd[b + s + 1] == sd_[a + 1]:
                    b += s; break
            else: break
        if len(path) > best[0]: best = (len(path), path)
    if best[1] is None: i += 1; continue
    ridx[i:i + best[0]] = best[1]; i += best[0]
key = lambda df: list(zip(df.t.round(0), df.adc1, df.adc2, df.P1.round(1)))
fk = {}
for j, kk in enumerate(key(F)): fk.setdefault(kk, j)
rk = key(Rw)
fidx = np.array([fk.get(rk[r], -1) if r >= 0 else -1 for r in ridx])
ok = fidx >= 0
prov = pd.DataFrame({"sample": np.arange(1, n + 1), "file": np.where(ok, F.file.to_numpy()[np.clip(fidx, 0, None)], ""),
                     "arduino_t_s": np.where(ok, F.t.to_numpy()[np.clip(fidx, 0, None)], np.nan),
                     "P1_kPa_abs": np.where(ok, F.P1.to_numpy()[np.clip(fidx, 0, None)], np.nan),
                     "dp_stored": D.dp.to_numpy(), "Q_stored": D.flowrate.to_numpy()})
prov["wall_clock"] = pd.Series(pd.to_datetime(np.where(ok, F["wall"].to_numpy()[np.clip(fidx, 0, None)], pd.NaT)))
prov["log_row"] = fidx
prov.to_csv(OUT / "a7_sample_provenance.csv", index=False)

m = prov[ok]
step = m.wall_clock.diff().dt.total_seconds()
within = step[(step > 0) & (step <= 5)]
gaps = m[(step > 15) | (step < 0)]
dup = m.duplicated("log_row", keep=False)
ds = m.loc[dup, "sample"].to_numpy(); br = np.where(np.diff(ds) != 1)[0]
runs = [(int(a), int(b)) for a, b in zip(np.r_[ds[0], ds[br + 1]], np.r_[ds[br], ds[-1]])]
seg = m.groupby((m.file != m.file.shift()).cumsum()).agg(first=("sample", "min"), last=("sample", "max"), file=("file", "first"),
                                                       start=("wall_clock", "min"), end=("wall_clock", "max"))
p1 = m.groupby((m.P1_kPa_abs != m.P1_kPa_abs.shift()).cumsum()).agg(first=("sample", "min"), last=("sample", "max"), P1=("P1_kPa_abs", "first"))
ev = {}
for s in (8191, 8340, 8954, 9681):
    a, b = m[m["sample"] == s - 1], m[m["sample"] == s]
    if len(a) and len(b):
        ev[str(s)] = {"prev": [a.file.iat[0], float(a.arduino_t_s.iat[0]), str(a.wall_clock.iat[0])],
                      "this": [b.file.iat[0], float(b.arduino_t_s.iat[0]), str(b.wall_clock.iat[0])],
                      "wall_clock_jump_s": float((b.wall_clock.iat[0] - a.wall_clock.iat[0]).total_seconds()),
                      "dp_prev_this": [float(a.dp_stored.iat[0]), float(b.dp_stored.iat[0])],
                      "Q_prev_this": [float(a.Q_stored.iat[0]), float(b.Q_stored.iat[0])],
                      "P1_prev_this": [float(a.P1_kPa_abs.iat[0]), float(b.P1_kPa_abs.iat[0])]}
res["provenance"] = {
    "samples_traced_to_a_log_row": int(ok.sum()),
    "samples_matching_no_log_row": [int(x) for x in prov.loc[~ok, "sample"]],
    "logging_interval_within_files_s": {"median": float(within.median()), "mean": round(float(within.mean()), 3),
                                        "share_1s": round(float((within == 1).mean()), 3), "share_2s": round(float((within == 2).mean()), 3)},
    "wall_clock_first_last": [str(m.wall_clock.min()), str(m.wall_clock.max())],
    "wall_clock_span_h": round((m.wall_clock.max() - m.wall_clock.min()).total_seconds() / 3600, 3),
    "gaps_over_15s": int((step > 15).sum()), "gaps_over_15s_total_s": float(step[step > 15].sum()),
    "backward_jumps": int((step < 0).sum()),
    "files_in_order": [dict(file=r.file, first=int(r.first), last=int(r.last), start=str(r.start), end=str(r.end)) for r in seg.itertuples()],
    "entered_P1_by_samples": [dict(first=int(r.first), last=int(r.last), P1_kPa_abs=float(r.P1)) for r in p1.itertuples()],
    "duplicated_log_rows_sample_runs": runs,
    "events": ev,
}

# ------------------------------------------- 3b. what the assembly removed
used = set(zip(prov.file, prov.arduino_t_s))
per_file = []
for f in FILES[:7]:
    _, _, dfl = load_log(LOGS / f)
    u = prov[prov.file == f]
    span = dfl[(dfl.t >= u.arduino_t_s.min()) & (dfl.t <= u.arduino_t_s.max())]
    rem = span[[(f, t) not in used for t in span.t]]
    g = np.diff(dfl.t.to_numpy())
    head, tail = dfl[dfl.t < u.arduino_t_s.min()], dfl[dfl.t > u.arduino_t_s.max()]
    rng = lambda x: [float(x.dp2.min()), float(x.dp2.max())] if len(x) else None
    per_file.append(dict(file=f, head_rows_not_used=len(head), head_Q_over_10=int((head.Q > 10).sum()), head_dp_range=rng(head),
                         first_head_t_dp_over_21_776=(float(head[head.dp2 > 21.776].t.min()) if (head.dp2 > 21.776).any() else None),
                         tail_rows_not_used=len(tail), tail_Q_over_10=int((tail.Q > 10).sum()), tail_dp_range=rng(tail),
                         log_t_range=[float(dfl.t.min()), float(dfl.t.max())],
                         log_rows_in_used_span=len(span), removed_rows=len(rem),
                         removed_rows_Q_over_10=int((rem.Q > 10).sum()),
                         logger_pauses_over_15s=int((g > 15).sum()), logger_pause_total_s=float(g[g > 15].sum())))
_, _, d3 = load_log(LOGS / "data2_3.csv")
ep = d3[(d3.t > 4226) & (d3.t < 4888)]
res["assembly"] = {
    "per_file": per_file,
    "removed_rows_total": int(sum(p["removed_rows"] for p in per_file)),
    "removed_rows_Q_over_10_total": int(sum(p["removed_rows_Q_over_10"] for p in per_file)),
    "head_tail_rows_not_used_total": int(sum(p["head_rows_not_used"] + p["tail_rows_not_used"] for p in per_file)),
    "logger_prompt_note": "the logger stops for an operator prompt to re-enter the gauge pressure about every 280 rows",
    "episode_removed_before_sample_8340": {
        "arduino_t_s": [4227, 4887], "rows": len(ep), "rows_Q_over_10": int((ep.Q > 10).sum()),
        "Q_median": float(ep[ep.Q > 10].Q.median()), "orifice_adc_range_when_Q_over_10": [int(ep[ep.Q > 10].adc1.min()), int(ep[ep.Q > 10].adc1.max())],
        "filter_dp_first_last": [float(ep.dp2.iloc[0]), float(ep.dp2.iloc[-1])],
        "last_sample_before_in_real_time": {"sample": 8190, "arduino_t_s": 4226, "dp": float(D.dp.iat[8189])},
        "inserted_instead": "samples 8,191-8,339 = repeat of samples 6,793-6,941 (arduino t 2286-2444)"},
    "orifice_adc_when_any_Q_over_20": [int(F[F.Q > 20].adc1.min()), int(F[F.Q > 20].adc1.max())],
}

# ------------------------------------------- 4. Q and dP conversion on the Arduino
Fv = F[F.dp1 > 0]
Rg, T = 287.05, 273.15 + 14.85; d_o, D_p = 6.9e-3, 53.2e-3; beta = d_o / D_p
rho = Fv.P1 * 1e3 / (Rg * T)
Qc = 0.60 / np.sqrt(1 - beta ** 4) * (np.pi / 4 * d_o ** 2) * np.sqrt(2 * Fv.dp1 * 1e3 / rho) * 3600
c1 = np.polyfit(Fv.v1, Fv.dp1, 1); F2 = F[F.dp2 > 0]; c2 = np.polyfit(F2.v2, F2.dp2, 1)
res["arduino_conversion"] = {
    "Q_logged_over_orifice_equation": {"median": round(float((Fv.Q / Qc).median()), 5),
                                       "p01_p99": [round(float(x), 5) for x in np.percentile(Fv.Q / Qc, [1, 99])],
                                       "assumptions": "C=0.60, eps=1, d=6.9 mm, D=53.2 mm, T=14.85 C, rho=P1/(R T), P1=entered gauge + 101.3 kPa"},
    "volts_per_count": round(float((F.v1 / F.adc1).median()), 7),
    "dp1_vs_volts": [round(float(c1[0]), 4), round(float(c1[1]), 4)],
    "dp2_vs_volts": [round(float(c2[0]), 4), round(float(c2[1]), 4)],
    "MPX5100_nominal": "V = 5.0 (0.009 P + 0.04)  =>  P = 22.222 V - 4.444 kPa",
    "kPa_per_count": round(5 / 1023 / 5 / 0.009, 4),
    "dp2_zero_clip_max_volts": float(F[F.dp2 == 0].v2.max()),
}

# ------------------------------------------- 5. main-run test scores without the duplicated block
tw = pd.read_csv(OUT / "test_window_predictions.csv")
th_x, th_r = 0.9333256940020761, 0.962581617734156     # main run, metrics.json
def score(d):
    return {"n": len(d), "auc_xgb": round(roc_auc_score(d.y, d.p_xgb), 3), "auc_rf": round(roc_auc_score(d.y, d.p_rf), 3),
            "auc_raw_dp": round(roc_auc_score(d.y, d.dp), 3), "f1_xgb": round(f1_score(d.y, d.p_xgb >= th_x), 3),
            "f1_rf": round(f1_score(d.y, d.p_rf >= th_r), 3)}
dmask = tw.row.between(8191, 8339)
res["duplicate_block_effect"] = {"all_test": score(tw), "without_8191_8339": score(tw[~dmask]),
                                 "block_labels": sorted(map(int, tw[dmask].y.unique())),
                                 "block_flags_xgb": int((tw[dmask].p_xgb >= th_x).sum())}

(OUT / "a7_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps(res, indent=1, default=str)[:6000])
