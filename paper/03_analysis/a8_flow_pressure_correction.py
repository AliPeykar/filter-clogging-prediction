"""Paper analysis A8: effect of the fixed density in the Arduino flow conversion.

The Arduino computed Q from the orifice dP with a density fixed by the
operator-entered gauge pressure (130/133/13/150 kPa + 101.3 kPa, 14.85 C;
Note S1). The rotameter discharges to the room, so the actual pressure at
the orifice is about atmospheric plus the losses downstream of the orifice,
which include the filter dP that rises from 0 to 62 kPa during the run.

This script
1. inverts the Arduino conversion to recover the orifice dP from logged Q;
2. estimates the upstream (orifice) pressure as
   P_up = P_atm + permanent orifice loss (ISO 5167-2) + filter dP,
   i.e. a LOWER bound (losses after the filter section, rotameter and
   fittings, were not measured and are set to zero);
3. recomputes the flow as volumetric flow at outlet conditions
   (P_atm, 14.85 C), Q_out = m / rho_atm, proportional to mass flow;
4. repeats the Q-based analyses of the paper with Q_out, next to the same
   analyses with the logged Q (which must reproduce the paper's numbers);
5. writes the data set with Q_out in place of Q for a retraining run.

P_atm is not recorded; 101.3 kPa is used (the value the Arduino adds). A
sensitivity with lower P_atm and with 5 kPa of unmeasured downstream loss
is reported. Temperature is fixed at the Arduino's 14.85 C and cancels in
all ratios.

Run from paper/:  python 03_analysis/a8_flow_pressure_correction.py
Outputs: 03_analysis/out/a8_results.json,
         03_analysis/out/data_flow_corrected.xlsx
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
OUT = HERE / "out"
CANDIDATES = [PAPER.parent / "Comprehensive_Filter_Analysis.xlsx",
              PAPER / "07_repository" / "Comprehensive_Filter_Analysis.xlsx",
              PAPER.parent / "Comprehensive_Filter_Analysis.xlsx"]
DATA = next(p for p in CANDIDATES if p.exists())

# ---------------------------------------------------------------- constants
R, T = 287.05, 273.15 + 14.85
d, D = 6.9e-3, 53.2e-3
beta = d / D
C = 0.60
A_d = np.pi / 4 * d ** 2
K = C / np.sqrt(1 - beta ** 4) * A_d          # Q[m3/s] = K sqrt(2 dP / rho)
P_ATM = 101.3                                  # kPa, as on the Arduino

# permanent pressure loss fraction of an orifice plate (ISO 5167-2, 5.4.4)
_s = np.sqrt(1 - beta ** 4 * (1 - C ** 2))
PERM = (_s - C * beta ** 2) / (_s + C * beta ** 2)

# split segments (1-based inclusive, Table 1)
SEG = {"Train-A": (1, 7140), "Val-A": (7201, 7800), "Test": (7861, 9600),
       "Val-B": (9661, 10260), "Train-B": (10321, 11519)}
STEP, ONSET = 8340, 8943
TH_XGB = 0.9333256940020761                    # main run, metrics.json


def entered_abs_pressure(n: int) -> np.ndarray:
    """Entered absolute pressure (kPa) per stored sample (Note S1)."""
    p = np.full(n, 231.3)
    idx = np.arange(1, n + 1)
    p[(idx >= 2042) & (idx <= 2321)] = 234.3
    p[(idx >= 5639) & (idx <= 5646)] = 114.3
    p[idx >= 9681] = 251.3
    return p


def corrected_flow(q_log, dp_filter, p_ass, p_atm=P_ATM, extra=0.0):
    """Outlet volumetric flow (m3/h at p_atm, T) and upstream pressure (kPa)."""
    rho_ass = p_ass * 1e3 / (R * T)
    dp_orif = rho_ass / 2 * (q_log / 3600 / K) ** 2 / 1e3          # kPa
    p_up = p_atm + PERM * dp_orif + dp_filter + extra
    # m = K sqrt(2 rho_up dP);  Q_out = m / rho_atm
    q_out = q_log * np.sqrt(p_ass * p_up) / p_atm
    return q_out, p_up, dp_orif


def seg(a, name):
    lo, hi = SEG[name]
    return a[lo - 1:hi]


def q_analyses(q, dp, y):
    """The Q-based quantities of the paper for one flow series."""
    r = {}
    for k in SEG:
        s = seg(q, k)
        r[f"{k}_min_median"] = [round(float(s.min()), 3), round(float(np.median(s)), 3)]
    r["median_drop_TrainA_to_TrainB_pct"] = round(
        100 * (1 - np.median(seg(q, "Train-B")) / np.median(seg(q, "Train-A"))), 1)
    healthy = np.r_[seg(q, "Train-A"), seg(q, "Val-A")]
    qmin = float(healthy.min())
    r["healthy_train_val_min"] = round(qmin, 3)
    t = seg(q, "Test"); tdp = seg(dp, "Test"); ty = seg(y, "Test")
    rows = np.arange(SEG["Test"][0], SEG["Test"][1] + 1)
    pre, mid, post = rows < STEP, (rows >= STEP) & (rows < ONSET), rows >= ONSET
    r["test_below_healthy_min_pct_pre_step_mid_post"] = [
        round(100 * float((t[m] < qmin).mean()), 1) for m in (pre, mid, post)]
    r["test_below_healthy_min_n"] = int((t < qmin).sum())
    band = (tdp > 18.038) & (tdp < 29.008)
    r["band_n_and_also_below_healthy_min"] = [int(band.sum()), int((band & (t < qmin)).sum())]
    r["roc_auc_minus_q"] = round(float(roc_auc_score(ty, -t)), 3)
    r["roc_auc_dp_over_q"] = round(float(roc_auc_score(ty, tdp / np.maximum(t, 0.5))), 3)
    # Q threshold sweep (flag if Q <= thr), 0.05 grid over the test range
    grid = np.round(np.arange(np.floor(t.min() * 20) / 20 - 0.2, np.ceil(np.median(t) * 20) / 20 + 0.6, 0.05), 2)
    best = (0.0, None)
    for th in grid:
        f = t <= th; tp = (f & (ty == 1)).sum(); fp = (f & (ty == 0)).sum(); fn = (~f & (ty == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (f1, th)
    r["q_sweep_max_f1_at"] = [round(float(best[0]), 3), float(best[1])]
    i = STEP - 1
    r["step_8340_q_change_pct"] = round(100 * (q[i] / q[i - 1] - 1), 2)
    r["step_8340_dp_over_q_change_pct"] = round(100 * ((dp[i] / q[i]) / (dp[i - 1] / q[i - 1]) - 1), 1)
    r["n_below_6_m3h_and_pct"] = [int((q < 6.0).sum()), round(100 * float((q < 6.0).mean()), 1)]
    r["n_above_15_m3h"] = int((q > 15.0).sum())
    rr = dp / np.maximum(q, 0.5)
    r["median_dp_over_q_1_7140_and_10321_11519"] = [round(float(np.median(rr[:7140])), 3),
                                                     round(float(np.median(rr[10320:])), 3)]
    return r, qmin


def main() -> None:
    df = pd.read_excel(DATA, sheet_name="Comprehensive_Analysis")
    q_log = df.flowrate.to_numpy(float); dp = df.dp.to_numpy(float)
    y = (df.filter_status.astype(str).str.strip() == "Clogged").astype(int).to_numpy()
    assert y.sum() == 2577, y.sum()
    p_ass = entered_abs_pressure(len(df))
    q_out, p_up, dp_orif = corrected_flow(q_log, dp, p_ass)

    res = {"data": str(DATA.name), "permanent_loss_fraction": round(float(PERM), 4),
           "assumptions": "P_up = 101.3 kPa + permanent orifice loss + filter dP (losses after the "
                          "filter section set to zero: lower bound); isothermal 14.85 C; "
                          "Q_out = flow at 101.3 kPa, proportional to mass flow"}
    res["orifice_dp_kPa_median_TrainA_TrainB"] = [round(float(np.median(seg(dp_orif, "Train-A"))), 2),
                                                  round(float(np.median(seg(dp_orif, "Train-B"))), 2)]
    res["p_up_kPa_abs_median_TrainA_Test_TrainB"] = [round(float(np.median(seg(p_up, k))), 1)
                                                     for k in ("Train-A", "Test", "Train-B")]
    res["entered_over_actual_density_median_TrainA_TrainB"] = [
        round(float(np.median(seg(p_ass / p_up, k))), 3) for k in ("Train-A", "Train-B")]
    res["q_out_over_q_log_median_TrainA_Test_TrainB"] = [
        round(float(np.median(seg(q_out / q_log, k))), 3) for k in ("Train-A", "Test", "Train-B")]
    # logged Q change at constant mass flow implied by density alone, Train-A -> Train-B
    a, b = np.median(seg(p_ass * p_up, "Train-A")), np.median(seg(p_ass * p_up, "Train-B"))
    pa, pb = np.median(seg(p_ass, "Train-A")), np.median(seg(p_ass, "Train-B"))
    pua, pub = np.median(seg(p_up, "Train-A")), np.median(seg(p_up, "Train-B"))
    res["logged_q_change_at_constant_mass_flow_pct"] = round(100 * (np.sqrt((pa * pua) / (pb * pub)) - 1), 1)
    # pipe Reynolds number at the median flow: as in the paper (logged Q,
    # rho 1.204, Table S1) and from the recomputed mass flow
    mu = 1.81e-5
    re = lambda mdot: 4 * mdot / (np.pi * D * mu)
    rho_atm = P_ATM * 1e3 / (R * T)
    res["re_D_median_logged_rho1204_and_recomputed"] = [
        int(round(re(1.204 * np.median(q_log) / 3600), -1)),
        int(round(re(rho_atm * np.median(q_out) / 3600), -1))]

    res["logged"], qmin_log = q_analyses(q_log, dp, y)
    res["corrected"], qmin_cor = q_analyses(q_out, dp, y)

    # agreement of XGBoost flags with a Q threshold (test window)
    tw = pd.read_csv(OUT / "test_window_predictions.csv")
    flags = (tw.p_xgb >= TH_XGB).to_numpy()
    tq_cor = q_out[tw.row.to_numpy() - 1]
    agree = {}
    for name, tq in (("logged", tw.flowrate.to_numpy()), ("corrected", tq_cor)):
        best = max(((float(((tq <= th) == flags).mean()), float(th))
                    for th in np.round(np.arange(tq.min(), np.median(tq) + 1, 0.01), 2)), key=lambda z: z[0])
        agree[name] = [round(100 * best[0], 1), best[1]]
    res["xgb_flag_agreement_with_q_threshold_pct_at"] = agree

    # sensitivity: lower atmospheric pressure, 5 kPa unmeasured downstream loss
    sens = {}
    for label, kw in (("p_atm_90", {"p_atm": 90.0}), ("extra_loss_5kPa", {"extra": 5.0})):
        qs, _, _ = corrected_flow(q_log, dp, p_ass, **kw)
        r, _ = q_analyses(qs, dp, y)
        sens[label] = {k: r[k] for k in ("median_drop_TrainA_to_TrainB_pct",
                                         "test_below_healthy_min_pct_pre_step_mid_post",
                                         "roc_auc_minus_q", "n_below_6_m3h_and_pct")}
    res["sensitivity"] = sens

    out = df.copy(); out["flowrate"] = q_out
    with pd.ExcelWriter(OUT / "data_flow_corrected.xlsx") as w:
        out.to_excel(w, sheet_name="Comprehensive_Analysis", index=False)
    (OUT / "a8_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
