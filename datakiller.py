"""
datakiller.py  -  Filter Clogging Detection & Lifecycle Prediction
==================================================================
Developed by : Ali Peykar  |  Hormozgan Gas Corporation
Version      : 2.0

Input
-----
  Excel file with 3 columns:
    time       [seconds]
    flowrate   [m3/h]
    dp         [psi]   (differential pressure across gas filter)

Pipeline
--------
  1.  Load & validate raw DP-meter data
  2.  Data-quality report (gaps, outliers, noise)
  3.  Darcy permeability tracking  K(t)  + exponential-decay model
  4.  XGBoost regression with rich time-series feature engineering
        - temporal 80/20 train/test split
        - hyperparameter search via RandomizedSearchCV + TimeSeriesSplit
  5.  SHAP feature-importance analysis
  6.  Clogging-time prediction  (physics model  +  ML model)
  7.  Remaining-Useful-Life (RUL) curve
  8.  Summary report saved to  <output_dir>/report.txt
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import shap

try:
    from statsmodels.graphics.tsaplots import plot_acf
    _HAS_SM = True
except ImportError:
    _HAS_SM = False

# ---- colour palette -------------------------------------------------------
_C = dict(
    flow  = "#1565C0",
    dp    = "#C62828",
    thr   = "#2E7D32",
    fit   = "#E65100",
    clog  = "#6A1B9A",
    perm  = "#00838F",
    test  = "#F57F17",
    shade = "#E3F2FD",
)

plt.rcParams.update({
    "figure.dpi"     : 130,
    "axes.grid"      : True,
    "grid.alpha"     : 0.25,
    "font.size"      : 10,
    "axes.titlesize" : 12,
    "axes.labelsize" : 10,
    "lines.linewidth": 1.6,
    "legend.fontsize": 9,
})
sns.set_context("notebook", font_scale=1.05)


class FilterAnalysis:
    """End-to-end analysis of gas-filter clogging from a DP-meter log."""

    VISCOSITY     = 1.81e-5
    DENSITY       = 1.225
    PSI_TO_PA     = 6894.76
    DP_THRESH_PSI = 15.0

    def __init__(self, file_path, filter_area=2.0, filter_length=0.02,
                 output_dir=None, show_plots=True):
        self.file_path     = file_path
        self.filter_area   = filter_area
        self.filter_length = filter_length
        self.show_plots    = show_plots
        self.output_dir    = output_dir or os.path.join(
            os.path.dirname(os.path.abspath(file_path)), "results")
        self.data        = None
        self.scaler      = StandardScaler()
        self.xgb_model   = None
        self.feat_cols   = []
        self.train_mask  = None
        self.test_mask   = None
        self.physics     = {}
        self._report     = []
        os.makedirs(self.output_dir, exist_ok=True)
        print("Results directory : " + self.output_dir)

    def _save(self, fig, name):
        path = os.path.join(self.output_dir, name + ".png")
        fig.savefig(path, bbox_inches="tight", dpi=130)
        print("  Saved -> " + os.path.basename(path))
        if self.show_plots:
            plt.show()
        plt.close(fig)

    def _log(self, line=""):
        print(line)
        self._report.append(line)

    def _find_col(self, *keywords):
        for col in self.data.columns:
            if any(kw in col.lower() for kw in keywords):
                return col
        return None

    # ------------------------------------------------------------------
    # 1.  LOAD & PREPROCESS
    # ------------------------------------------------------------------
    def load_data(self):
        try:
            raw = pd.read_excel(self.file_path)
        except Exception as exc:
            print("[ERROR] Cannot open file: " + str(exc))
            return False

        print("Raw data  :  " + str(raw.shape[0]) + " rows x " + str(raw.shape[1]) + " cols")
        print("Columns   :  " + str(raw.columns.tolist()))
        self.data = raw.copy()

        time_col = self._find_col("time", "sec", "min", "t(", "t_")
        flow_col = self._find_col("flow", "rate", "m3", "q(", "q_")
        dp_col   = self._find_col("dp", "pressure", "delta", "diff", "psi", "pa")

        if time_col and flow_col and dp_col:
            self.data = self.data[[time_col, flow_col, dp_col]].copy()
            self.data.columns = ["time", "flowrate", "dp"]
        else:
            print("  Warning: auto-detection failed; using first 3 columns as "
                  "[time (s), flowrate (m3/h), dp (psi)].")
            self.data = self.data.iloc[:, :3].copy()
            self.data.columns = ["time", "flowrate", "dp"]

        for col in ["time", "flowrate", "dp"]:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
        n0 = len(self.data)
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(subset=["time"], keep="first", inplace=True)
        self.data.sort_values("time", inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self._log("  Removed " + str(n0 - len(self.data)) + " invalid/duplicate rows  ->  " + str(len(self.data)) + " remain")

        zero_mask = self.data["dp"] < 0.01
        n_zero = zero_mask.sum()
        if n_zero:
            self.data = self.data[~zero_mask].copy()
            self.data.reset_index(drop=True, inplace=True)
            self._log("  Removed " + str(n_zero) + " zero-DP startup rows  ->  " + str(len(self.data)) + " remain")

        flow_z = ((self.data["flowrate"] - self.data["flowrate"].mean())
                  / self.data["flowrate"].std())
        self.data["flow_outlier"] = (flow_z.abs() > 3).astype(int)
        n_spk = self.data["flow_outlier"].sum()
        if n_spk:
            self._log("  Flagged " + str(n_spk) + " flow-rate spikes as outliers (|z| > 3)")

        dt = self.data["time"].diff().fillna(1.0).clip(lower=0.1)
        self.data["time_min"]  = self.data["time"] / 60.0
        self.data["velocity"]  = (self.data["flowrate"] / 3600.0) / self.filter_area
        self.data["dp_Pa"]     = self.data["dp"] * self.PSI_TO_PA
        self.data["dp_per_L"]  = self.data["dp_Pa"] / self.filter_length
        self.data["cum_vol"]   = (self.data["flowrate"] / 3600.0 * dt).cumsum()

        safe_dp = self.data["dp_Pa"].replace(0, np.nan)
        self.data["permeability"] = (
            self.VISCOSITY * self.data["velocity"] * self.filter_length / safe_dp)

        safe_q = self.data["flowrate"].replace(0, np.nan)
        self.data["resistance"] = self.data["dp"] / safe_q
        self.data["dp_rate"]    = (self.data["dp"].diff() / dt).fillna(0.0)
        self.data["dp_norm"]    = self.data["dp"] / self.data["dp"].max()
        self.data["clogged"]    = (self.data["dp"] >= self.DP_THRESH_PSI).astype(int)

        clog_rows = self.data[self.data["clogged"] == 1]
        self._log("\n" + "=" * 60)
        self._log("FILTER STATUS")
        self._log("=" * 60)
        if len(clog_rows):
            t_clog = clog_rows["time"].iloc[0]
            self._log("  WARNING  DP threshold (" + str(self.DP_THRESH_PSI) +
                      " psi) first exceeded at  t = " + str(int(t_clog)) + " s  (" +
                      str(round(t_clog/60, 1)) + " min)")
            self._log("  Current DP   : " + str(round(self.data["dp"].iloc[-1], 2)) + " psi")
            self._log("  Peak DP      : " + str(round(self.data["dp"].max(), 2)) + " psi")
        else:
            self._log("  OK  DP has not exceeded the " + str(self.DP_THRESH_PSI) + " psi threshold yet.")

        self._log("\nBasic statistics:")
        stats = self.data[["time", "flowrate", "dp", "permeability"]].describe()
        self._log(stats.round(5).to_string())
        return True


    # ------------------------------------------------------------------
    # 2.  RAW-DATA PLOT
    # ------------------------------------------------------------------
    def plot_raw_data(self):
        if self.data is None:
            print("Load data first.")
            return
        d = self.data
        t_clog_mask = d["clogged"] == 1
        t_clog = d.loc[t_clog_mask, "time"].iloc[0] if t_clog_mask.any() else None

        fig = plt.figure(figsize=(14, 11))
        gs  = gridspec.GridSpec(3, 1, hspace=0.40)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(d["time_min"], d["flowrate"], color=_C["flow"], lw=1.2, label="Flow rate")
        spk = d[d["flow_outlier"] == 1]
        if len(spk):
            ax1.scatter(spk["time_min"], spk["flowrate"], color="red",
                        s=15, zorder=5, label="Spikes (" + str(len(spk)) + " pts)")
        ax1.set_ylabel("Flow Rate  [m3/h]")
        ax1.set_title("Raw Acquisition Data  -  Gas Filter DP Meter")
        ax1.legend(loc="upper right")

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(d["time_min"], d["dp"], color=_C["dp"], lw=1.2, label="dP (measured)")
        ax2.axhline(self.DP_THRESH_PSI, color=_C["thr"], ls="--", lw=1.5,
                    label="Clogging threshold (" + str(self.DP_THRESH_PSI) + " psi)")
        if t_clog is not None:
            ax2.axvline(t_clog / 60, color=_C["clog"], ls=":", lw=1.8,
                        label="Threshold crossed  (t = " + str(round(t_clog/60, 1)) + " min)")
        ax2.set_ylabel("Pressure Drop  [psi]")
        ax2.legend(loc="upper left")

        ax3 = fig.add_subplot(gs[2])
        sc  = ax3.scatter(d["flowrate"], d["dp"], c=d["time_min"],
                          cmap="plasma", s=4, alpha=0.6, rasterized=True)
        cbar = fig.colorbar(sc, ax=ax3, pad=0.01)
        cbar.set_label("Time  [min]")
        ax3.axhline(self.DP_THRESH_PSI, color=_C["thr"], ls="--", lw=1.5)
        ax3.set_xlabel("Flow Rate  [m3/h]")
        ax3.set_ylabel("Pressure Drop  [psi]")
        ax3.set_title("dP vs Flow Rate  (colour = elapsed time)")

        fig.text(0.5, 0.01, "Time  [min]", ha="center", fontsize=10)
        self._save(fig, "01_raw_data")

    # ------------------------------------------------------------------
    # 3.  DATA QUALITY
    # ------------------------------------------------------------------
    def check_data_quality(self):
        if self.data is None:
            print("Load data first.")
            return
        d = self.data
        self._log("\n" + "=" * 60)
        self._log("DATA QUALITY REPORT")
        self._log("=" * 60)

        dt      = d["time"].diff().dropna()
        avg_dt  = dt.mean()
        max_dt  = dt.max()
        self._log("  Sampling interval  :  avg = " + str(round(avg_dt, 2)) +
                  " s  |  max = " + str(round(max_dt, 2)) + " s")
        n_gaps = (dt > 5 * avg_dt).sum()
        if n_gaps:
            self._log("  WARNING  " + str(n_gaps) + " gap(s) detected (> 5x average interval)")
        else:
            self._log("  OK  No significant gaps in time series")

        w = min(60, len(d) // 10)
        roll_flow_std = d["flowrate"].rolling(w, center=True).std()
        roll_dp_std   = d["dp"].rolling(w, center=True).std()
        self._log("  Flow noise (mean rolling sigma, w=" + str(w) + ")  :  " +
                  str(round(roll_flow_std.mean(), 4)) + " m3/h")
        self._log("  dP   noise (mean rolling sigma, w=" + str(w) + ")  :  " +
                  str(round(roll_dp_std.mean(), 4)) + " psi")

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle("Data Quality  -  Rolling Statistics", fontsize=13)

        for ax, col, noise, unit, color, lbl in [
            (axes[0], "flowrate", roll_flow_std, "m3/h", _C["flow"], "Flow Rate"),
            (axes[1], "dp",       roll_dp_std,   "psi",  _C["dp"],   "dP"),
        ]:
            rm = d[col].rolling(w, center=True).mean()
            ax.plot(d["time_min"], d[col], color=color, alpha=0.45, lw=0.8, label="Raw")
            ax.plot(d["time_min"], rm, color=color, lw=1.6,
                    label="Rolling mean  (w = " + str(w) + " s)")
            ax.fill_between(d["time_min"], rm - noise, rm + noise,
                            color=color, alpha=0.15, label="+/-1 sigma")
            ax.set_ylabel(lbl + "  [" + unit + "]")
            ax.legend(loc="upper left")

        axes[1].axhline(self.DP_THRESH_PSI, color=_C["thr"], ls="--", label="Threshold")
        axes[1].legend(loc="upper left")
        axes[1].set_xlabel("Time  [min]")
        plt.tight_layout()
        self._save(fig, "02_data_quality")



    # ------------------------------------------------------------------
    # 4.  DARCY PERMEABILITY TRACKING
    # ------------------------------------------------------------------
    def track_permeability(self):
        if self.data is None:
            print("Load data first.")
            return
        mask = ((self.data["flow_outlier"] == 0) &
                (self.data["permeability"].notna()) &
                (self.data["permeability"] > 0) &
                (self.data["dp"] > 0.1))
        d  = self.data[mask].copy()
        t  = d["time"].values
        K  = d["permeability"].values

        wl = min(201, len(K) // 10 * 2 + 1)
        wl = max(wl, 5)
        if wl % 2 == 0:
            wl += 1
        try:
            K_smooth = savgol_filter(K, window_length=wl, polyorder=3)
            K_smooth = np.clip(K_smooth, 1e-18, None)
        except Exception:
            K_smooth = K.copy()

        # Log-linear regression:  ln(K) = ln(K0) - t/tau
        valid  = (K_smooth > 0) & np.isfinite(K_smooth)
        t_v    = t[valid]
        K_v    = K_smooth[valid]
        log_K  = np.log(K_v)
        coef   = np.polyfit(t_v, log_K, 1)
        tau    = -1.0 / coef[0]
        K0     = np.exp(coef[1])
        log_K_pred = np.polyval(coef, t_v)
        r2_phys = np.corrcoef(log_K, log_K_pred)[0, 1] ** 2

        def exp_decay(t_, K0_, tau_):
            return K0_ * np.exp(-t_ / tau_)

        Q_nom   = self.data.loc[self.data["flow_outlier"] == 0, "flowrate"].mean()
        v_nom   = Q_nom / 3600 / self.filter_area
        K_crit  = (self.VISCOSITY * v_nom * self.filter_length
                   / (self.DP_THRESH_PSI * self.PSI_TO_PA))
        t_clog  = -tau * np.log(K_crit / K0) if K_crit < K0 else float("nan")

        self.physics = dict(K0=K0, tau=tau, K_crit=K_crit,
                            t_clog=t_clog, r2=r2_phys, Q_nom=Q_nom)

        self._log("\n" + "=" * 60)
        self._log("PERMEABILITY ANALYSIS  (Darcy model)")
        self._log("=" * 60)
        self._log("  Fit model    :  K(t) = K0 * exp(-t / tau)  [log-linear regression]")
        self._log("  K0           :  " + "{:.3e}".format(K0) + "  m2")
        self._log("  tau (decay)  :  " + str(round(tau, 1)) + " s  (" + str(round(tau/60, 1)) + " min)")
        self._log("  R2 (log K)   :  " + str(round(r2_phys, 4)))
        self._log("  K_critical   :  " + "{:.3e}".format(K_crit) + "  m2")
        if not np.isnan(t_clog):
            self._log("  Predicted clogging time (physics) :  " +
                      str(int(t_clog)) + " s  (" + str(round(t_clog/60, 1)) + " min)")
        else:
            self._log("  WARNING  Physics model predicts clogging has already occurred.")

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
        fig.suptitle("Darcy Permeability Tracking", fontsize=13)

        ax1 = axes[0]
        ax1.scatter(t / 60, K, s=3, color=_C["perm"], alpha=0.3,
                    label="K(t) - raw", rasterized=True)
        ax1.plot(t / 60, K_smooth, color=_C["perm"], lw=1.8, label="K(t) - smoothed")
        t_ext = np.linspace(t[0], max(t[-1], t_clog if not np.isnan(t_clog) else t[-1]) * 1.05, 400)
        ax1.plot(t_ext / 60, exp_decay(t_ext, K0, tau), "--",
                 color=_C["fit"], lw=1.8, label="Exp. decay fit")
        ax1.axhline(K_crit, color=_C["thr"], ls=":", lw=1.5,
                    label="K_critical (" + "{:.2e}".format(K_crit) + " m2)")
        if not np.isnan(t_clog):
            ax1.axvline(t_clog / 60, color=_C["clog"], ls="--", lw=1.5,
                        label="Predicted clogging  (" + str(round(t_clog/60, 1)) + " min)")
        ax1.set_yscale("log")
        ax1.set_ylabel("Permeability  K  [m2]")
        ax1.set_xlabel("Time  [min]")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.set_title("Permeability vs Time  (Darcy's Law)")

        ax2 = axes[1]
        ax2.plot(self.data["time_min"], self.data["dp"],
                 color=_C["dp"], lw=1.2, label="dP measured")
        ax2.axhline(self.DP_THRESH_PSI, color=_C["thr"], ls="--",
                    label="Threshold (" + str(self.DP_THRESH_PSI) + " psi)")
        clog_cross = self.data[self.data["clogged"] == 1]
        if len(clog_cross):
            tc = clog_cross["time_min"].iloc[0]
            ax2.axvline(tc, color=_C["clog"], ls=":", lw=1.8,
                        label="Actual crossing  (" + str(round(tc, 1)) + " min)")
        ax2.set_ylabel("Pressure Drop  [psi]")
        ax2.set_xlabel("Time  [min]")
        ax2.legend(loc="upper left")
        ax2.set_title("Measured dP vs Time")
        plt.tight_layout()
        self._save(fig, "03_permeability")



    # ------------------------------------------------------------------
    # 5.  FEATURE ENGINEERING
    # ------------------------------------------------------------------
    def prepare_xgboost_data(self):
        if self.data is None:
            print("Load data first.")
            return None, None

        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        d = self.data.copy()

        for lag in [60, 300, 600]:
            d["dp_lag_" + str(lag)] = d["dp"].shift(lag)

        w2 = 120
        d["dp_roll_mean"]   = d["dp"].rolling(w2, min_periods=10).mean()
        d["dp_roll_std"]    = d["dp"].rolling(w2, min_periods=10).std().fillna(0)
        d["flow_roll_mean"] = d["flowrate"].rolling(w2, min_periods=10).mean()
        d["flow_sq"]        = d["flowrate"] ** 2

        self.feat_cols = [
            "time", "flowrate", "flow_sq", "cum_vol",
            "dp_lag_60", "dp_lag_300", "dp_lag_600",
            "dp_roll_mean", "dp_roll_std", "flow_roll_mean",
            "dp_rate", "resistance",
        ]

        d_clean = d.dropna(subset=self.feat_cols + ["dp"]).copy()
        d_clean.reset_index(drop=True, inplace=True)

        X = d_clean[self.feat_cols]
        y = d_clean["dp"]

        split_idx        = int(len(d_clean) * 0.80)
        self.train_mask  = d_clean.index < split_idx
        self.test_mask   = d_clean.index >= split_idx
        self._model_df   = d_clean

        print("  Total usable rows :  " + str(len(d_clean)))
        print("  Features          :  " + str(len(self.feat_cols)))
        print("  Train rows        :  " + str(self.train_mask.sum()) +
              "  (t = 0 ... " + str(round(d_clean.loc[self.train_mask, "time"].max()/60, 1)) + " min)")
        print("  Test  rows        :  " + str(self.test_mask.sum()) +
              "  (t = " + str(round(d_clean.loc[self.test_mask, "time"].min()/60, 1)) +
              " ... " + str(round(d_clean.loc[self.test_mask, "time"].max()/60, 1)) + " min)")
        return X, y

    # ------------------------------------------------------------------
    # 6.  XGBOOST TRAINING
    # ------------------------------------------------------------------
    def train_xgboost_model(self, X, y, optimize=True):
        if X is None or y is None:
            return None

        print("\n" + "=" * 60)
        print("XGBOOST TRAINING")
        print("=" * 60)

        X_train = X[self.train_mask].values
        y_train = y[self.train_mask].values
        X_test  = X[self.test_mask].values
        y_test  = y[self.test_mask].values

        X_tr_sc = self.scaler.fit_transform(X_train)
        X_te_sc = self.scaler.transform(X_test)

        tscv = TimeSeriesSplit(n_splits=5)

        if optimize:
            print("  Running hyperparameter search ...")
            param_dist = {
                "n_estimators"    : [100, 200, 300, 500],
                "max_depth"       : [3, 4, 5, 6, 7, 8],
                "learning_rate"   : [0.01, 0.05, 0.1, 0.15, 0.2],
                "subsample"       : [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5, 7],
                "gamma"           : [0, 0.05, 0.1, 0.2, 0.3],
                "reg_alpha"       : [0, 0.01, 0.1, 1.0],
                "reg_lambda"      : [0.1, 1.0, 5.0, 10.0],
            }
            base = xgb.XGBRegressor(objective="reg:squarederror",
                                    tree_method="hist", random_state=42, verbosity=0)
            rs = RandomizedSearchCV(estimator=base,
                                    param_distributions=param_dist,
                                    n_iter=60, cv=tscv,
                                    scoring="neg_root_mean_squared_error",
                                    n_jobs=-1, random_state=42, verbose=0)
            rs.fit(X_tr_sc, y_train)
            best_params = rs.best_params_
            cv_rmse     = -rs.best_score_
            print("  CV RMSE (train) :  " + str(round(cv_rmse, 4)) + " psi")
            print("  Best params:")
            for k, v in sorted(best_params.items()):
                print("    " + k.ljust(25) + ": " + str(v))
            model = xgb.XGBRegressor(objective="reg:squarederror",
                                     tree_method="hist", random_state=42,
                                     verbosity=0, **best_params)
        else:
            model = xgb.XGBRegressor(objective="reg:squarederror",
                                     tree_method="hist", n_estimators=300,
                                     max_depth=5, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=42, verbosity=0)

        model.fit(X_tr_sc, y_train,
                  eval_set=[(X_te_sc, y_test)], verbose=False)

        def metrics(y_true, y_pred, label):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae  = mean_absolute_error(y_true, y_pred)
            r2   = r2_score(y_true, y_pred)
            self._log("  " + label.ljust(8) + "  RMSE = " + str(round(rmse, 4)) +
                      " psi  |  MAE = " + str(round(mae, 4)) + " psi  |  R2 = " + str(round(r2, 4)))
            return rmse, mae, r2

        y_tr_pred = model.predict(X_tr_sc)
        y_te_pred = model.predict(X_te_sc)

        self._log("\n" + "=" * 60)
        self._log("XGBOOST MODEL PERFORMANCE")
        self._log("=" * 60)
        metrics(y_train, y_tr_pred, "Train")
        _, _, r2_te = metrics(y_test, y_te_pred, "Test")

        self.xgb_model = model
        self._model_df.loc[self.test_mask,  "dp_pred"] = y_te_pred
        self._model_df.loc[self.train_mask, "dp_pred"] = y_tr_pred

        d = self._model_df
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("XGBoost  -  Actual vs Predicted  dP", fontsize=13)

        ax = axes[0]
        ax.scatter(d.loc[self.train_mask, "dp"], d.loc[self.train_mask, "dp_pred"],
                   s=3, alpha=0.3, color=_C["flow"], label="Train")
        ax.scatter(d.loc[self.test_mask, "dp"], d.loc[self.test_mask, "dp_pred"],
                   s=5, alpha=0.5, color=_C["test"], label="Test (out-of-sample)")
        lo = min(y.min(), d["dp_pred"].min())
        hi = max(y.max(), d["dp_pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1:1 line")
        ax.set_xlabel("Actual dP  [psi]")
        ax.set_ylabel("Predicted dP  [psi]")
        ax.legend()
        ax.set_title("Actual vs Predicted  (Test R2 = " + str(round(r2_te, 4)) + ")")

        ax = axes[1]
        ax.plot(d["time_min"], d["dp"], color=_C["dp"], lw=1.0, alpha=0.7, label="Actual dP")
        ax.plot(d.loc[self.test_mask, "time_min"],
                d.loc[self.test_mask, "dp_pred"],
                color=_C["test"], lw=1.5, label="XGBoost prediction (test set)")
        ax.axhline(self.DP_THRESH_PSI, color=_C["thr"], ls="--",
                   label="Threshold (" + str(self.DP_THRESH_PSI) + " psi)")
        t_split = d.loc[self.test_mask, "time_min"].min()
        ax.axvspan(t_split, d["time_min"].max(), color=_C["test"],
                   alpha=0.07, label="Test region")
        ax.set_xlabel("Time  [min]")
        ax.set_ylabel("dP  [psi]")
        ax.legend(fontsize=8)
        ax.set_title("Time-Series View  (test = last 20%)")
        plt.tight_layout()
        self._save(fig, "04_xgboost_predictions")

        residuals = d["dp"] - d["dp_pred"]
        ncols = 3 if _HAS_SM else 2
        fig, axes = plt.subplots(1, ncols, figsize=(16, 5))
        fig.suptitle("XGBoost Residual Diagnostics", fontsize=13)
        axes[0].scatter(d["time_min"], residuals, s=3, alpha=0.3, color=_C["dp"])
        axes[0].axhline(0, color="k", lw=1)
        axes[0].set_xlabel("Time  [min]")
        axes[0].set_ylabel("Residual  [psi]")
        axes[0].set_title("Residuals vs Time")
        axes[1].scatter(d["dp_pred"], residuals, s=3, alpha=0.3, color=_C["dp"])
        axes[1].axhline(0, color="k", lw=1)
        axes[1].set_xlabel("Predicted dP  [psi]")
        axes[1].set_ylabel("Residual  [psi]")
        axes[1].set_title("Residuals vs Predicted")
        if _HAS_SM:
            plot_acf(residuals.dropna(), lags=60, alpha=0.05, ax=axes[2])
            axes[2].set_title("Residual Autocorrelation")
        plt.tight_layout()
        self._save(fig, "05_residuals")
        return model



    # ------------------------------------------------------------------
    # 7.  SHAP FEATURE IMPORTANCE
    # ------------------------------------------------------------------
    def analyze_feature_importance(self):
        if self.xgb_model is None:
            print("Train model first.")
            return

        print("\n" + "=" * 60)
        print("SHAP FEATURE IMPORTANCE")
        print("=" * 60)

        d_test  = self._model_df[self.test_mask]
        X_test  = d_test[self.feat_cols].values
        X_te_sc = self.scaler.transform(X_test)

        explainer   = shap.Explainer(self.xgb_model)
        shap_values = explainer(X_te_sc)

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_te_sc,
                          feature_names=self.feat_cols, show=False, plot_size=None)
        plt.title("SHAP Summary  -  Test Set", fontsize=13)
        plt.tight_layout()
        self._save(fig, "06_shap_summary")

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        order        = np.argsort(mean_abs_shap)[::-1]
        feat_sorted  = [self.feat_cols[i] for i in order]
        shap_sorted  = mean_abs_shap[order]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(feat_sorted[::-1], shap_sorted[::-1],
                color=_C["perm"], edgecolor="white")
        ax.set_xlabel("Mean |SHAP value|  [psi]")
        ax.set_title("Feature Importance  -  Mean |SHAP|  (test set)")
        plt.tight_layout()
        self._save(fig, "07_shap_bar")

        self._log("\n" + "=" * 60)
        self._log("FEATURE IMPORTANCE  (mean |SHAP| on test set)")
        self._log("=" * 60)
        for f, s in zip(feat_sorted, shap_sorted):
            self._log("  " + f.ljust(25) + ":  " + str(round(s, 4)) + " psi")

    # ------------------------------------------------------------------
    # 8.  CLOGGING-TIME PREDICTION + RUL
    # ------------------------------------------------------------------
    def predict_clogging_time(self):
        if self.data is None:
            print("Load data first.")
            return

        self._log("\n" + "=" * 60)
        self._log("CLOGGING-TIME PREDICTION")
        self._log("=" * 60)

        t_clog_phys = self.physics.get("t_clog", None)
        if t_clog_phys and not np.isnan(t_clog_phys) and t_clog_phys > 0:
            self._log("  [Physics model]   t_clog = " + str(int(t_clog_phys)) +
                      " s  (" + str(round(t_clog_phys/60, 1)) + " min)")
        else:
            self._log("  [Physics model]   Could not compute clogging time.")
            t_clog_phys = None

        def dp_model(t_, a, b, c):
            return a * (1 - np.exp(-t_ / b)) + c

        d_tr = self._model_df[self.train_mask].copy() if hasattr(self, "_model_df") else self.data.copy()
        t_tr = d_tr["time"].values
        y_tr = d_tr["dp"].values

        t_clog_emp = None
        a_fit = b_fit = c_fit = None
        try:
            p0 = (y_tr.max(), t_tr.max() / 2, y_tr.min())
            popt, _ = curve_fit(dp_model, t_tr, y_tr, p0=p0,
                                maxfev=20000, bounds=([0, 1, -5], [500, 1e6, 50]))
            a_fit, b_fit, c_fit = popt
            t_ext  = np.linspace(t_tr[0], t_tr[-1] * 3, 5000)
            dp_ext = dp_model(t_ext, a_fit, b_fit, c_fit)
            cross  = np.where(dp_ext >= self.DP_THRESH_PSI)[0]
            if len(cross):
                t_clog_emp = t_ext[cross[0]]
                self._log("  [Empirical fit]   t_clog = " + str(int(t_clog_emp)) +
                          " s  (" + str(round(t_clog_emp/60, 1)) + " min)")
            else:
                self._log("  [Empirical fit]   Clogging not predicted in extended range.")
        except Exception as exc:
            print("  WARNING  Empirical dP fit failed: " + str(exc))

        clog_cross    = self.data[self.data["clogged"] == 1]
        t_clog_actual = None
        if len(clog_cross):
            t_clog_actual = clog_cross["time"].iloc[0]
            self._log("  [Actual data]     t_clog = " + str(int(t_clog_actual)) +
                      " s  (" + str(round(t_clog_actual/60, 1)) + " min)")

        t_clog_xgb = None
        if self.xgb_model is not None and hasattr(self, "_model_df"):
            dp_pred_all = self._model_df["dp_pred"]
            cross_xgb   = self._model_df[(self.test_mask) & (dp_pred_all >= self.DP_THRESH_PSI)]
            if len(cross_xgb):
                t_clog_xgb = cross_xgb["time"].iloc[0]
                self._log("  [XGBoost test]    dP threshold entered at  t = " +
                          str(int(t_clog_xgb)) + " s  (" + str(round(t_clog_xgb/60, 1)) + " min)")

        best_t_clog = t_clog_phys or t_clog_emp
        if best_t_clog:
            self.data["RUL_s"]   = (best_t_clog - self.data["time"]).clip(lower=0)
            self.data["RUL_min"] = self.data["RUL_s"] / 60
            last_rul = self.data["RUL_min"].iloc[-1]
            self._log("\n  RUL at last data point :  " + str(round(last_rul, 1)) +
                      " min  (physics model, t_clog = " + str(round(best_t_clog/60, 1)) + " min)")
            if last_rul <= 0:
                self._log("  WARNING  Filter is CLOGGED -- replacement required.")

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
        fig.suptitle("Clogging-Time Prediction  &  Remaining Useful Life", fontsize=13)

        ax1 = axes[0]
        t_all = self.data["time_min"]
        ax1.plot(t_all, self.data["dp"], color=_C["dp"], lw=1.2, label="Actual dP")
        ax1.axhline(self.DP_THRESH_PSI, color=_C["thr"], ls="--", lw=1.6,
                    label="Alarm threshold (" + str(self.DP_THRESH_PSI) + " psi)")
        if a_fit is not None:
            t_show = np.linspace(t_tr[0], max(t_tr[-1], t_clog_emp or t_tr[-1]) * 1.02, 600)
            ax1.plot(t_show / 60, dp_model(t_show, a_fit, b_fit, c_fit),
                     color=_C["fit"], ls="-.", lw=1.6, label="Empirical fit")
        if self.xgb_model is not None and hasattr(self, "_model_df"):
            d_te = self._model_df[self.test_mask]
            ax1.plot(d_te["time_min"], d_te["dp_pred"],
                     color=_C["test"], lw=1.8, label="XGBoost (test set)")
        for t_val, label, col in [
            (t_clog_actual, "Actual crossing", _C["clog"]),
            (t_clog_phys,   "Physics model",   _C["perm"]),
            (t_clog_emp,    "Empirical fit",    _C["fit"]),
        ]:
            if t_val and not (isinstance(t_val, float) and np.isnan(t_val)) and t_val > 0:
                ax1.axvline(t_val / 60, color=col, ls=":", lw=1.6,
                            label=label + " (" + str(round(t_val/60, 1)) + " min)")
        ax1.set_ylabel("Pressure Drop  [psi]")
        ax1.set_xlabel("Time  [min]")
        ax1.legend(fontsize=8, loc="upper left")
        ax1.set_title("dP  -  Actual vs Model Predictions")

        ax2 = axes[1]
        if "RUL_min" in self.data.columns:
            ax2.plot(t_all, self.data["RUL_min"], color=_C["perm"], lw=1.8, label="RUL (physics model)")
            ax2.axhline(0, color="red", ls="--", lw=1.2)
            ax2.fill_between(t_all, 0, self.data["RUL_min"],
                             where=self.data["RUL_min"] > 0,
                             color=_C["perm"], alpha=0.15)
            ax2.set_ylabel("Remaining Useful Life  [min]")
            ax2.set_xlabel("Time  [min]")
            ax2.set_title("Remaining Useful Life  (RUL)")
            ax2.set_ylim(bottom=0)
            ax2.legend()
        else:
            ax2.set_visible(False)
        plt.tight_layout()
        self._save(fig, "08_clogging_prediction_rul")

        return {"t_clog_physics": t_clog_phys, "t_clog_empirical": t_clog_emp,
                "t_clog_actual": t_clog_actual, "t_clog_xgb_test": t_clog_xgb}

    # ------------------------------------------------------------------
    # 9.  SUMMARY REPORT
    # ------------------------------------------------------------------
    def generate_report(self):
        path = os.path.join(self.output_dir, "report.txt")
        header = [
            "=" * 60,
            "  FILTER ANALYSIS REPORT",
            "  File    : " + os.path.basename(self.file_path),
            "  Filter area  : " + str(self.filter_area) + " m2",
            "  Filter depth : " + str(round(self.filter_length*100, 1)) + " cm",
            "=" * 60,
        ]
        lines = header + self._report
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print("  Report saved -> " + path)
        csv_path = os.path.join(self.output_dir, "processed_data.csv")
        self.data.to_csv(csv_path, index=False)
        print("  Processed data -> " + csv_path)

    # ------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------
    def run_full_analysis(self, optimize_xgb=True):
        print("\n" + "=" * 60)
        print("  FILTER ANALYSIS PIPELINE  -  START")
        print("=" * 60 + "\n")
        if not self.load_data():
            return False
        self.plot_raw_data()
        self.check_data_quality()
        self.track_permeability()
        X, y = self.prepare_xgboost_data()
        if X is not None:
            self.train_xgboost_model(X, y, optimize=optimize_xgb)
            self.analyze_feature_importance()
        self.predict_clogging_time()
        self.generate_report()
        print("\n" + "=" * 60)
        print("  ANALYSIS COMPLETE")
        print("  All outputs saved to :  " + self.output_dir)
        print("=" * 60)
        return True


# ============================================================
# Stand-alone entry point
# ============================================================
def main():
    import glob as _glob
    excel_files = _glob.glob("*.xlsx") + _glob.glob("*.xls")
    if not excel_files:
        file_path = input("Enter path to Excel file: ").strip()
    elif len(excel_files) == 1:
        file_path = excel_files[0]
        print("Using: " + file_path)
    else:
        print("Found Excel files:")
        for i, f in enumerate(excel_files, 1):
            print("  " + str(i) + ". " + f)
        sel = int(input("Select file number: ")) - 1
        file_path = excel_files[sel]

    analyzer = FilterAnalysis(
        file_path=file_path,
        filter_area=2.0,
        filter_length=0.02,
        show_plots=True,
    )
    analyzer.run_full_analysis(optimize_xgb=True)


if __name__ == "__main__":
    main()
