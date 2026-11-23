"""Causal feature engineering.

Fixes the two central defects of the legacy ``ProductionFeatureEngineer``:

1. *Train/serve mismatch* — legacy trained with ``.rolling(min_periods=1)``
   but predicted with ``.expanding()``. Here a single ``_compute`` path is
   used everywhere.
2. *Label circularity* — the precomputed label-rule columns
   (``validation_score``, ``adaptive_cusum``, ``short_ema``, ``long_ema``,
   ``cusum_threshold``) and the absolute ``time`` index are banned from the
   leakage-free design matrix; the pipeline raises if they slip in.

All row-t features use only samples <= t (rolling windows ending at t,
recursive EMA/CUSUM). Windows use ``min_periods=window``; the resulting
warm-up rows carry NaN and are excluded from modeling sets by the caller
instead of being zero-filled (legacy zero-fill fabricated plausible values).

EMA spans (10/50) and the CUSUM reference (k = 0.5 after z-normalization) are
fixed a priori and deliberately different from the label rule's parameters
(EMA 50/200, adaptive CUSUM threshold), so these features are causal
transforms of the raw sensors, not reconstructions of the rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BANNED_COLUMNS

RAW_SENSORS = ("flowrate", "dp")


class CausalFeatureEngineer:
    """sklearn-style fit/transform; fit learns *train-only* statistics."""

    def __init__(
        self,
        feature_set: str = "leakage_free",
        windows: tuple[int, ...] = (5, 15, 30),
        ema_spans: tuple[int, int] = (10, 50),
        cusum_k: float = 0.5,
    ) -> None:
        if feature_set not in ("leakage_free", "thesis_full"):
            raise ValueError(f"unknown feature_set {feature_set!r}")
        self.feature_set = feature_set
        self.windows = tuple(windows)
        self.ema_spans = ema_spans
        self.cusum_k = cusum_k
        self.stats_: dict[str, float] = {}
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(self, df_train: pd.DataFrame) -> "CausalFeatureEngineer":
        s: dict[str, float] = {}
        for col in RAW_SENSORS:
            x = df_train[col]
            s[f"{col}_mean"] = float(x.mean())
            s[f"{col}_std"] = float(x.std(ddof=0)) or 1.0
        s["max_flow"] = float(df_train["flowrate"].max())
        # baseline resistance: median over the first 20% of TRAIN rows
        # (mirrors the thesis's "baseline = initial 20%" convention but is
        # computed on training data only)
        n_base = max(len(df_train) // 5, 10)
        base = df_train.iloc[:n_base]
        resistance = base["dp"] / base["flowrate"].clip(lower=0.5)
        s["baseline_resistance"] = float(resistance.median()) or 1e-6
        self.stats_ = s
        self.fitted_ = True
        return self

    # ------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("fit() must be called before transform()")
        X = self._compute(df)
        if self.feature_set == "thesis_full":
            X = pd.concat([X, self._label_rule_columns(df)], axis=1)
        self.assert_no_banned(X)
        return X

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df_train).transform(df_train)

    def assert_no_banned(self, X: pd.DataFrame) -> None:
        if self.feature_set != "leakage_free":
            return
        leaked = [c for c in X.columns if c in BANNED_COLUMNS]
        if leaked:
            raise ValueError(
                f"banned label-constituent columns present in leakage-free "
                f"design matrix: {leaked}"
            )

    def monotone_constraints(self, columns: list[str]) -> tuple[int, ...]:
        """Physics-informed monotonicity map for XGBoost.

        Per the thesis's Darcy/Forchheimer analysis, progressive clogging
        raises differential pressure and flow resistance and lowers the
        achievable flow. Encoding this as monotone constraints pins the
        model's *extrapolation direction* in regions the single-lifecycle
        training data does not cover (the test window's dp band lies between
        the healthy and clogged training segments).

        Principle: **levels constrained, dynamics unconstrained**. Level
        features (dp, rolling means, EMAs, resistance, cumulative CUSUM, flow
        levels) relate monotonically to the clogged state. Rate/deviation
        features (differences, slopes = fouling rate, local MAD-z,
        volatilities) are large during *active* clogging but return toward
        zero on the clogged plateau — a non-monotone relationship, so
        constraining them would encode false physics. +1 = risk
        non-decreasing, -1 = non-increasing, 0 = unconstrained.
        """
        increasing_prefixes = (
            "dp", "resistance", "ema_spread", "ema_ratio",
        )
        decreasing_prefixes = ("flowrate", "flow_efficiency")
        ambiguous_markers = (
            "_diff1", "_rate", "_rstd_", "_rchg_", "_slope_", "_madz_", "interaction",
        )

        cons = []
        for col in columns:
            if any(m in col for m in ambiguous_markers):
                cons.append(0)
            elif any(col.startswith(p) for p in increasing_prefixes):
                cons.append(1)
            elif any(col.startswith(p) for p in decreasing_prefixes):
                cons.append(-1)
            else:
                cons.append(0)
        return tuple(cons)

    @property
    def warmup_rows(self) -> int:
        """Rows at the start of any transformed frame that contain NaN from
        rolling warm-up and must be excluded from modeling sets. The MAD-z
        feature nests two trailing windows, so warm-up is twice the largest
        window."""
        return 2 * max(self.windows)

    # ---------------------------------------------------------------- internals
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        s = self.stats_
        out: dict[str, pd.Series] = {}

        for col in RAW_SENSORS:
            x = df[col].astype(float)
            out[col] = x
            out[f"{col}_norm"] = (x - s[f"{col}_mean"]) / s[f"{col}_std"]
            out[f"{col}_diff1"] = x.diff()
            out[f"{col}_rate"] = x.pct_change(fill_method=None).clip(-10, 10)
            for w in self.windows:
                r = x.rolling(window=w, min_periods=w)
                out[f"{col}_rmean_{w}"] = r.mean()
                out[f"{col}_rstd_{w}"] = r.std(ddof=0)
                out[f"{col}_rchg_{w}"] = x - x.shift(w)

        dp = df["dp"].astype(float)
        flow = df["flowrate"].astype(float)

        # rise-rate: OLS slope of dp over the trailing window
        for w in (15, 30):
            out[f"dp_slope_{w}"] = _rolling_slope(dp, w)

        # robust deviation from the local (trailing) level
        w = 30
        med = dp.rolling(w, min_periods=w).median()
        mad = (dp - med).abs().rolling(w, min_periods=w).median()
        out["dp_madz_30"] = (dp - med) / (1.4826 * mad + 1e-9)

        # physics-based (see physics.py / docs/physics_formulation.md):
        # under the constant-flow protocol the identifiable state variable is
        # the specific flow resistance R = dp/Q (Darcy: R ∝ 1/permeability);
        # its trailing slope is the instantaneous fouling rate dR/dt
        resistance = dp / flow.clip(lower=0.5)
        out["resistance"] = resistance
        out["resistance_ratio"] = resistance / s["baseline_resistance"]
        for w in (15, 30):
            out[f"resistance_slope_{w}"] = _rolling_slope(resistance, w)
        out["flow_efficiency"] = (flow / s["max_flow"]).clip(0, 2)
        out["flow_dp_interaction"] = (flow / s["flowrate_mean"]) * (dp / max(s["dp_mean"], 1e-9))

        # causally recomputed EMA (recursive => past-only)
        fast = dp.ewm(span=self.ema_spans[0], adjust=False).mean()
        slow = dp.ewm(span=self.ema_spans[1], adjust=False).mean()
        out["dp_ema_fast"] = fast
        out["dp_ema_slow"] = slow
        out["ema_spread"] = fast - slow
        out["ema_ratio"] = fast / (slow + 1e-9)

        # causal one-sided CUSUM on z-normalized dp, fixed reference k
        z = (dp - s["dp_mean"]) / s["dp_std"]
        out["dp_cusum_pos"] = _one_sided_cusum(z.to_numpy(), self.cusum_k)

        X = pd.DataFrame(out, index=df.index)
        return X.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _label_rule_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ablation-only: the precomputed label-rule columns exactly as the
        legacy `_add_domain_features` consumed them."""
        cols = ["validation_score", "cusum_threshold", "adaptive_cusum", "short_ema", "long_ema"]
        X = df[cols].astype(float).copy()
        X["thesis_ema_spread"] = X["short_ema"] - X["long_ema"]
        X["thesis_ema_ratio"] = X["short_ema"] / (X["long_ema"] + 1e-9)
        return X


def _rolling_slope(x: pd.Series, window: int) -> pd.Series:
    """OLS slope of x against 0..w-1 over a trailing window, via the closed
    form slope = cov(t, x) / var(t) with constant t weights (O(n*w))."""
    t = np.arange(window, dtype=float)
    t_center = t - t.mean()
    denom = float((t_center**2).sum())

    def slope(vals: np.ndarray) -> float:
        return float((t_center * (vals - vals.mean())).sum() / denom)

    return x.rolling(window, min_periods=window).apply(slope, raw=True)


def _one_sided_cusum(z: np.ndarray, k: float) -> np.ndarray:
    """S_t = max(0, S_{t-1} + z_t - k); recursive, strictly causal."""
    out = np.empty_like(z, dtype=float)
    acc = 0.0
    for i, val in enumerate(z):
        acc = max(0.0, acc + val - k)
        out[i] = acc
    return out
