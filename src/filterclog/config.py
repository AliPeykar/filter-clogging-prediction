"""Experiment configuration: dataclass, YAML loader, validation.

A single resolved ``ExperimentConfig`` drives the whole pipeline so that every
run is fully described by one YAML file (archived into the run's output
directory for reproducibility).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

FeatureSet = Literal["leakage_free", "thesis_full"]
SplitScheme = Literal["held_out_transition", "strict_temporal"]

#: Columns that must never appear in the design matrix in leakage-free mode.
#: validation_score IS the label rule's score; adaptive_cusum / short_ema /
#: long_ema are the precomputed label-rule inputs; cusum_threshold is a rule
#: hyperparameter (95.6% constant); time indexes the single lifecycle.
BANNED_COLUMNS = (
    "validation_score",
    "cusum_threshold",
    "adaptive_cusum",
    "short_ema",
    "long_ema",
    "time",
)


@dataclass
class SplitConfig:
    """Row boundaries (1-based, inclusive) for the held-out-transition split.

    Defaults implement scheme S2 from the plan: the test window straddles the
    single healthy->clogged transition (row 8,943) so both classes and the
    onset event itself are evaluated on unseen data. Purge gaps >= the largest
    rolling window / horizon prevent feature or label bleed across segments.
    """

    scheme: SplitScheme = "held_out_transition"
    train_a: tuple[int, int] = (1, 7140)
    val_a: tuple[int, int] = (7201, 7800)
    test: tuple[int, int] = (7861, 9600)
    val_b: tuple[int, int] = (9661, 10260)
    train_b: tuple[int, int] = (10321, 11519)
    purge_gap: int = 60
    # strict_temporal (S1) boundaries
    strict_train: tuple[int, int] = (1, 9600)
    strict_test: tuple[int, int] = (9661, 11519)


@dataclass
class HPOConfig:
    n_trials: int = 75
    n_cv_splits: int = 4
    cv_gap: int = 60
    timeout_s: int | None = None
    early_stopping_rounds: int = 50
    max_n_estimators: int = 2000


@dataclass
class ExperimentConfig:
    name: str = "main"
    data_path: str = "Comprehensive_Filter_Analysis.xlsx"
    sheet_name: str = "Comprehensive_Analysis"
    output_root: str = "outputs"
    seed: int = 42

    feature_set: FeatureSet = "leakage_free"
    rolling_windows: tuple[int, ...] = (5, 15, 30)
    ema_spans: tuple[int, int] = (10, 50)  # deliberately != label rule's 50/200
    cusum_k: float = 0.5
    #: physics-informed monotone constraints (Darcy/Forchheimer: risk
    #: non-decreasing in dp/resistance, non-increasing in flow); pins the
    #: extrapolation direction inside the dp band the training segments do
    #: not cover
    monotone_constraints: bool = True

    split: SplitConfig = field(default_factory=SplitConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)

    #: Early-warning horizons in rows; 0 = nowcast (onset detection).
    horizons: tuple[int, ...] = (0,)

    #: k-of-m sustained-alarm rule for detection delay (k consecutive rows).
    alarm_consecutive: int = 3

    calibration: Literal["sigmoid", "none"] = "sigmoid"
    threshold_rule: Literal["max_f1", "fpr_1pct"] = "max_f1"

    bootstrap_blocks: int = 50
    bootstrap_resamples: int = 1000

    run_risk_levels: bool = False
    run_shap: bool = True
    run_figures: bool = True

    def validate(self) -> None:
        s = self.split
        if s.scheme == "held_out_transition":
            segs = [s.train_a, s.val_a, s.test, s.val_b, s.train_b]
            for (a_lo, a_hi), (b_lo, b_hi) in zip(segs, segs[1:]):
                if a_hi >= b_lo:
                    raise ValueError(f"split segments overlap: {(a_lo, a_hi)} vs {(b_lo, b_hi)}")
                if b_lo - a_hi - 1 < s.purge_gap:
                    raise ValueError(
                        f"purge gap {b_lo - a_hi - 1} between {(a_lo, a_hi)} and "
                        f"{(b_lo, b_hi)} is below required {s.purge_gap}"
                    )
        elif s.scheme == "strict_temporal":
            if s.strict_train[1] >= s.strict_test[0]:
                raise ValueError("strict split: train/test overlap")
            if s.strict_test[0] - s.strict_train[1] - 1 < s.purge_gap:
                raise ValueError("strict split: purge gap below required")
        max_window = max(self.rolling_windows)
        max_h = max(self.horizons)
        if s.purge_gap < max(max_window, max_h):
            raise ValueError(
                f"purge_gap ({s.purge_gap}) must cover max rolling window "
                f"({max_window}) and max horizon ({max_h})"
            )
        if self.feature_set not in ("leakage_free", "thesis_full"):
            raise ValueError(f"unknown feature_set {self.feature_set!r}")


def _apply(cfg: object, data: dict) -> None:
    valid = {f.name for f in dataclasses.fields(cfg)}  # type: ignore[arg-type]
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"unknown config key {key!r} for {type(cfg).__name__}")
        current = getattr(cfg, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(cfg, key, tuple(tuple(v) if isinstance(v, list) else v for v in value))
        else:
            setattr(cfg, key, value)


def load_config(path: str | Path) -> ExperimentConfig:
    cfg = ExperimentConfig()
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    _apply(cfg, data)
    cfg.validate()
    return cfg


def config_to_dict(cfg: ExperimentConfig) -> dict:
    return dataclasses.asdict(cfg)
