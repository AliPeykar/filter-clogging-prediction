"""Dataset loading, structural validation, and sampling-interval verification.

The thesis text is internally inconsistent about the sampling interval
(11,519 rows over a ~11.5 h experiment is ~3.6 s/row, not the 2-min storage
interval quoted for the industrial SCADA data), so the interval is *verified
from the data* here and every downstream "minutes" figure is derived from it
rather than hard-coded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "time",
    "flowrate",
    "dp",
    "short_ema",
    "long_ema",
    "adaptive_cusum",
    "cusum_threshold",
    "validation_score",
    "filter_status",
]

#: Duration of the lab experiment reported in the thesis (phase 2, constant
#: flow with powder injection). Used only to derive seconds-per-row.
THESIS_EXPERIMENT_HOURS = 11.5


def load_dataset(path: str | Path, sheet_name: str = "Comprehensive_Analysis") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_excel(path, sheet_name=sheet_name)
    logger.info("loaded %s: %d rows x %d cols", path.name, len(df), df.shape[1])
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Hard assertions on the known structure of the lab dataset."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected columns: {missing}")
    if df[EXPECTED_COLUMNS[:-1]].isna().any().any():
        raise ValueError("unexpected NaN in sensor/score columns")
    if df.duplicated().any():
        raise ValueError("unexpected duplicate rows")
    time = df["time"].to_numpy()
    if not (np.diff(time) > 0).all():
        raise ValueError("time column is not strictly increasing")
    statuses = set(df["filter_status"].unique())
    if statuses != {"Not Clogged", "Clogged"}:
        raise ValueError(f"unexpected filter_status values: {statuses}")
    if (df["flowrate"] <= 0).any() or (df["dp"] < 0).any():
        raise ValueError("physically implausible sensor values (flow<=0 or dp<0)")


def verify_sampling_interval(df: pd.DataFrame) -> dict:
    """Infer per-row timing.

    The ``time`` column is a bare row counter (1..N), so the interval cannot be
    read from timestamps. We derive seconds-per-row from the thesis's reported
    experiment duration and record the assumption explicitly so it can be
    revised if the raw logger files ever surface.
    """
    time = df["time"].to_numpy()
    diffs = np.diff(time)
    counter_like = bool((diffs == 1).all())
    seconds_per_row = THESIS_EXPERIMENT_HOURS * 3600.0 / len(df)
    info = {
        "time_column_is_row_counter": counter_like,
        "n_rows": int(len(df)),
        "assumed_experiment_hours": THESIS_EXPERIMENT_HOURS,
        "derived_seconds_per_row": round(float(seconds_per_row), 3),
        "derived_minutes_per_row": round(float(seconds_per_row / 60.0), 5),
        "note": (
            "time column carries no units; interval derived from the thesis's "
            "~11.5 h experiment duration. The 2-min storage interval quoted in "
            "the thesis applies to the industrial SCADA data, not this lab set."
        ),
    }
    logger.info("sampling interval: %.2f s/row (derived)", seconds_per_row)
    return info
