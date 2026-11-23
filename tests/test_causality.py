"""Leakage unit test: truncation equivalence.

For any row t, features computed on the truncated frame df[:t+1] must equal
row t of features computed on the full frame — i.e. no feature can depend on
data after t. This is the property the legacy pipeline violated (its rolling/
expanding paths disagreed between training and prediction).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filterclog.features import CausalFeatureEngineer


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame(
        {
            "flowrate": 7 + rng.normal(0, 0.5, n).cumsum() * 0.01,
            "dp": np.linspace(0, 30, n) + rng.normal(0, 0.8, n),
        }
    )


@pytest.fixture(scope="module")
def engineer(df):
    return CausalFeatureEngineer().fit(df.iloc[:300])


def test_truncation_equivalence(df, engineer):
    full = engineer.transform(df)
    rng = np.random.default_rng(1)
    for t in rng.integers(50, len(df), size=12):
        trunc = engineer.transform(df.iloc[: t + 1])
        pd.testing.assert_series_equal(
            trunc.iloc[-1], full.iloc[t], check_names=False, rtol=1e-10, atol=1e-12
        )


def test_warmup_rows_are_nan_not_zero(df, engineer):
    X = engineer.transform(df)
    w = engineer.warmup_rows
    # the nested MAD-z window is still warming up right before the boundary
    assert X.iloc[w - 3].isna().any()
    assert not X.iloc[w:].isna().any().any()


def test_banned_columns_raise(df):
    eng = CausalFeatureEngineer().fit(df.iloc[:300])
    X = eng.transform(df)
    X["validation_score"] = 1.0
    with pytest.raises(ValueError, match="banned"):
        eng.assert_no_banned(X)


def test_single_code_path_train_equals_predict(df, engineer):
    # same frame transformed twice must be identical (no mode switching)
    a = engineer.transform(df)
    b = engineer.transform(df.copy())
    pd.testing.assert_frame_equal(a, b)
