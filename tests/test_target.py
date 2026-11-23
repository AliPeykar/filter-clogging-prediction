"""Label audit and horizon-target assertions against the real dataset."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filterclog.data import load_dataset, validate_dataset, verify_sampling_interval
from filterclog.target import audit_labels, make_binary_target, make_horizon_target


@pytest.fixture(scope="module")
def df():
    frame = load_dataset(ROOT / "Comprehensive_Filter_Analysis.xlsx")
    validate_dataset(frame)
    return frame


def test_known_shape(df):
    assert df.shape == (11519, 9)


def test_single_transition_at_8943(df):
    audit = audit_labels(df)
    assert audit["n_transitions"] == 1
    assert audit["transition_rows_1based"] == [8943]
    assert audit["n_healthy"] == 8942
    assert audit["n_clogged"] == 2577


def test_class_balance(df):
    y = make_binary_target(df)
    assert abs(y.mean() - 0.2237) < 0.001


def test_horizon_target_shift(df):
    y = make_binary_target(df)
    h = 30
    y_h = make_horizon_target(y, h)
    assert (y_h[:-h] == y[h:]).all()
    assert (y_h[-h:] == -1).all()
    # transition moves h rows earlier
    first_pos = int(np.argmax(y_h == 1))
    assert first_pos == int(np.argmax(y == 1)) - h


def test_sampling_interval_derived(df):
    info = verify_sampling_interval(df)
    assert info["time_column_is_row_counter"] is True
    # ~11.5 h over 11,519 rows -> ~3.6 s/row
    assert 3.0 < info["derived_seconds_per_row"] < 4.5
