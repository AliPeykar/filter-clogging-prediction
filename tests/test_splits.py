"""Split assertions: disjointness, purge gaps, class composition per plan."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filterclog.config import SplitConfig
from filterclog.splitting import describe_split, make_splits

N = 11519
TRANSITION_0BASED = 8942  # first clogged row (1-based row 8,943)


@pytest.fixture(scope="module")
def y():
    yy = np.zeros(N, dtype=int)
    yy[TRANSITION_0BASED:] = 1
    return yy


def test_held_out_transition_composition(y):
    split = make_splits(N, SplitConfig())
    split.assert_disjoint(N)
    d = describe_split(split, y)
    # test window straddles the onset: both classes present
    assert d["test"]["n_pos"] > 0 and d["test"]["n_neg"] > 0
    assert d["test"]["n"] == 1740
    assert d["test"]["n_neg"] == 1082 and d["test"]["n_pos"] == 658
    # train contains both classes (healthy head + clogged tail)
    assert d["train"]["n_pos"] > 0 and d["train"]["n_neg"] > 0
    assert d["train"]["n"] == 7140 + 1199
    # val balanced-ish: 600 healthy + 600 clogged
    assert d["val"]["n"] == 1200
    assert d["val"]["n_pos"] == 600


def test_purge_gaps_at_least_60():
    cfg = SplitConfig()
    split = make_splits(N, cfg)
    segs = [cfg.train_a, cfg.val_a, cfg.test, cfg.val_b, cfg.train_b]
    for (lo1, hi1), (lo2, hi2) in zip(segs, segs[1:]):
        assert lo2 - hi1 - 1 >= 60


def test_strict_temporal_one_class_test(y):
    split = make_splits(N, SplitConfig(scheme="strict_temporal"))
    split.assert_disjoint(N)
    d = describe_split(split, y)
    assert d["test"]["n_neg"] == 0  # documents the degenerate one-class test
    assert d["train"]["n_pos"] > 0  # train sees the transition
    assert len(split.val) == 0  # S1 reuses S2's tuning; no val block
    # purge gap between train and test
    assert split.test.min() - split.train.max() - 1 >= 60


def test_no_index_overlap_or_out_of_range():
    for scheme in ("held_out_transition", "strict_temporal"):
        split = make_splits(N, SplitConfig(scheme=scheme))
        all_idx = np.concatenate([split.train, split.val, split.test, split.purged])
        assert len(np.unique(all_idx)) == len(all_idx)
        assert all_idx.min() >= 0 and all_idx.max() < N
