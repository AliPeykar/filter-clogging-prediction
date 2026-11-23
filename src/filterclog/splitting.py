"""Temporal splits with purge gaps.

With a single healthy->clogged transition (row 8,943) no contiguous temporal
split puts both classes on both sides. Two schemes are provided and the
trade-off is reported explicitly:

* ``held_out_transition`` (S2, primary): the test window straddles the onset;
  training data is the healthy head (Train-A) plus the fully-clogged tail
  (Train-B). Caveat carried into every report: Train-B post-dates the test
  window, so this evaluates interpolation across a held-out transition, not
  pure forecasting — the unavoidable cost of one lifecycle.
* ``strict_temporal`` (S1, secondary): train on the past, test on the future.
  Causally clean, but the test window is one-class (all clogged); only
  sustained-detection behaviour is meaningful. Included to demonstrate why the
  legacy 80/20 temporal evaluation produced vacuous 1.0 metrics.

Purge gaps (>= max rolling window and max horizon) are dropped between
adjacent segments so trailing-window features and shifted horizon labels
cannot bleed across split boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SplitConfig


@dataclass
class SplitIndices:
    """0-based positional indices into the time-ordered frame."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    purged: np.ndarray
    scheme: str
    segments: dict[str, tuple[int, int]] = field(default_factory=dict)

    def assert_disjoint(self, n_rows: int) -> None:
        parts = [self.train, self.val, self.test, self.purged]
        total = np.concatenate(parts)
        if len(np.unique(total)) != len(total):
            raise AssertionError("split index overlap detected")
        if total.max() >= n_rows or total.min() < 0:
            raise AssertionError("split index out of range")


def _rows(bounds: tuple[int, int]) -> np.ndarray:
    """1-based inclusive bounds -> 0-based positional indices."""
    lo, hi = bounds
    return np.arange(lo - 1, hi)


def make_splits(n_rows: int, cfg: SplitConfig) -> SplitIndices:
    if cfg.scheme == "held_out_transition":
        segments = {
            "train_a": cfg.train_a,
            "val_a": cfg.val_a,
            "test": cfg.test,
            "val_b": cfg.val_b,
            "train_b": cfg.train_b,
        }
        train = np.concatenate([_rows(cfg.train_a), _rows(cfg.train_b)])
        val = np.concatenate([_rows(cfg.val_a), _rows(cfg.val_b)])
        test = _rows(cfg.test)
    elif cfg.scheme == "strict_temporal":
        # S1 has no validation block: the only positives in the training range
        # are its final rows, so any held-out tail would strip the training
        # set of one class entirely. S1 runs therefore reuse hyperparameters,
        # iteration count, and decision threshold tuned under S2 and serve
        # purely as a demonstration of the degenerate one-class future window.
        segments = {"train": cfg.strict_train, "test": cfg.strict_test}
        train = _rows(cfg.strict_train)
        val = np.array([], dtype=int)
        test = _rows(cfg.strict_test)
    else:
        raise ValueError(f"unknown split scheme {cfg.scheme!r}")

    used = set(np.concatenate([train, val, test]).tolist())
    purged = np.array(sorted(set(range(n_rows)) - used), dtype=int)

    split = SplitIndices(
        train=np.sort(train), val=np.sort(val), test=np.sort(test),
        purged=purged, scheme=cfg.scheme, segments=segments,
    )
    split.assert_disjoint(n_rows)
    return split


def describe_split(split: SplitIndices, y: np.ndarray) -> dict:
    def stats(idx: np.ndarray) -> dict:
        yy = y[idx]
        return {
            "n": int(len(idx)),
            "n_pos": int((yy == 1).sum()),
            "n_neg": int((yy == 0).sum()),
            "pos_frac": round(float((yy == 1).mean()), 4) if len(idx) else 0.0,
        }

    return {
        "scheme": split.scheme,
        "segments_1based": {k: list(v) for k, v in split.segments.items()},
        "train": stats(split.train),
        "val": stats(split.val),
        "test": stats(split.test),
        "n_purged": int(len(split.purged)),
    }
