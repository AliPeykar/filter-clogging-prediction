"""Time-series-aware cross-validation folds.

Replaces the legacy ``StratifiedKFold(shuffle=True)`` (which shuffles a
strongly autocorrelated series, leaking near-identical neighbouring rows
between folds) with *segment-wise expanding-window* folds.

Why segment-wise: the S2 training set is two temporally contiguous segments —
the healthy head (Train-A) and the clogged tail (Train-B). A plain expanding
window over the concatenated index would produce one-class fit sets (all
positives sit at the end), so instead fold i fits on the first i/(n+1)
fraction *of each segment* and validates on the following block of each
segment, with a purge gap between fit and validation inside every segment.
Within-segment temporal order is never violated; the across-segment caveat is
the same one already attached to the S2 split itself.

Folds whose fit set lacks a class or whose validation block lacks positives
are dropped with a warning; at least ``min_usable`` folds must remain.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _contiguous_segments(idx: np.ndarray) -> list[np.ndarray]:
    idx = np.sort(np.asarray(idx))
    breaks = np.flatnonzero(np.diff(idx) > 1)
    return np.split(idx, breaks + 1)


def expanding_window_folds(
    train_idx: np.ndarray,
    y: np.ndarray,
    n_splits: int = 4,
    gap: int = 60,
    min_usable: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(fit_idx, val_idx), ...] of positional indices (subsets of
    ``train_idx``). Handles multi-segment training sets (see module doc)."""
    segments = _contiguous_segments(train_idx)
    boundaries = np.linspace(0, 1, n_splits + 2)  # fold i fits up to boundaries[i+1]

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_splits + 1):
        fit_parts, val_parts = [], []
        for seg in segments:
            n = len(seg)
            fit_end = int(n * boundaries[i])
            val_start = fit_end + gap
            val_end = int(n * boundaries[i + 1])
            if fit_end > 0:
                fit_parts.append(seg[:fit_end])
            if val_start < val_end:
                val_parts.append(seg[val_start:val_end])
        if not fit_parts or not val_parts:
            continue
        fit_idx = np.concatenate(fit_parts)
        val_idx = np.concatenate(val_parts)
        y_fit, y_val = y[fit_idx], y[val_idx]
        if len(np.unique(y_fit)) < 2 or (y_val == 1).sum() == 0:
            logger.warning(
                "fold %d dropped: fit classes=%s, val positives=%d",
                i, np.unique(y_fit).tolist(), int((y_val == 1).sum()),
            )
            continue
        folds.append((fit_idx, val_idx))

    if len(folds) < min_usable:
        raise RuntimeError(
            f"only {len(folds)} usable expanding-window folds (need >= {min_usable}); "
            "training composition cannot support time-series CV"
        )
    return folds
