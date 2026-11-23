"""Paper analysis A3: how the validation scheme alone changes the score.

Same leakage-free design matrix, same fixed XGBoost settings, three
resampling schemes over the full series, plus the legacy 80/20 temporal
hold-out. Nothing is tuned here; the point is the scheme, not the model.

Writes paper/03_analysis/out/a3_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filterclog.config import ExperimentConfig  # noqa: E402
from filterclog.data import load_dataset, validate_dataset  # noqa: E402
from filterclog.features import CausalFeatureEngineer  # noqa: E402
from filterclog.target import make_binary_target  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
SEED = 42
cfg = ExperimentConfig()


def model(spw: float) -> XGBClassifier:
    # library defaults except a fixed, modest ensemble size
    return XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                         scale_pos_weight=spw, random_state=SEED, n_jobs=2,
                         tree_method="hist", eval_metric="logloss")


def fit_eval(X, y, tr, te) -> dict:
    n_pos = int((y[tr] == 1).sum())
    if n_pos == 0 or n_pos == len(tr):
        return {"n_test": int(len(te)), "test_pos_frac": float(y[te].mean()),
                "note": "training fold has one class; model not fitted"}
    m = model(float((y[tr] == 0).sum() / n_pos))
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    pred = (p >= 0.5).astype(int)
    out = {
        "n_test": int(len(te)), "test_pos_frac": float(y[te].mean()),
        "accuracy": float((pred == y[te]).mean()),
        "f1": float(f1_score(y[te], pred, zero_division=0)),
        "f1_weighted": float(f1_score(y[te], pred, average="weighted", zero_division=0)),
    }
    out["roc_auc"] = (float(roc_auc_score(y[te], p))
                      if len(np.unique(y[te])) == 2 else None)
    return out


def main() -> None:
    df = load_dataset(ROOT / cfg.data_path, cfg.sheet_name)
    validate_dataset(df)
    y = make_binary_target(df)
    n = len(df)
    res: dict = {}

    def design(train_rows: np.ndarray) -> tuple[np.ndarray, int]:
        eng = CausalFeatureEngineer("leakage_free", cfg.rolling_windows, cfg.ema_spans, cfg.cusum_k)
        eng.fit(df.iloc[train_rows])
        return eng.transform(df).to_numpy(dtype=np.float32), eng.warmup_rows

    # --- (a) shuffled stratified 5-fold over the whole series (legacy HPO scheme)
    X, warm = design(np.arange(n))  # statistics from all rows, as a shuffled CV implies
    rows = np.arange(warm, n)
    folds = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(rows, y[rows]):
        folds.append(fit_eval(X, y, rows[tr], rows[te]))
    res["shuffled_stratified_5fold"] = {
        "folds": folds,
        "mean_f1": float(np.mean([f["f1"] for f in folds])),
        "mean_f1_weighted": float(np.mean([f["f1_weighted"] for f in folds])),
        "mean_roc_auc": float(np.mean([f["roc_auc"] for f in folds])),
        "min_f1": float(np.min([f["f1"] for f in folds])),
    }

    # --- (b) contiguous 5-fold, no shuffling (each fold is one block of time)
    bounds = np.linspace(warm, n, 6).astype(int)
    folds = []
    for k in range(5):
        te = np.arange(bounds[k], bounds[k + 1])
        tr = np.setdiff1d(rows, te)
        # purge 60 rows on each side of the test block
        tr = tr[(tr < bounds[k] - 60) | (tr >= bounds[k + 1] + 60)]
        f = fit_eval(X, y, tr, te)
        f["test_rows_1based"] = [int(te[0] + 1), int(te[-1] + 1)]
        folds.append(f)
    res["contiguous_5fold_purged"] = {"folds": folds}

    # --- (c) legacy 80/20 temporal hold-out
    cut = int(0.8 * n)
    Xl, warm_l = design(np.arange(cut))
    tr, te = np.arange(warm_l, cut), np.arange(cut, n)
    f = fit_eval(Xl, y, tr, te)
    f["cut_row_1based"] = cut + 1
    f["train_pos_frac"] = float(y[tr].mean())
    const = np.ones(len(te), dtype=int)  # "always clogged", no model at all
    f["constant_always_clogged"] = {
        "accuracy": float((const == y[te]).mean()),
        "f1": float(f1_score(y[te], const, zero_division=0)),
        "f1_weighted": float(f1_score(y[te], const, average="weighted", zero_division=0)),
    }
    res["legacy_80_20_temporal"] = f

    # --- (d) random stratified 80/20 hold-out (legacy GA-script scheme:
    #         train_test_split(test_size=0.2, random_state=42, stratify=y))
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(rows, test_size=0.2, random_state=SEED, stratify=y[rows])
    f = fit_eval(X, y, tr, te)
    # how close in time is each test sample to its nearest training sample?
    tr_sorted = np.sort(tr)
    pos = np.searchsorted(tr_sorted, te)
    left = np.abs(te - tr_sorted[np.clip(pos - 1, 0, len(tr_sorted) - 1)])
    right = np.abs(tr_sorted[np.clip(pos, 0, len(tr_sorted) - 1)] - te)
    gap = np.minimum(left, right)
    f["test_samples_with_training_neighbour_within_1"] = float((gap <= 1).mean())
    f["max_gap_to_nearest_training_sample"] = int(gap.max())
    res["random_stratified_80_20_holdout"] = f

    with open(OUT / "a3_results.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
