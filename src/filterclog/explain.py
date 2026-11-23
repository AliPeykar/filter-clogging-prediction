"""SHAP interpretation, persisted (legacy computed SHAP but never saved it)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_and_persist_shap(
    model,
    X: pd.DataFrame,
    out_dir: str | Path,
    max_display: int = 15,
    top_dependence: int = 3,
) -> pd.DataFrame:
    """TreeExplainer SHAP values on the given frame; persists raw values,
    a mean-|SHAP| ranking CSV, a beeswarm plot, and dependence plots for the
    top features. Returns the ranking."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older API returns [neg, pos]
        values = values[1]
    np.save(out_dir / "shap_values.npy", values)
    X.to_csv(out_dir / "X_explained.csv.gz", index=True, compression="gzip")

    ranking = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking.to_csv(out_dir / "shap_ranking.csv", index=False)

    plt.figure()
    shap.summary_plot(values, X, max_display=max_display, show=False)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(out_dir / f"shap_beeswarm.{ext}", dpi=300, bbox_inches="tight")
    plt.close("all")

    for feat in ranking["feature"].head(top_dependence):
        plt.figure()
        shap.dependence_plot(feat, values, X, interaction_index=None, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_dependence_{feat}.png", dpi=300, bbox_inches="tight")
        plt.close("all")

    logger.info("SHAP persisted to %s (top: %s)", out_dir, ranking['feature'].head(5).tolist())
    return ranking
