"""Run manifest, metrics archive, and human-readable report."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def make_run_dir(output_root: str | Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def git_commit_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    logger.info("wrote %s", path)


def write_report_md(results: dict, path: str | Path) -> None:
    """Compact markdown report of a single run."""
    lines = [
        f"# Run report: {results['config']['name']}",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- git commit: {results.get('git_commit')}",
        f"- feature set: {results['config']['feature_set']}",
        f"- split scheme: {results['config']['split']['scheme']}",
        f"- seconds per row (derived): {results['sampling']['derived_seconds_per_row']}",
        "",
        "## Label audit",
        "```json",
        json.dumps({k: v for k, v in results["label_audit"].items() if k != "note"}, indent=2),
        "```",
        "",
        "## Split composition",
        "```json",
        json.dumps(results["split"], indent=2),
        "```",
    ]
    for h_key, block in results.get("horizons", {}).items():
        lines += [
            "",
            f"## Horizon {h_key}",
            "",
            "### XGBoost (primary) — test metrics",
            "```json",
            json.dumps(block["xgb"]["test_metrics"], indent=2),
            "```",
            "### Bootstrap 95% CIs (moving-block)",
            "```json",
            json.dumps(block["xgb"].get("bootstrap_ci", {}), indent=2, default=str),
            "```",
        ]
        if "detection_delay" in block["xgb"]:
            lines += [
                "### Detection delay",
                "```json",
                json.dumps(block["xgb"]["detection_delay"], indent=2),
                "```",
            ]
        if "rf" in block:
            lines += [
                "### RandomForest baseline — test metrics",
                "```json",
                json.dumps(block["rf"]["test_metrics"], indent=2),
                "```",
            ]
        if "comparison" in block:
            lines += [
                "### XGB vs RF comparison",
                "```json",
                json.dumps(block["comparison"], indent=2),
                "```",
            ]
    if "risk_levels" in results:
        lines += [
            "",
            "## Six-level risk (secondary target)",
            "```json",
            json.dumps(results["risk_levels"], indent=2),
            "```",
        ]
    if "caveats" in results:
        lines += ["", "## Caveats (mandatory in any publication)"]
        lines += [f"- {c}" for c in results["caveats"]]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", path)


def export_tables(results: dict, out_dir: str | Path) -> None:
    """Publication summary tables: CSV (machine-readable) + LaTeX (booktabs,
    drop-in for thesis/paper). Eliminates hand-transcription of metrics."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = ["balanced_accuracy", "precision", "recall", "f1", "mcc",
                   "pr_auc", "roc_auc", "brier", "ece"]
    rows = []
    for h_key, block in results.get("horizons", {}).items():
        for model in ("xgb", "rf"):
            if model not in block:
                continue
            m = block[model]["test_metrics"]
            ci = block[model].get("bootstrap_ci", {})
            row = {"horizon": h_key, "model": model.upper()}
            for k in metric_keys:
                row[k] = m.get(k)
                c = ci.get(k)
                if isinstance(c, dict) and c.get("ci_low") is not None:
                    row[f"{k}_ci"] = f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]"
            rows.append(row)
    headline = pd.DataFrame(rows)
    headline.to_csv(out_dir / "table_headline_metrics.csv", index=False)
    _to_latex(headline, out_dir / "table_headline_metrics.tex",
              caption="Test-set performance (S2 held-out-transition split); "
                      "95\\% moving-block bootstrap CIs where defined.",
              label="tab:headline")

    for h_key, block in results.get("horizons", {}).items():
        prof = block.get("xgb", {}).get("anticipation_profile")
        if prof and isinstance(prof[0], dict) and "anticipation_rows" in prof[0]:
            df_a = pd.DataFrame(prof)
            df_a.to_csv(out_dir / f"table_anticipation_{h_key}.csv", index=False)
        sens = block.get("xgb", {}).get("delay_sensitivity")
        if sens:
            df_s = pd.DataFrame([r for r in sens if "threshold" in r])
            df_s.to_csv(out_dir / f"table_delay_sensitivity_{h_key}.csv", index=False)

    if "risk_levels" in results:
        r = results["risk_levels"]
        df_r = pd.DataFrame({
            "metric": ["quadratic_weighted_kappa", "level_mae", "within_one_accuracy"],
            "value": [r["quadratic_weighted_kappa"], r["level_mae"],
                      r["within_one_accuracy"]],
        })
        df_r.to_csv(out_dir / "table_risk_levels.csv", index=False)
        _to_latex(df_r, out_dir / "table_risk_levels.tex",
                  caption="Six-level risk classification (ordinal metrics).",
                  label="tab:risk")
    logger.info("summary tables written to %s", out_dir)


def _to_latex(df, path: Path, caption: str, label: str) -> None:
    body = df.to_latex(index=False, float_format=lambda v: f"{v:.3f}",
                       na_rep="--", escape=True)
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\resizebox{\\linewidth}{!}{%\n" + body + "}\n\\end{table}\n"
    )
    Path(path).write_text(tex, encoding="utf-8")


STANDARD_CAVEATS = [
    "Single filter lifecycle: all results describe one clogging episode on one "
    "rig; cross-filter generalization is untested and unclaimed.",
    "Labels are generated by the thesis's deterministic multi-criteria rule, "
    "not by independent ground truth (e.g. visual inspection); supervised "
    "results therefore measure how well the model reproduces/anticipates the "
    "rule from causal sensor features.",
    "Held-out-transition split (S2): the clogged training segment post-dates "
    "the test window, so S2 evaluates interpolation across a held-out "
    "transition, not pure forecasting.",
    "Detection delay is estimated from a single onset event; no confidence "
    "interval is possible (sensitivity across alarm rules reported instead).",
    "Block-bootstrap CIs quantify within-lifecycle sampling noise only.",
]
