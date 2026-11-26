# Leakage-aware validation of machine-learning clogging detection for a gas filter

Data, code and analysis scripts for the article

> A. Peykar, S. Movahedirad. *From perfect scores to credible evaluation:
> leakage-aware validation of machine-learning clogging detection for a
> natural-gas filter.* Submitted to *Proc. IMechE Part O: Journal of Risk and
> Reliability* (under review).

The article audits an earlier pipeline of ours that reported accuracy, F1 and
ROC-AUC of 1.000 on a laboratory clogging run. It shows where those scores came
from (a single-class test set, predictors derived from the labeling rule, and
shuffled resampling) and gives a corrected pipeline with signal-only
baselines. Everything needed to reproduce the reported numbers, tables and
figures is in this repository.

## Contents

| Path | What it is |
|---|---|
| `Comprehensive_Filter_Analysis.xlsx` | Analysed data set: 11,519 samples of flow rate Q (m³/h) and filter differential pressure ΔP (kPa) from one clogging run, with the labeling tool's EMA, CUSUM, score and label columns |
| `raw_logs/` | Raw Arduino logs (PuTTY captures, about 1 Hz) of 22 April 2025 (`data2*.csv`, nine files; the data set joins seven of them); `newdata/data.csv` and `newdata/data.py` are the concatenation step |
| `labeling_tool/filter-analyzer-complete.html` | Source of the browser tool that produced the score and label columns |
| `src/filterclog/`, `run_experiment.py`, `configs/`, `tests/` | Corrected, leakage-audited pipeline (signal-only features, purged held-out-transition split, XGBoost and Random Forest, calibration, block bootstrap) with unit tests |
| `docs/` | Modification log, verification checklist and physics notes for the corrected pipeline |
| `legacy/` | The earlier pipeline (reported scores of 1.000), archived unmodified with its reports and plots; trained model files are not included |
| `paper/01_reproduction/runs/` | Archived outputs of the four configurations as reported in the article |
| `paper/03_analysis/` | Analysis scripts A1–A12, sensitivity configurations and runs, figure script and their outputs (`out/`) |

## Data notes

* Units come from the log headers: ΔP in kPa, Q in m³/h, time in s. The `time`
  column of the data set is a row counter; durations in the article are given
  in samples.
* Q was computed on the Arduino from the orifice-plate ΔP with the orifice
  equation (C = 0.60, ε = 1, fixed temperature, operator-entered upstream
  gauge pressure); no calibration constant enters the conversion. ΔP was
  converted with the nominal MPX5100 transfer function.
* The data set joins seven logs. Samples 8,191–8,339 repeat samples
  6,793–6,941; 874 log rows were removed; 10 samples match no log row. Script
  A7 maps every sample to its log row and reports all of this.
* The labels come from a deterministic rule on the same ΔP signal (score ≥ 11,
  equivalent to ΔP > 21.776 kPa and sample index > 7,500 apart from a
  30-sample look-ahead); there is no independent ground truth.

## Reproducing the results

Python 3.12–3.13; package versions are pinned in `requirements.txt`.

```bash
pip install -r requirements.txt
python -m pytest tests/
python run_experiment.py --config configs/main.yaml
python run_experiment.py --config configs/ablation_thesis_features.yaml
python run_experiment.py --config configs/early_warning.yaml
python run_experiment.py --config configs/strict_temporal.yaml --reuse-run outputs/main_<stamp>
```

The analyses run from the repository root, for example
`python paper/03_analysis/a7_raw_logs_and_label_tool.py`; `make_figures.py`
redraws the article figures from `paper/03_analysis/out/` and the archived
runs. Re-runs on other machines can differ in the third decimal; the article
reports the archived results.

Analysis A11 uses a second, public data set that is not redistributed here:
`Extruder Parameters.zip` from Zenodo, https://doi.org/10.5281/zenodo.21987832
(Ferreira et al., Data in Brief 69 (2026) 113254; CC BY 4.0). Extract it to
`paper/08_extrusion/extracted/` before running
`python paper/03_analysis/a11_extrusion_second_dataset.py`.

## Licence

Code: MIT (see `LICENSE`). Data, logs and documentation: CC BY 4.0.

## Contact

Salman Movahedirad, School of Chemical, Petroleum and Gas Engineering, Iran
University of Science and Technology, Tehran, Iran (movahedirad@iust.ac.ir).
