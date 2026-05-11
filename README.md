# Filter Clogging Prediction System

**Version 2.1 | Hormozgan Gas Corporation**  
Developed by Ali Peykar

---

## Overview

A desktop application for intelligent monitoring and early prediction of gas filter clogging using DP meter data. Combines **Darcy permeability physics modeling** with **XGBoost machine learning** to predict when a filter will reach critical differential pressure thresholds.

---

## Features

- **Automatic data cleaning** — detects and removes startup artefacts and flow-rate spikes
- **Darcy permeability tracking** — computes K(t) at every time step and fits an exponential decay model
- **XGBoost model** — 12 engineered features (time lags, rolling stats, dp rate, resistance) with RandomizedSearchCV + TimeSeriesSplit cross-validation
- **SHAP feature importance** — identifies which operational parameters drive pressure drop
- **Dual clogging prediction** — physics model + empirical ΔP model for cross-validation
- **Remaining Useful Life (RUL)** — plots how much life is left at every point in the filter's lifecycle
- **GUI results viewer** — 8 analysis plots with navigation, export, and live console log

---

## Requirements

```
Python 3.9+
pandas
numpy
scipy
scikit-learn
xgboost
shap
matplotlib
FreeSimpleGUI
Pillow
openpyxl
```

Install all dependencies:

```bash
pip install pandas numpy scipy scikit-learn xgboost shap matplotlib FreeSimpleGUI Pillow openpyxl
```

---

## Usage

### Run the GUI application

```bash
python gui.py
```

1. Select your Excel data file (columns: `time` [s], `flowrate` [m³/h], `dp` [psi])
2. Set filter face area (m²) and filter depth (m)
3. Click **Run Analysis**
4. Results open automatically in the built-in viewer

### Run analysis programmatically

```python
from datakiller import FilterAnalysis

analyzer = FilterAnalysis(
    file_path="your_data.xlsx",
    filter_area=2.0,       # m²
    filter_length=0.02,    # m
    output_dir="results",
    show_plots=False,
)
analyzer.run_full_analysis(optimize_xgb=True)
```

---

## Output

All results are saved to the `results/` folder:

| File | Description |
|---|---|
| `01_raw_data.png` | Raw data overview |
| `02_data_quality.png` | Data quality check |
| `03_permeability.png` | Darcy permeability tracking |
| `04_xgboost_predictions.png` | XGBoost predictions vs actual |
| `05_residuals.png` | Residual diagnostics |
| `06_shap_summary.png` | SHAP feature importance |
| `07_shap_bar.png` | Feature importance bar chart |
| `08_clogging_prediction_rul.png` | Clogging prediction & RUL |
| `report.txt` | Full quantitative summary |
| `processed_data.csv` | Cleaned & feature-engineered data |

---

## Results on Real Data (Bringer.xlsx)

| Metric | Value |
|---|---|
| Total run duration | 192 min (3.2 h) |
| Total gas filtered | 21.08 m³ |
| Clogging threshold | 15 psi |
| Actual threshold crossing | 89.9 min |
| Physics model prediction | 107.2 min (~19% error) |
| XGBoost model prediction | 97.0 min (~8% error) |
| Peak ΔP recorded | 62.46 psi |
| Initial permeability K₀ | 2.79 × 10⁻¹⁴ m² |
| Decay time constant τ | 50 min |

---

## Build Executable

A PyInstaller spec file is included for building a standalone `.exe`:

```bash
pyinstaller FilterAnalysisApp.spec
```

---

## License

Developed for internal use by **Hormozgan Gas Corporation**.  
© Ali Peykar — All rights reserved.
