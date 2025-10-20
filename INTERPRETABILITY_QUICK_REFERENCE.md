# Model Interpretability - Quick Reference Card

## Installation

```bash
# Option 1: Install all dependencies
pip install -r requirements.txt

# Option 2: Install interpretability packages only
pip install shap lime plotly

# Option 3: Use installation script
# Windows:
install_interpretability.bat
# Linux/Mac:
chmod +x install_interpretability.sh && ./install_interpretability.sh
```

---

## Quick Start

```bash
# Run pipeline (interpretability included automatically)
python main.py your_data.csv

# Check output
ls plots/interpretability/
```

---

## Visualization Types Cheat Sheet

| Want to... | Use This | Function | Output |
|-----------|----------|----------|--------|
| See which features matter most globally | SHAP Summary | `plot_shap_summary()` | Bar plot with colors |
| Understand how a feature affects predictions | SHAP Dependence | `plot_shap_dependence()` | Scatter plot |
| Explain one specific prediction | SHAP Waterfall | `plot_shap_waterfall()` | Waterfall diagram |
| Interactively explore prediction | SHAP Force | `plot_shap_force()` | HTML visualization |
| Explain with any model type | LIME | `plot_lime_explanation()` | Bar chart + HTML |
| See average feature effects | Partial Dependence | `plot_partial_dependence()` | Line plots |
| Get robust feature importance | Permutation | `plot_permutation_importance()` | Bar chart with error bars |
| Generate everything at once | Dashboard | `create_interpretability_dashboard()` | All of the above |

---

## Code Snippets

### Use Everything (Recommended)

```python
from evaluation import create_interpretability_dashboard

results = create_interpretability_dashboard(
    model=model,
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    model_name='MyModel',
    save_path='plots/interpretability'
)
```

### SHAP Only (Fast)

```python
from evaluation import plot_shap_summary

explainer, shap_values = plot_shap_summary(
    model=model,
    X=X_test,
    feature_names=feature_names,
    model_name='MyModel'
)
```

### Explain Single Prediction

```python
from evaluation import explain_prediction

explanation = explain_prediction(
    model=model,
    X_sample=X_test[0],
    feature_names=feature_names,
    X_train=X_train,
    model_name='MyModel'
)
```

---

## Configuration Quick Reference

Edit `config.py`:

```python
CONFIG = {
    'interpretability': {
        'enabled': True,  # Set to False to disable
        'methods': ['shap', 'lime', 'pdp', 'permutation'],

        'shap': {
            'max_display': 20,  # Top N features
            'sample_indices': [0, 1, 2],  # Which samples to explain
        },

        'lime': {
            'num_features': 20,
            'sample_indices': [0, 1],
        },
    }
}
```

---

## Interpretation Guide

### SHAP Summary Plot
```
Feature_1  ●●●●●●●○○○ ──────────────────────
Feature_2  ●●●●○○○○○○ ──────────────────────
Feature_3  ●●●○○○○○○○ ──────────────────────
           -1        0        +1
           (decreases risk)  (increases risk)
```
- **Left side**: Features that decrease risk
- **Right side**: Features that increase risk
- **Red dots**: High feature values
- **Blue dots**: Low feature values

### SHAP Waterfall Plot
```
E[f(X)] = 0.3
+ Feature_1 = 0.5  ──────────> 0.8
+ Feature_2 = -0.2 ──────────> 0.6
+ Feature_3 = 0.1  ──────────> 0.7
= f(X) = 0.7
```
- Start from base value (E[f(X)])
- Each feature adds or subtracts
- End at final prediction f(X)

### LIME Explanation
```
Probability (Clogging) = 0.85

Top Features:
  Feature_1 > 50    ████████  +0.30
  Feature_2 < 20    ██        +0.10
  Feature_3 = 35    █         +0.05
```
- Green bars: Support predicted class
- Orange bars: Oppose predicted class
- Values show contribution magnitude

---

## Common Tasks

### Find Top 5 Important Features

```python
# After running dashboard
shap_importance = np.abs(shap_values).mean(axis=0)
top_5_idx = np.argsort(shap_importance)[-5:][::-1]
for idx in top_5_idx:
    print(feature_names[idx])
```

### Analyze Specific Feature

```python
from evaluation import plot_shap_dependence

plot_shap_dependence(
    shap_values, X_test, feature_names,
    feature_idx='differential_pressure',
    model_name='MyModel'
)
```

### Debug Misclassification

```python
# Find misclassified samples
errors = np.where(y_pred != y_test)[0]

# Explain first error
from evaluation import explain_prediction
explanation = explain_prediction(
    model, X_test[errors[0]], feature_names,
    X_train=X_train, sample_idx=errors[0]
)
```

---

## Troubleshooting

### "SHAP not available"
```bash
pip install shap
```

### "LIME not available"
```bash
pip install lime
```

### Memory Error with SHAP
```python
# Subsample data
sample_size = min(1000, len(X_test))
X_sample = X_test[:sample_size]
plot_shap_summary(model, X_sample, feature_names)
```

### SHAP is Slow
```python
# Use TreeExplainer (much faster for tree models)
import shap
explainer = shap.TreeExplainer(model)  # Fast!
# vs
explainer = shap.KernelExplainer(...)  # Slow
```

### Model Not Supported
```python
# Use model-agnostic methods
plot_lime_explanation(...)         # Works with any model
plot_permutation_importance(...)   # Works with any model
plot_partial_dependence(...)       # Works with any model
```

---

## File Locations

After running the pipeline:

```
plots/
├── interpretability/
│   ├── model_shap_summary.png
│   ├── model_shap_dependence_*.png
│   ├── model_shap_waterfall_sample_*.png
│   ├── model_shap_force_sample_*.html  ← Open in browser
│   ├── model_shap_decision.png
│   ├── model_lime_sample_*.png
│   ├── model_lime_sample_*.html        ← Open in browser
│   ├── model_partial_dependence.png
│   ├── model_permutation_importance.png
│   └── model_treebased_feature_importance.png
```

---

## API Quick Reference

```python
# All in evaluation.py
from evaluation import (
    plot_shap_summary,              # SHAP global importance
    plot_shap_dependence,           # SHAP feature effect
    plot_shap_waterfall,            # SHAP single prediction
    plot_shap_force,                # SHAP interactive
    plot_shap_decision,             # SHAP decision paths
    plot_lime_explanation,          # LIME local explanation
    plot_partial_dependence,        # PDP feature effects
    plot_permutation_importance,    # Permutation importance
    create_interpretability_dashboard,  # All-in-one
    explain_prediction              # Single prediction multi-method
)
```

---

## Best Practices

1. ✅ **Start with dashboard**: `create_interpretability_dashboard()` gives overview
2. ✅ **Check consistency**: Compare SHAP, permutation, and tree importance
3. ✅ **Investigate top features**: Use dependence plots for top 3-5 features
4. ✅ **Explain errors**: Use waterfall/LIME on misclassifications
5. ✅ **Validate domain knowledge**: Check if model uses expected features
6. ✅ **Save artifacts**: Keep interpretability plots with trained models

---

## Performance Tips

- 🚀 **Small dataset (<1K)**: Use all methods
- 🚀 **Medium dataset (1K-10K)**: SHAP + PDP + Permutation
- 🚀 **Large dataset (>10K)**: Subsample for SHAP/LIME
- 🚀 **Tree models**: Use SHAP TreeExplainer (very fast)
- 🚀 **Other models**: Use LIME + Permutation (model-agnostic)

---

## Further Reading

- **Full Documentation**: [INTERPRETABILITY_GUIDE.md](INTERPRETABILITY_GUIDE.md)
- **Implementation Details**: [VISUALIZATION_ENHANCEMENT_SUMMARY.md](VISUALIZATION_ENHANCEMENT_SUMMARY.md)
- **SHAP Documentation**: https://shap.readthedocs.io/
- **LIME Documentation**: https://lime-ml.readthedocs.io/
- **Interpretable ML Book**: https://christophm.github.io/interpretable-ml-book/

---

## Quick Checklist

Before deploying model:
- [ ] Generated SHAP summary (global importance)
- [ ] Checked top 3-5 features make domain sense
- [ ] Analyzed feature dependence plots
- [ ] Explained sample predictions (waterfall/LIME)
- [ ] Compared importance methods (SHAP vs permutation)
- [ ] Validated model uses expected relationships (PDP)
- [ ] Documented key findings
- [ ] Saved all interpretability plots

---

**Need More Help?** See [INTERPRETABILITY_GUIDE.md](INTERPRETABILITY_GUIDE.md) for comprehensive documentation with examples.
