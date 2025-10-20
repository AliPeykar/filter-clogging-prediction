# Model Interpretability Guide

## Overview

The visualization module has been significantly enhanced with advanced model interpretability features. This guide explains how to use SHAP, LIME, Partial Dependence Plots, and other interpretability methods to better understand your filter clogging prediction models.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Available Methods](#available-methods)
3. [SHAP Explanations](#shap-explanations)
4. [LIME Explanations](#lime-explanations)
5. [Partial Dependence Plots](#partial-dependence-plots)
6. [Permutation Importance](#permutation-importance)
7. [Configuration](#configuration)
8. [API Reference](#api-reference)

---

## Quick Start

### Installation

Install the required interpretability libraries:

```bash
pip install shap lime
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Basic Usage

The interpretability dashboard is automatically generated when you run the main pipeline:

```bash
python main.py your_data.csv
```

All interpretability visualizations will be saved to `plots/interpretability/`.

---

## Available Methods

The enhanced visualization module provides the following interpretability methods:

| Method | Type | Description | Best For |
|--------|------|-------------|----------|
| **SHAP Summary** | Global | Shows overall feature importance across all predictions | Understanding which features matter most |
| **SHAP Dependence** | Global | Shows how feature values affect predictions | Understanding feature relationships |
| **SHAP Waterfall** | Local | Explains individual predictions step-by-step | Debugging specific predictions |
| **SHAP Force Plot** | Local | Interactive visualization of prediction forces | Exploring individual cases |
| **SHAP Decision Plot** | Global/Local | Shows decision paths for multiple samples | Comparing prediction patterns |
| **LIME** | Local | Model-agnostic local explanations | Understanding any model type |
| **Partial Dependence** | Global | Marginal effect of features | Understanding feature impact |
| **Permutation Importance** | Global | Model-agnostic feature importance | Robust importance ranking |

---

## SHAP Explanations

**SHAP (SHapley Additive exPlanations)** uses game theory to explain predictions by computing each feature's contribution.

### SHAP Summary Plot

Shows global feature importance with direction of effect:

```python
from evaluation import plot_shap_summary

explainer, shap_values = plot_shap_summary(
    model=trained_model,
    X=X_test,
    feature_names=feature_names,
    model_name='MyModel',
    max_display=20,
    save_path='plots'
)
```

**Output:** `plots/mymodel_shap_summary.png`

**Interpretation:**
- Features are ranked by importance (top to bottom)
- Color shows feature value (red = high, blue = low)
- X-axis shows impact on prediction (positive = increases risk)

### SHAP Dependence Plot

Shows how a specific feature affects predictions:

```python
from evaluation import plot_shap_dependence

plot_shap_dependence(
    shap_values=shap_values,
    X=X_test,
    feature_names=feature_names,
    feature_idx='differential_pressure',  # or feature index
    interaction_idx='auto',  # automatically detects interactions
    model_name='MyModel',
    save_path='plots'
)
```

**Output:** `plots/mymodel_shap_dependence_differential_pressure.png`

**Interpretation:**
- X-axis: feature value
- Y-axis: SHAP value (impact on prediction)
- Color: interaction feature (if detected)

### SHAP Waterfall Plot

Explains a single prediction:

```python
from evaluation import plot_shap_waterfall

plot_shap_waterfall(
    explainer=explainer,
    shap_values=shap_values,
    X=X_test,
    feature_names=feature_names,
    sample_idx=0,  # which sample to explain
    model_name='MyModel',
    save_path='plots'
)
```

**Output:** `plots/mymodel_shap_waterfall_sample_0.png`

**Interpretation:**
- Shows how prediction builds up from base value
- Red bars push prediction higher (toward clogging)
- Blue bars push prediction lower (toward healthy)

### SHAP Force Plot

Interactive HTML visualization:

```python
from evaluation import plot_shap_force

plot_shap_force(
    explainer=explainer,
    shap_values=shap_values,
    X=X_test,
    feature_names=feature_names,
    sample_idx=0,
    model_name='MyModel',
    save_path='plots'
)
```

**Output:** `plots/mymodel_shap_force_sample_0.html` (open in browser)

**Interpretation:**
- Interactive visualization showing forces pushing prediction
- Red forces increase risk, blue forces decrease risk
- Hover over features to see exact values

---

## LIME Explanations

**LIME (Local Interpretable Model-agnostic Explanations)** creates local linear approximations around specific predictions.

### Basic LIME Usage

```python
from evaluation import plot_lime_explanation

lime_exp = plot_lime_explanation(
    model=trained_model,
    X_train=X_train,  # background data
    X_test=X_test,
    feature_names=feature_names,
    sample_idx=0,
    class_names=['Healthy', 'Clogging'],
    model_name='MyModel',
    save_path='plots',
    num_features=20
)
```

**Output:**
- `plots/mymodel_lime_sample_0.png` (static image)
- `plots/mymodel_lime_sample_0.html` (interactive HTML)

**Interpretation:**
- Shows top features contributing to prediction
- Green bars support predicted class
- Orange bars oppose predicted class
- Works with any model type (not just tree-based)

---

## Partial Dependence Plots

**PDP** shows the marginal effect of features on predictions, averaging over other features.

### Basic PDP Usage

```python
from evaluation import plot_partial_dependence

plot_partial_dependence(
    model=trained_model,
    X=X_test,
    feature_names=feature_names,
    features_to_plot=[0, 1, 2, 3],  # or feature names
    model_name='MyModel',
    save_path='plots',
    grid_resolution=50
)
```

**Output:** `plots/mymodel_partial_dependence.png`

**Interpretation:**
- X-axis: feature value range
- Y-axis: predicted probability
- Shows average relationship between feature and prediction
- Useful for understanding non-linear relationships

---

## Permutation Importance

**Permutation Importance** measures importance by randomly shuffling each feature and measuring performance drop.

### Basic Usage

```python
from evaluation import plot_permutation_importance

perm_result = plot_permutation_importance(
    model=trained_model,
    X=X_test,
    y=y_test,
    feature_names=feature_names,
    model_name='MyModel',
    n_repeats=10,
    top_n=20,
    save_path='plots'
)
```

**Output:** `plots/mymodel_permutation_importance.png`

**Interpretation:**
- Features ranked by importance
- Error bars show variability across permutations
- Model-agnostic (works with any model)
- More robust than tree-based feature importance

---

## Configuration

Configure interpretability settings in `config.py`:

```python
CONFIG = {
    # ... other config ...

    'interpretability': {
        'enabled': True,  # Enable/disable interpretability analysis
        'methods': ['shap', 'lime', 'pdp', 'permutation'],

        'shap': {
            'max_display': 20,  # Top N features to show
            'sample_indices': [0, 1, 2],  # Samples to explain in detail
            'num_samples_for_decision_plot': 10,
        },

        'lime': {
            'num_features': 20,  # Features to include in explanation
            'sample_indices': [0, 1],  # Samples to explain with LIME
        },

        'pdp': {
            'grid_resolution': 50,  # Number of grid points
            'num_features': 4,  # Number of features for PDP
        },

        'permutation': {
            'n_repeats': 10,  # Permutation repetitions
            'top_n': 20,  # Top N features to display
        },
    },
}
```

---

## API Reference

### Comprehensive Dashboard

Create all interpretability visualizations at once:

```python
from evaluation import create_interpretability_dashboard

results = create_interpretability_dashboard(
    model=trained_model,
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    model_name='MyModel',
    save_path='plots/interpretability',
    sample_indices=[0, 1, 2]
)
```

**Returns:**
- `results['shap_explainer']`: SHAP explainer object
- `results['shap_values']`: SHAP values array
- `results['permutation_importance']`: Permutation importance results

### Single Prediction Explanation

Explain one prediction using multiple methods:

```python
from evaluation import explain_prediction

explanation = explain_prediction(
    model=trained_model,
    X_sample=X_test[0],  # single sample
    feature_names=feature_names,
    explainer=shap_explainer,  # optional, will create if None
    shap_values=shap_values,   # optional, will compute if None
    X_train=X_train,           # needed for LIME
    model_name='MyModel',
    save_path='plots',
    sample_idx=0
)
```

**Returns:**
- `explanation['prediction']`: predicted class
- `explanation['probability']`: prediction probability
- `explanation['shap_values']`: SHAP values for this sample
- `explanation['lime']`: LIME explanation object
- `explanation['feature_importances']`: tree-based importances (if available)

---

## Advanced Usage

### Custom SHAP Analysis

For custom SHAP analysis workflows:

```python
import shap
from evaluation import plot_shap_summary, plot_shap_dependence

# 1. Generate SHAP values
explainer, shap_values = plot_shap_summary(
    model, X_test, feature_names, 'MyModel', save_path='plots'
)

# 2. Analyze top 3 features in detail
if hasattr(model, 'feature_importances_'):
    top_features = np.argsort(model.feature_importances_)[-3:][::-1]

    for feat_idx in top_features:
        plot_shap_dependence(
            shap_values, X_test, feature_names,
            feature_idx=feat_idx,
            model_name='MyModel',
            save_path='plots'
        )
```

### Comparing Multiple Models

Compare interpretability across models:

```python
models = {
    'RandomForest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgbm_model
}

for name, model in models.items():
    create_interpretability_dashboard(
        model, X_train, X_test, y_test,
        feature_names, model_name=name,
        save_path=f'plots/interpretability/{name.lower()}'
    )
```

---

## Troubleshooting

### SHAP Installation Issues

If you encounter SHAP installation problems:

```bash
# Try installing with specific version
pip install shap==0.41.0

# For Windows users with C++ compiler issues:
pip install shap --no-build-isolation
```

### Memory Issues with SHAP

For large datasets, subsample before computing SHAP values:

```python
# Sample 1000 points for SHAP analysis
sample_size = min(1000, len(X_test))
sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
X_test_sample = X_test[sample_indices]

explainer, shap_values = plot_shap_summary(
    model, X_test_sample, feature_names, 'MyModel'
)
```

### LIME Performance

LIME can be slow for complex models. Reduce samples:

```python
# Explain only 2-3 most interesting samples
plot_lime_explanation(
    model, X_train, X_test, feature_names,
    sample_idx=0,  # Most uncertain prediction
    model_name='MyModel'
)
```

---

## Best Practices

1. **Start with SHAP Summary**: Get overall feature importance first
2. **Investigate Top Features**: Use dependence plots for top 3-5 features
3. **Explain Edge Cases**: Use waterfall/force plots for misclassifications
4. **Cross-validate with Multiple Methods**: Compare SHAP, LIME, and permutation importance
5. **Consider Interactions**: Use SHAP dependence plots with `interaction_idx='auto'`
6. **Document Findings**: Save interpretability plots alongside model results

---

## Example Workflow

```python
from predictor import FilterCloggingPredictor
from evaluation import create_interpretability_dashboard

# 1. Train model
predictor = FilterCloggingPredictor()
predictor.fit(X_train, y_train, ...)

# 2. Get base model (unwrap if calibrated)
model = predictor.classification_models['rf']
if hasattr(model, 'calibrated_classifiers_'):
    base_model = model.calibrated_classifiers_[0].estimator
else:
    base_model = model

# 3. Generate comprehensive interpretability analysis
results = create_interpretability_dashboard(
    model=base_model,
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=predictor.feature_names,
    model_name='FilterClogging_RF',
    save_path='plots/interpretability'
)

# 4. Analyze results
print(f"Top 5 most important features (SHAP):")
shap_importance = np.abs(results['shap_values']).mean(axis=0)
top_5_idx = np.argsort(shap_importance)[-5:][::-1]
for idx in top_5_idx:
    print(f"  {predictor.feature_names[idx]}: {shap_importance[idx]:.4f}")
```

---

## References

- **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
- **LIME**: Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining predictions of any classifier. KDD.
- **Partial Dependence**: Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.

---

## Additional Resources

- SHAP Documentation: https://shap.readthedocs.io/
- LIME Documentation: https://lime-ml.readthedocs.io/
- Interpretable ML Book: https://christophm.github.io/interpretable-ml-book/

---

## Summary

The enhanced visualization module provides state-of-the-art model interpretability tools:

✅ **Global Understanding**: SHAP summary, permutation importance, PDP
✅ **Local Explanations**: SHAP waterfall/force, LIME
✅ **Interactive Visualizations**: HTML force plots, decision plots
✅ **Model-Agnostic**: Works with any scikit-learn compatible model
✅ **Production-Ready**: Integrated into main pipeline with error handling

For questions or issues, refer to the main README or open an issue on the project repository.
