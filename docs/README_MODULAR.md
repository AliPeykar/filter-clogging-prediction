# Filter Clogging Prediction System (Modular v2.0)

A comprehensive, modular system for predicting filter clogging using multiple machine learning approaches.

## Overview

A **production-ready machine learning system** for predicting filter clogging events using time-series sensor data (differential pressure and flow rate). This modular architecture combines anomaly detection, classification, survival analysis, and regression models for robust prediction.

### Key Features:
- ✅ **Modular Architecture**: Clean separation of concerns for easy maintenance and extension
- ✅ **Anomaly Detection**: Unsupervised learning on healthy data (Isolation Forest, LOF, One-Class SVM)
- ✅ **Classification Models**: Random Forest, XGBoost, LightGBM with cost-sensitive learning
- ✅ **Survival Analysis**: Cox Proportional Hazards and Random Survival Forest
- ✅ **Regression Models**: Continuous time-to-clog prediction
- ✅ **Smart Data Splitting**: Timeline-aware splits for severely imbalanced data
- ✅ **Temporal Weighting**: Samples closer to clogging get higher weights
- ✅ **Cost-Based Optimization**: Minimize operational costs (missed clogs vs false alarms)

## Project Structure

```
final3/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration management
├── main.py                     # Main execution script
├── predictor.py                # FilterCloggingPredictor class
├── data_processing.py          # Data loading and preprocessing
├── feature_engineering.py      # Feature creation functions
├── utils.py                    # Utility functions
├── anomaly_detection.py        # Anomaly detection module
├── survival_models.py          # Cox and RSF models
├── regression_models.py        # Regression predictors
├── evaluation.py               # Evaluation metrics and plots
├── optimization.py             # Hyperparameter optimization
└── README.md                   # Documentation
```

## Quick Start

### Installation

```bash
# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn
pip install xgboost lightgbm optuna
pip install lifelines scikit-survival  # For survival analysis
```

### Basic Usage

```python
from main import main

# Run complete pipeline
predictor, metrics = main(
    data_filepath='filter_data.csv',
    clog_index=8940  # Optional: auto-detected if not provided
)
```

### Command Line Usage

```bash
python main.py filter_data.csv 8940
```

## Module Documentation

### 1. Configuration (`config.py`)

Central configuration for all model parameters:

```python
from config import CONFIG, update_config

# View current config
print(CONFIG)

# Update configuration
update_config({
    'forecast_horizon_steps': 150,
    'cost_fn': 200,  # Increase cost of false negatives
})
```

**Key configuration options:**
- `forecast_horizon_steps`: Prediction horizon (default: 120)
- `cost_fn`: Cost of false negative (default: 100)
- `cost_fp`: Cost of false positive (default: 1)
- `models_to_use`: List of classification models
- `use_survival`: Enable survival analysis
- `use_regression`: Enable regression models

### 2. Data Processing (`data_processing.py`)

Functions for loading and preparing data:

```python
from data_processing import (
    load_and_prepare_data,
    compute_target_labels,
    prepare_features_and_targets
)

# Load data
df = load_and_prepare_data('filter_data.csv')

# Compute targets
df = compute_target_labels(df)

# Prepare for modeling
X, y_class, y_time, y_duration, y_event, feature_names = \
    prepare_features_and_targets(df)
```

### 3. Feature Engineering (`feature_engineering.py`)

Automated feature creation:

```python
from feature_engineering import engineer_all_features

# Create all features automatically
df = engineer_all_features(df, config=CONFIG)
```

**Features created:**
- Rolling statistics (mean, std, min, max)
- Lag features
- Exponentially weighted moving averages
- Rate of change and acceleration
- Degradation index
- Interaction features

### 4. Main Predictor (`predictor.py`)

Unified interface for all models:

```python
from predictor import FilterCloggingPredictor

# Initialize
predictor = FilterCloggingPredictor(config=CONFIG)

# Train
predictor.fit(
    X_train, y_class, y_time, y_duration, y_event,
    X_val=X_val, y_class_val=y_val_class,
    healthy_idx=healthy_idx
)

# Predict
y_pred = predictor.predict(X_test)
y_proba = predictor.predict_proba(X_test)
time_to_clog = predictor.predict_time_to_clog(X_test)
risk_scores = predictor.predict_risk_scores(X_test)
```

### 5. Anomaly Detection (`anomaly_detection.py`)

Unsupervised learning for degradation detection:

```python
from anomaly_detection import AnomalyDetectionModule

# Initialize
anomaly_detector = AnomalyDetectionModule(config=CONFIG)

# Train on healthy data only
anomaly_detector.fit(X_healthy)

# Detect anomalies
results = anomaly_detector.predict_anomaly_scores(X_test)
print(results['ensemble_score'])  # 0-1 anomaly scores
print(results['is_anomaly'])       # Binary predictions
```

### 6. Survival Models (`survival_models.py`)

Cox and Random Survival Forest:

```python
from survival_models import CoxPredictionModel, RandomSurvivalForestModel

# Cox model
cox = CoxPredictionModel()
cox.fit(X_train, duration_train, event_train, feature_names)
risk_scores = cox.predict_risk(X_test)

# Random Survival Forest
rsf = RandomSurvivalForestModel(n_estimators=100)
rsf.fit(X_train, duration_train, event_train)
risk_scores = rsf.predict_risk(X_test)
```

### 7. Regression Models (`regression_models.py`)

Continuous time-to-clog prediction:

```python
from regression_models import RegressionPredictor

# Single model
reg = RegressionPredictor(model_type='rf')
reg.fit(X_train, y_time, censored_mask=censored_mask)

# Predictions
time_predictions = reg.predict_time_to_clog(X_test)
risk_scores = reg.predict_risk_score(X_test, horizon=120)
```

### 8. Evaluation (`evaluation.py`)

Comprehensive evaluation and visualization:

```python
from evaluation import (
    evaluate_classification_model,
    plot_confusion_matrix,
    plot_roc_curve
)

# Evaluate
metrics = evaluate_classification_model(
    y_true, y_pred, y_proba,
    model_name='My Model'
)

# Plot results
plot_confusion_matrix(metrics['confusion_matrix'], model_name='My Model')
plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'])
```

### 9. Utilities (`utils.py`)

Helper functions:

```python
from utils import (
    time_series_split_imbalanced,
    optimize_threshold_by_cost,
    compute_temporal_weights
)

# Smart data splitting for imbalanced time-series
train_idx, val_idx, test_idx, healthy_idx = \
    time_series_split_imbalanced(df, clog_index=8940)

# Optimize decision threshold
optimal_threshold, cost, results = \
    optimize_threshold_by_cost(y_true, y_proba, cost_fn=100, cost_fp=1)

# Compute temporal weights
weights = compute_temporal_weights(time_to_clog)
```

## Advanced Usage

### Custom Configuration

```python
from config import CONFIG

# Customize for your use case
CONFIG['forecast_horizon_steps'] = 200
CONFIG['cost_fn'] = 500  # Higher cost for missed clogs
CONFIG['anomaly_detection']['contamination'] = 0.02
CONFIG['models_to_use'] = ['rf', 'xgb']  # Skip LightGBM
```

### Hyperparameter Optimization

```python
from optimization import optimize_hyperparameters

best_params, study = optimize_hyperparameters(
    model_type='rf',
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    n_trials=100,
    timeout=3600
)
```

### Using Individual Components

```python
# Use only anomaly detection
from anomaly_detection import AnomalyDetectionModule

detector = AnomalyDetectionModule()
detector.fit(X_healthy)
scores = detector.predict_anomaly_scores(X_test)

# Use only regression
from regression_models import EnsembleRegressionPredictor

ensemble = EnsembleRegressionPredictor(models=['rf', 'xgb'])
ensemble.fit(X_train, y_time)
predictions = ensemble.predict_time_to_clog(X_test)
```

## Key Design Decisions

### 1. No SMOTE
SMOTE is inappropriate for time-series data. Instead:
- Cost-sensitive learning with `class_weight='balanced'`
- Temporal weighting (exponential decay)
- Anomaly detection on healthy data

### 2. Extended Forecast Horizon
Changed from 25 to 120 steps to:
- Improve class balance
- Provide earlier warnings
- Better match operational needs

### 3. Multiple Model Types
Combining classification, survival, and regression:
- **Classification**: Binary risk prediction
- **Survival**: Handles censored data naturally
- **Regression**: Continuous time-to-clog estimation

### 4. Timeline-Aware Splitting
For severely imbalanced data (clogging at 8940/9000):
- Healthy set: 0-85% of clog point
- Training: 0-70%
- Validation: 70-95%
- Test: 95-100%

## Performance Expectations

With proper configuration:
- **Early Detection**: 300-500 steps before clogging
- **False Positive Rate**: <5%
- **Recall**: >90% (catches most clogs)
- **Operational Cost**: Minimized through threshold optimization

## Troubleshooting

### Cox Model Convergence Issues
The Cox model includes automatic fallback:
1. Tries multiple penalty levels (0.1, 1.0, 5.0, 10.0)
2. Removes highly correlated features
3. Gracefully skips if all attempts fail

### Imbalanced Data
For extreme imbalance:
1. Enable anomaly detection: `CONFIG['anomaly_detection']['enabled'] = True`
2. Use temporal weighting: `CONFIG['use_temporal_weighting'] = True`
3. Adjust cost ratio: `CONFIG['cost_fn'] = 100`, `CONFIG['cost_fp'] = 1`

### Memory Issues
For large datasets:
1. Reduce rolling window sizes
2. Use fewer lag features
3. Reduce number of Optuna trials

## License

MIT License - Feel free to use and modify for your needs.

## Contributing

Contributions welcome! Key areas for improvement:
- Additional model types
- More feature engineering methods
- Enhanced visualization
- Performance optimizations
