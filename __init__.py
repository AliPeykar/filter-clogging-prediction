"""
Filter Clogging Prediction System - Modular Package

A comprehensive system for predicting filter clogging using:
- Anomaly detection
- Classification models
- Survival analysis
- Regression models
"""

__version__ = '2.0.0'
__author__ = 'Filter Clogging Prediction Team'

# Import main components for easy access
from .config import CONFIG, RANDOM_SEED, get_config, update_config
from .predictor import FilterCloggingPredictor
from .anomaly_detection import AnomalyDetectionModule
from .survival_models import CoxPredictionModel, RandomSurvivalForestModel
from .regression_models import RegressionPredictor, EnsembleRegressionPredictor

# Import data processing
from .data_processing import (
    load_and_prepare_data,
    compute_target_labels,
    create_ratio_features,
    prepare_features_and_targets
)

# Import feature engineering
from .feature_engineering import engineer_all_features

# Import utilities
from .utils import (
    time_series_split_imbalanced,
    compute_temporal_weights,
    optimize_threshold_by_cost
)

# Import evaluation
from .evaluation import (
    evaluate_classification_model,
    evaluate_regression_model,
    plot_confusion_matrix,
    plot_roc_curve
)

__all__ = [
    # Main predictor
    'FilterCloggingPredictor',

    # Configuration
    'CONFIG',
    'RANDOM_SEED',
    'get_config',
    'update_config',

    # Model components
    'AnomalyDetectionModule',
    'CoxPredictionModel',
    'RandomSurvivalForestModel',
    'RegressionPredictor',
    'EnsembleRegressionPredictor',

    # Data processing
    'load_and_prepare_data',
    'compute_target_labels',
    'create_ratio_features',
    'prepare_features_and_targets',

    # Feature engineering
    'engineer_all_features',

    # Utilities
    'time_series_split_imbalanced',
    'compute_temporal_weights',
    'optimize_threshold_by_cost',

    # Evaluation
    'evaluate_classification_model',
    'evaluate_regression_model',
    'plot_confusion_matrix',
    'plot_roc_curve',
]
