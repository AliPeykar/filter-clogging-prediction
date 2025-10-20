"""
Configuration management for filter clogging prediction system.
"""

import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42

# Main configuration dictionary
CONFIG = {
    # Feature engineering
    'rolling_window_sizes': [5, 10, 20, 50],
    'lag_features': [1, 5, 10, 20, 50],
    'ewm_spans': [10, 30, 50],

    # Prediction parameters
    'forecast_horizon_steps': 120,
    'risk_thresholds': {
        'T_high': 40,
        'T_low': 100,
    },

    # Cost-sensitive learning
    'cost_fn': 100,
    'cost_fp': 1,
    'use_temporal_weighting': True,
    'weight_decay_factor': 0.2,

    # Model parameters
    'models_to_use': ['rf', 'xgb', 'lgbm'],
    'use_survival': True,
    'survival_models': ['cox', 'rsf'],
    'use_regression': True,
    'regression_models': ['rf', 'xgb'],

    # Anomaly detection
    'anomaly_detection': {
        'enabled': True,
        'healthy_data_fraction': 0.85,
        'contamination': 0.01,
        'methods': ['isolation_forest', 'lof', 'ocsvm'],
        'ensemble_weights': [0.5, 0.3, 0.2],
        'n_estimators': 200,
        'lof_neighbors': 20,
    },
    'anomaly_blend_weight': 0.15,  # Weight for anomaly scores in final prediction (15% anomaly, 85% classification)

    # Hyperparameter optimization
    'optuna': {
        'n_trials': 50,
        'timeout': 3600,
        'n_jobs': -1,
    },

    # Visualization
    'viz': {
        'figsize': (14, 8),
        'dpi': 100,
        'style': 'seaborn-v0_8-darkgrid',
    },

    # Model Interpretability
    'interpretability': {
        'enabled': True,
        'methods': ['shap', 'lime', 'pdp', 'permutation'],
        'shap': {
            'max_display': 20,
            'sample_indices': [0, 1, 2],  # Indices of samples to explain in detail
            'num_samples_for_decision_plot': 10,
        },
        'lime': {
            'num_features': 20,
            'sample_indices': [0, 1],  # Indices of samples to explain with LIME
        },
        'pdp': {
            'grid_resolution': 50,
            'num_features': 4,  # Number of top features for PDP
        },
        'permutation': {
            'n_repeats': 10,
            'top_n': 20,
        },
    },

    # 4-Level Risk Scoring System
    'risk_levels': {
        'enabled': True,
        'num_levels': 4,
        'thresholds': {
            'low': 0.25,       # 0.00 - 0.25: LOW RISK
            'moderate': 0.50,  # 0.25 - 0.50: MODERATE RISK
            'high': 0.75,      # 0.50 - 0.75: HIGH RISK
            'critical': 1.00   # 0.75 - 1.00: CRITICAL RISK
        },
        'labels': {
            0: 'LOW',
            1: 'MODERATE',
            2: 'HIGH',
            3: 'CRITICAL'
        },
        'colors': {
            0: '#28a745',  # Green
            1: '#ffc107',  # Yellow/Amber
            2: '#fd7e14',  # Orange
            3: '#dc3545'   # Red
        },
        'actions': {
            0: 'Normal operation - Continue routine monitoring',
            1: 'Early warning - Increase monitoring frequency',
            2: 'Action recommended - Schedule maintenance within 24-48 hours',
            3: 'Immediate action required - Urgent maintenance needed NOW'
        },
        'descriptions': {
            0: 'Filter operating normally. No signs of clogging detected.',
            1: 'Minor degradation detected. Filter performance within acceptable range.',
            2: 'Significant degradation detected. Filter approaching critical threshold.',
            3: 'Severe clogging detected. Filter performance critically impaired.'
        },
        # Cost weighting for operational cost calculation
        'level_costs': {
            0: 0,    # No cost for correct low risk
            1: 5,    # Small cost for moderate risk (increased monitoring)
            2: 25,   # Moderate cost for high risk (scheduled maintenance)
            3: 100   # High cost for critical risk (urgent maintenance/failure)
        }
    },

    # Logging
    'verbose': True,
}


def get_config():
    """Get a copy of the configuration dictionary."""
    return CONFIG.copy()


def update_config(updates):
    """
    Update configuration with new values.

    Parameters:
    -----------
    updates : dict
        Dictionary with configuration updates
    """
    CONFIG.update(updates)
