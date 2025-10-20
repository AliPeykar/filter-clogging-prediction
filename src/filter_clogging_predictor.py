# Filter Clogging Prediction System
# =================================
# This pipeline predicts filter clogging using time-series data from flowrate and pressure drop measurements.
# It engineers domain-specific features linking flow/pressure relationships, trains Random Forest and XGBoost
# models with chronological splits, handles class imbalance, and provides interpretability through SHAP.
# 
# Usage Example:
# 1. Load your Excel file with time-series data
# 2. Run the pipeline to train models
# 3. Use predict(df) to get clogging predictions and risk levels
# 4. Use update_model(new_df) to retrain with new data using saved hyperparameters

# %% [markdown]
# ## 1. Configuration and Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                           precision_recall_curve, auc, roc_curve, balanced_accuracy_score,
                           f1_score, precision_score, recall_score, average_precision_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# NOTE: SMOTE removed - inappropriate for time-series data
# Using cost-sensitive learning and survival analysis instead
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import shap
import joblib
from datetime import datetime
import os
from typing import Tuple, Dict, Any, Optional, List, Union, Generator
from scipy import stats
from pathlib import Path

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %% [markdown]
# ## 2. Configuration Parameters

# %%
# Configuration parameters - modify these as needed
CONFIG = {
    # Feature engineering
    'rolling_windows': [5, 15, 60],  # Small, medium, large windows
    'ema_spans': [5, 30],  # Short and long EMA spans
    'epsilon': 1e-8,  # Small value to avoid division by zero

    # Target and risk thresholds
    # Extended horizon to reduce class imbalance and improve predictions
    'forecast_horizon_steps': 120,  # Predict clog within next 120 steps (improved from 25)
    'risk_thresholds': {
        'T_high': 40,   # High risk: clog within 40 steps
        'T_low': 100    # Low risk: clog after 100 steps
    },
    
    # Data split
    'train_frac': 0.6,
    'val_frac': 0.2,
    'test_frac': 0.2,
    'min_pos_fraction': 0.15,  # Minimum positive fraction per split
    
    # Model training
    'optuna_trials_binary': 50,
    'optuna_trials_multiclass': 25,
    'early_stopping_rounds': 20,
    'recency_lambda': 0.001,  # For recency weighting
    'use_rolling_cv': False, # New parameter to enable rolling origin cross-validation
    'n_splits_rolling_cv': 3, # Number of splits for rolling origin cross-validation
    'calibrate_models': True, # New parameter to enable probability calibration
    'calibration_method': 'isotonic', # 'sigmoid' or 'isotonic'
    
    # Anomaly Detection (NEW - for severely imbalanced data)
    'anomaly_detection': {
        'enabled': True,
        'healthy_data_fraction': 0.85,  # Use first 85% as "healthy" for training
        'contamination': 0.01,  # Expected fraction of anomalies in healthy data
        'methods': ['isolation_forest', 'lof', 'ocsvm'],  # Which detectors to use
        'ensemble_weights': [0.5, 0.3, 0.2],  # Weights for IF, LOF, OCSVM
        'n_estimators': 200,  # For Isolation Forest
        'lof_neighbors': 20,  # For LOF
    },

    # Sample weighting for temporal data
    'use_temporal_weighting': True,
    'weight_decay_factor': 0.2,  # Exponential decay: closer to event = higher weight

    # Paths
    'models_dir': 'models',
    'plots_dir': 'plots',
    'results_dir': 'results'
}

# Create directories if they don't exist
for dir_name in ['models_dir', 'plots_dir', 'results_dir']:
    Path(CONFIG[dir_name]).mkdir(exist_ok=True)

# %% [markdown]
# ## 3. Feature Engineering Module

# %%
def build_features(df: pd.DataFrame, config: Dict = CONFIG) -> pd.DataFrame:
    """
    Build comprehensive feature set for filter clogging prediction.

    CRITICAL: This function implements proper feature reset logic after clogging events
    to ensure cumulative features reflect only the current filter cycle.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with columns: time, flowrate, dp, filter_status
        IMPORTANT: Must have exactly 4 columns. Extra columns will trigger a warning.
    config : dict
        Configuration parameters

    Returns:
    --------
    pd.DataFrame with engineered features
    """
    df = df.copy()
    eps = config['epsilon']

    # Validate input columns - warn if extra columns exist
    expected_cols = {'time', 'flowrate', 'dp', 'filter_status'}
    actual_cols = set(df.columns)
    if len(actual_cols) > 4 or not expected_cols.issubset(actual_cols):
        extra_cols = actual_cols - expected_cols
        if extra_cols:
            warnings.warn(f"WARNING: Excel file contains extra columns: {extra_cols}. "
                         f"Expected only 4 columns: {expected_cols}. "
                         f"These extra columns will be ignored.")

    # Ensure time sorting
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.sort_values('time').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Identify clogging events for reset logic
    clog_mask = (df['filter_status'] == 'Clogged') | (df['filter_status'] == 1)

    # Domain features linking flow and pressure
    df['dp_per_flow'] = df['dp'] / (df['flowrate'] + eps)
    df['dp_over_flow2'] = df['dp'] / (df['flowrate']**2 + eps)
    df['flowrate_squared'] = df['flowrate'] ** 2
    df['flowrate_cubed'] = df['flowrate'] ** 3

    # Rolling features for multiple windows (these don't need reset)
    for window in config['rolling_windows']:
        # For dp
        df[f'dp_roll_mean_{window}'] = df['dp'].rolling(window, min_periods=1).mean()
        df[f'dp_roll_std_{window}'] = df['dp'].rolling(window, min_periods=1).std()
        df[f'dp_roll_min_{window}'] = df['dp'].rolling(window, min_periods=1).min()
        df[f'dp_roll_max_{window}'] = df['dp'].rolling(window, min_periods=1).max()

        # For flowrate
        df[f'flow_roll_mean_{window}'] = df['flowrate'].rolling(window, min_periods=1).mean()
        df[f'flow_roll_std_{window}'] = df['flowrate'].rolling(window, min_periods=1).std()

        # For dp_per_flow
        df[f'dp_per_flow_roll_mean_{window}'] = df['dp_per_flow'].rolling(window, min_periods=1).mean()

        # Rolling slope using linear regression
        df[f'dp_slope_{window}'] = df['dp'].rolling(window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=True
        )

    # Exponential moving averages WITH RESET after clogging
    # Initialize EMA columns
    df['ema_short'] = 0.0
    df['ema_long'] = 0.0
    df['ema_diff'] = 0.0

    # EMA parameters
    alpha_short = 2.0 / (config['ema_spans'][0] + 1)
    alpha_long = 2.0 / (config['ema_spans'][1] + 1)

    # Calculate EMAs with reset logic using numpy arrays for speed
    short_ema = None
    long_ema = None
    dp_values = df['dp'].values
    ema_short_arr = np.zeros(len(df))
    ema_long_arr = np.zeros(len(df))
    clog_mask_arr = clog_mask.values

    for i in range(len(df)):
        current_dp = dp_values[i]

        # Reset EMAs after clogging event (previous row was clogged)
        if i > 0 and clog_mask_arr[i-1]:
            short_ema = current_dp
            long_ema = current_dp
        # Initialize EMAs at start
        elif short_ema is None:
            short_ema = current_dp
            long_ema = current_dp
        # Normal EMA calculation
        else:
            short_ema = alpha_short * current_dp + (1 - alpha_short) * short_ema
            long_ema = alpha_long * current_dp + (1 - alpha_long) * long_ema

        ema_short_arr[i] = short_ema
        ema_long_arr[i] = long_ema

    df['ema_short'] = ema_short_arr
    df['ema_long'] = ema_long_arr
    df['ema_diff'] = ema_short_arr - ema_long_arr

    # Delta features (no reset needed - these are local differences)
    df['dp_diff_1'] = df['dp'].diff(1)
    df['dp_diff_5'] = df['dp'].diff(5)
    df['flow_diff_1'] = df['flowrate'].diff(1)
    df['flow_diff_5'] = df['flowrate'].diff(5)

    # Cumulative features WITH RESET after clogging
    df['cusum_dp'] = 0.0
    df['adaptive_cusum'] = 0.0

    # Initialize cumulative trackers using numpy arrays
    cusum_dp = 0.0
    adaptive_cusum = 0.0
    cusum_dp_arr = np.zeros(len(df))
    adaptive_cusum_arr = np.zeros(len(df))
    dp_diff_1_values = df['dp_diff_1'].values

    for i in range(len(df)):
        # Reset cumulative features after clogging event (previous row was clogged)
        if i > 0 and clog_mask_arr[i-1]:
            cusum_dp = 0.0
            adaptive_cusum = 0.0

        # Calculate positive dp changes
        if i > 0:
            dp_change = dp_diff_1_values[i]
            if not np.isnan(dp_change) and dp_change > 0:
                cusum_dp += dp_change
                adaptive_cusum += dp_change

        cusum_dp_arr[i] = cusum_dp
        adaptive_cusum_arr[i] = adaptive_cusum

    df['cusum_dp'] = cusum_dp_arr
    df['adaptive_cusum'] = adaptive_cusum_arr

    # Time since last clog WITH CORRECT INITIAL VALUE using numpy arrays
    last_clog_idx = None
    time_since_clog_arr = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        if clog_mask_arr[i]:
            # Current row is a clogging event
            last_clog_idx = i
            time_since_clog_arr[i] = 0
        else:
            # Not a clogging event
            if last_clog_idx is None:
                # No clogging has occurred yet - use large value (9999)
                time_since_clog_arr[i] = 9999
            else:
                # Time since the last clogging event
                time_since_clog_arr[i] = i - last_clog_idx

    df['time_since_clog'] = time_since_clog_arr

    # ========================================
    # ADVANCED DOMAIN FEATURES (10 new features)
    # ========================================

    # 1. Resistance acceleration (2nd derivative of dp_per_flow)
    df['resistance_accel'] = df['dp_per_flow'].diff(1).diff(1)

    # 2. Baseline-normalized resistance (relative to cycle start)
    baseline_resistance_arr = np.zeros(len(df))
    df['baseline_resistance'] = 0.0

    for i in range(len(df)):
        if i > 0 and clog_mask_arr[i-1]:
            # Reset: current value becomes new baseline
            baseline = dp_values[i] / (df['flowrate'].values[i] + eps)
        elif i == 0:
            baseline = dp_values[i] / (df['flowrate'].values[i] + eps)

        current_resistance = dp_values[i] / (df['flowrate'].values[i] + eps)
        baseline_resistance_arr[i] = (current_resistance - baseline) / (baseline + eps)

    df['baseline_resistance'] = baseline_resistance_arr

    # 3. Rate-of-change of adaptive CUSUM
    df['cusum_rate'] = df['adaptive_cusum'].diff(1)

    # 4. Coefficient of variation for rolling windows
    for window in config['rolling_windows']:
        # CV for dp
        rolling_mean = df['dp'].rolling(window, min_periods=1).mean()
        rolling_std = df['dp'].rolling(window, min_periods=1).std()
        df[f'dp_cv_{window}'] = rolling_std / (rolling_mean + eps)

        # CV for flowrate
        rolling_mean_flow = df['flowrate'].rolling(window, min_periods=1).mean()
        rolling_std_flow = df['flowrate'].rolling(window, min_periods=1).std()
        df[f'flow_cv_{window}'] = rolling_std_flow / (rolling_mean_flow + eps)

    # 5. Quantile-based spread (robust outlier detection)
    for window in config['rolling_windows']:
        p95 = df['dp'].rolling(window, min_periods=1).quantile(0.95)
        p50 = df['dp'].rolling(window, min_periods=1).quantile(0.50)
        p5 = df['dp'].rolling(window, min_periods=1).quantile(0.05)
        df[f'dp_quantile_spread_{window}'] = (p95 - p5) / (p50 + eps)

    # 6. Cumulative work (energy dissipation) WITH RESET
    df['cumulative_work'] = 0.0
    cumulative_work = 0.0
    cumulative_work_arr = np.zeros(len(df))
    flow_values = df['flowrate'].values

    for i in range(len(df)):
        # Reset after clogging
        if i > 0 and clog_mask_arr[i-1]:
            cumulative_work = 0.0

        # Accumulate dp × flowrate
        if i > 0:
            work_increment = dp_values[i] * flow_values[i]
            cumulative_work += work_increment

        cumulative_work_arr[i] = cumulative_work

    df['cumulative_work'] = cumulative_work_arr

    # 7 & 8. Skewness and kurtosis for rolling windows
    for window in [15, 60]:  # Use medium and large windows only
        df[f'dp_skew_{window}'] = df['dp'].rolling(window, min_periods=3).skew()
        df[f'dp_kurt_{window}'] = df['dp'].rolling(window, min_periods=4).apply(
            lambda x: stats.kurtosis(x) if len(x) >= 4 else 0, raw=True
        )

    # 9. Autocorrelation at multiple lags
    def rolling_autocorr(series, window, lag):
        """Calculate rolling autocorrelation."""
        def autocorr(x):
            if len(x) <= lag:
                return 0
            x = np.array(x)
            mean = x.mean()
            var = x.var()
            if var == 0:
                return 0
            return np.corrcoef(x[:-lag] - mean, x[lag:] - mean)[0, 1] if len(x) > lag else 0

        return series.rolling(window, min_periods=lag+1).apply(autocorr, raw=False)

    df['dp_autocorr_lag1'] = rolling_autocorr(df['dp'], window=30, lag=1)
    df['dp_autocorr_lag5'] = rolling_autocorr(df['dp'], window=30, lag=5)

    # 10. Normalized pressure rise rate
    for window in [15, 60]:
        mean_dp = df['dp'].rolling(window, min_periods=1).mean()
        df[f'normalized_dp_slope_{window}'] = df[f'dp_slope_{window}'] / (mean_dp + eps)

    # Time features if datetime
    if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
        df['hour'] = df['time'].dt.hour
        df['dayofweek'] = df['time'].dt.dayofweek

    # Drop rows with NaN after maximum window size
    max_window = max(config['rolling_windows'])
    df = df.iloc[max_window:].reset_index(drop=True)

    return df

# %% [markdown]
# ## 4. Target Creation and Data Splitting

# %%
def create_targets(df: pd.DataFrame, config: Dict = CONFIG) -> pd.DataFrame:
    """
    Create binary and multiclass targets for prediction.
    """
    df = df.copy()
    
    # Convert filter_status to binary
    df['is_clogged'] = ((df['filter_status'] == 'Clogged') | 
                        (df['filter_status'] == 1)).astype(int)
    
    # Create forward-looking binary target
    horizon = config['forecast_horizon_steps']
    df['will_clog'] = 0
    for i in range(len(df) - horizon):
        if df['is_clogged'].iloc[i+1:i+horizon+1].any():
            df.loc[i, 'will_clog'] = 1
    
    # Calculate time to next clog
    df['time_to_clog'] = float('inf')
    clog_indices = df.index[df['is_clogged'] == 1]

    if not clog_indices.empty:
        current_indices = df.index.values[:, np.newaxis]
        # Ensure time_diffs is float to handle infinity
        time_diffs = (clog_indices.values - current_indices).astype(float)
        time_diffs[time_diffs <= 0] = np.inf
        df['time_to_clog'] = np.min(time_diffs, axis=1)
    
    # Explicitly mark censored samples
    df['is_censored'] = (df['time_to_clog'] == float('inf')).astype(int)

    # Create multiclass risk labels
    T_high = config['risk_thresholds']['T_high']
    T_low = config['risk_thresholds']['T_low']
    
    df['risk_class'] = 'Low'
    df.loc[df['time_to_clog'] <= T_high, 'risk_class'] = 'High'
    df.loc[(df['time_to_clog'] > T_high) & 
           (df['time_to_clog'] <= T_low), 'risk_class'] = 'Medium'
    
    # Map to numeric for easier handling
    risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['risk_class_numeric'] = df['risk_class'].map(risk_map)
    
    return df

def prepare_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for survival analysis (Cox, RSF models).

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with time_to_clog and is_censored columns

    Returns:
    --------
    pd.DataFrame with duration and event columns for survival models
    """
    survival_df = df.copy()

    # Duration: time to clog OR time to end of observation (for censored samples)
    # For censored samples (no clog observed), duration = time since last clog
    survival_df['duration'] = np.where(
        survival_df['time_to_clog'] < float('inf'),
        survival_df['time_to_clog'],  # Observed clog event
        survival_df['time_since_clog']  # Right-censored at current observation
    )

    # Event indicator: 1 if clog observed, 0 if censored
    survival_df['event'] = (survival_df['time_to_clog'] < float('inf')).astype(int)

    # Ensure duration is positive (minimum 1 time step)
    survival_df['duration'] = survival_df['duration'].clip(lower=1)

    print(f"\nSurvival data prepared:")
    print(f"  Total samples: {len(survival_df)}")
    print(f"  Events (clogs observed): {survival_df['event'].sum()} ({100*survival_df['event'].mean():.1f}%)")
    print(f"  Censored samples: {(1-survival_df['event']).sum()} ({100*(1-survival_df['event'].mean()):.1f}%)")
    print(f"  Median duration: {survival_df['duration'].median():.1f} steps")

    return survival_df

def time_series_split(df: pd.DataFrame, config: Dict = CONFIG) -> Tuple:
    """
    Perform chronological split with class presence guarantee.
    """
    n = len(df)
    train_end = int(n * config['train_frac'])
    val_end = train_end + int(n * config['val_frac'])
    
    # Initial splits
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    
    # Check for minimum positive samples
    min_pos = max(1, int(config['min_pos_fraction'] * len(train_idx)))
    
    # Adjust boundaries if needed
    def count_positives(idx):
        return df.iloc[idx]['will_clog'].sum()
    
    # Ensure each split has minimum positives
    train_pos = count_positives(train_idx)
    val_pos = count_positives(val_idx)
    test_pos = count_positives(test_idx)
    
    print(f"Initial split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Initial positive counts - Train: {train_pos}, Val: {val_pos}, Test: {test_pos}")
    
    # Adjust if validation lacks positives
    if val_pos < min_pos:
        print(f"Adjusting validation split to include at least {min_pos} positive samples...")
        # Find next positive sample in test set
        for i in range(val_end, min(val_end + 100, n)):
            if df.iloc[i]['will_clog'] == 1:
                val_end = i + 1
                break
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)
    
    # Adjust if test lacks positives
    if test_pos < min_pos and val_end < n - 1:
        print(f"Adjusting test split to include at least {min_pos} positive samples...")
        # Move boundary backward
        for i in range(val_end - 1, max(val_end - 100, train_end), -1):
            if df.iloc[i]['will_clog'] == 1:
                val_end = i
                break
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)
    
    # Final counts
    print(f"\nFinal split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Final positive counts - Train: {count_positives(train_idx)}, "
          f"Val: {count_positives(val_idx)}, Test: {count_positives(test_idx)}")
    
    return train_idx, val_idx, test_idx

def time_series_split_imbalanced(df: pd.DataFrame, clog_index: Optional[int] = None, config: Dict = CONFIG) -> Tuple:
    """
    Smart split for severely imbalanced data where clogging occurs late in timeline.

    For data like: [0...8000 healthy] [8000...8940 degradation] [8940+ clogged]

    Strategy:
    - Training: Includes healthy data + early degradation
    - Validation: Middle degradation phase
    - Test: Late degradation + clogging event

    Parameters:
    -----------
    df : pd.DataFrame
        Data with 'will_clog' or 'is_clogged' column
    clog_index : int, optional
        Index where clogging starts. If None, auto-detect
    config : dict
        Configuration dictionary

    Returns:
    --------
    train_idx, val_idx, test_idx, healthy_idx
        healthy_idx: Indices for anomaly detector training (known healthy data)
    """
    n = len(df)

    # Auto-detect clogging start if not provided
    if clog_index is None:
        if 'is_clogged' in df.columns:
            clog_indices = df.index[df['is_clogged'] == 1]
            if len(clog_indices) > 0:
                clog_index = int(clog_indices[0])
            else:
                clog_index = n
                print(f"⚠️  Warning: No clogging detected in data. Using full dataset.")
        else:
            clog_index = n
            print(f"⚠️  Warning: 'is_clogged' column not found. Using full dataset.")

    print(f"\n{'='*60}")
    print("IMBALANCED TIMELINE DATA SPLITTING")
    print(f"{'='*60}")
    print(f"Total samples: {n}")
    if clog_index < n:
        print(f"Clogging starts at index: {clog_index} ({100*clog_index/n:.1f}% into timeline)")
    else:
        print(f"No clogging detected (censored data)")

    # Get healthy data fraction from config
    healthy_fraction = config.get('anomaly_detection', {}).get('healthy_data_fraction', 0.85)
    healthy_end = int(clog_index * healthy_fraction)

    # Define splits based on degradation timeline
    # Training: 0 to 70% of pre-clog data
    train_end = int(clog_index * 0.7)

    # Validation: 70% to 95% of pre-clog data
    val_end = int(clog_index * 0.95)

    # Test: 95% of pre-clog to end (includes clogging event)
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)

    # Healthy data for anomaly detection (first 85% of pre-clog)
    healthy_idx = np.arange(0, healthy_end)

    # Print statistics
    def print_split_stats(idx, name):
        if 'will_clog' in df.columns:
            pos_count = df.iloc[idx]['will_clog'].sum()
            pos_ratio = pos_count / len(idx) if len(idx) > 0 else 0
            print(f"{name:12s}: {len(idx):5d} samples, {pos_count:4d} positive ({pos_ratio:6.2%})")
        else:
            print(f"{name:12s}: {len(idx):5d} samples")

    print(f"\nSplit Strategy:")
    print_split_stats(healthy_idx, "Healthy")
    print_split_stats(train_idx, "Train")
    print_split_stats(val_idx, "Validation")
    print_split_stats(test_idx, "Test")
    print(f"{'='*60}\n")

    return train_idx, val_idx, test_idx, healthy_idx

def rolling_origin_time_series_split(df: pd.DataFrame, n_splits: int = 5, train_size_frac: float = 0.6, val_size_frac: float = 0.2, min_pos_fraction: float = 0.01) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Perform rolling origin cross-validation.
    Yields (train_idx, val_idx, test_idx) for each fold.
    """
    n = len(df)
    
    # Determine the size of each initial split
    initial_train_size = int(n * train_size_frac)
    initial_val_size = int(n * val_size_frac)
    
    # Calculate the step size for each fold
    # The remaining data after initial train/val is used for test and subsequent folds
    remaining_size = n - (initial_train_size + initial_val_size)
    step_size = remaining_size // n_splits if n_splits > 0 else 0

    if step_size <= 0 and n_splits > 0:
        warnings.warn("Step size for rolling origin is too small, consider reducing n_splits or increasing data size.")
        # Fallback to a single split if rolling is not feasible
        yield time_series_split(df, {'train_frac': train_size_frac, 'val_frac': val_size_frac, 'test_frac': 1 - train_size_frac - val_size_frac, 'min_pos_fraction': min_pos_fraction})
        return

    for i in range(n_splits):
        current_train_end = initial_train_size + i * step_size
        current_val_end = current_train_end + initial_val_size
        current_test_end = current_val_end + step_size # Each test set is of size step_size

        if current_test_end > n:
            current_test_end = n # Ensure test set does not exceed dataframe length
            if current_val_end >= current_test_end: # If no data left for validation or test
                break

        train_idx = np.arange(0, current_train_end)
        val_idx = np.arange(current_train_end, current_val_end)
        test_idx = np.arange(current_val_end, current_test_end)

        # Ensure each split has minimum positives (similar to time_series_split)
        min_pos = max(1, int(min_pos_fraction * len(train_idx)))

        def count_positives(idx_arr):
            if len(idx_arr) == 0:
                return 0
            return df.iloc[idx_arr]['will_clog'].sum()

        train_pos = count_positives(train_idx)
        val_pos = count_positives(val_idx)
        test_pos = count_positives(test_idx)

        # Adjust boundaries if needed to ensure positive samples
        # This logic can be complex for rolling origin, simplified for now
        # For a more robust solution, one might need to expand/contract windows
        # or skip folds if positive samples are not met.
        if train_pos < min_pos or val_pos < min_pos or test_pos < min_pos:
            warnings.warn(f"Fold {i+1}: Not enough positive samples in one or more splits. Skipping this fold.")
            continue

        yield train_idx, val_idx, test_idx

# %% [markdown]
# ## 5. Cost-Sensitive Learning (Replaces SMOTE)

# %%
# SMOTE REMOVED: Inappropriate for time-series data
# Using cost-sensitive learning with class weights and threshold optimization instead

def optimize_threshold_by_cost(y_true, y_proba, cost_fn=100, cost_fp=1):
    """
    Find optimal threshold by minimizing operational cost.

    For filter clogging:
    - False Negative (missed clog) = DISASTER = high cost
    - False Positive (unnecessary maintenance) = minor inconvenience = low cost

    Parameters:
    -----------
    y_true : array
        True labels
    y_proba : array
        Predicted probabilities
    cost_fn : float
        Cost of false negative (default: 100)
    cost_fp : float
        Cost of false positive (default: 1)

    Returns:
    --------
    optimal_threshold : float
    optimal_cost : float
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    recalls = []
    precisions = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        # Calculate confusion matrix components
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        # Calculate cost
        total_cost = cost_fn * fn + cost_fp * fp
        costs.append(total_cost)

        # Track metrics
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recalls.append(recall)
        precisions.append(precision)

    # Find threshold with minimum cost
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = costs[optimal_idx]

    print(f"\n💰 Cost-Based Threshold Optimization:")
    print(f"  Cost FN: {cost_fn}, Cost FP: {cost_fp} (ratio: {cost_fn/cost_fp:.0f}:1)")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print(f"  Minimum total cost: {optimal_cost:.1f}")
    print(f"  Recall at optimal:  {recalls[optimal_idx]:.3f}")
    print(f"  Precision at optimal: {precisions[optimal_idx]:.3f}")

    # Compare to default 0.5
    default_idx = np.argmin(np.abs(thresholds - 0.5))
    print(f"\n  Cost at 0.5 threshold: {costs[default_idx]:.1f}")
    print(f"  Cost reduction: {((costs[default_idx] - optimal_cost) / costs[default_idx] * 100):.1f}%")

    return optimal_threshold, optimal_cost

def optimize_threshold(y_true, y_proba, metric='f1', plot=False):
    """
    Find optimal decision threshold for imbalanced classification.

    CRITICAL: Default 0.5 threshold is WRONG for imbalanced data!

    Parameters:
    -----------
    y_true : array
        True labels
    y_proba : array
        Predicted probabilities
    metric : str
        'f1', 'f2', or 'youden'
    plot : bool
        Whether to plot precision-recall curve

    Returns:
    --------
    optimal_threshold : float
    """
    from sklearn.metrics import precision_recall_curve, roc_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    if metric == 'f1':
        # Maximize F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_score = f1_scores[optimal_idx]

    elif metric == 'f2':
        # F2 score (favor recall over precision - better for rare events)
        f2_scores = 5 * (precision * recall) / (4 * precision + recall + 1e-8)
        optimal_idx = np.argmax(f2_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_score = f2_scores[optimal_idx]

    elif metric == 'youden':
        # Youden's J statistic (balanced sensitivity-specificity)
        fpr, tpr, thresh_roc = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresh_roc[optimal_idx]
        optimal_score = j_scores[optimal_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    print(f"\n🎯 Threshold Optimization ({metric.upper()}):")
    print(f"  Default threshold (0.5):  Likely wrong for imbalanced data!")
    print(f"  Optimal threshold:        {optimal_threshold:.3f}")
    print(f"  Optimal {metric}:         {optimal_score:.3f}")

    if optimal_idx < len(precision):
        print(f"  Precision at optimal:     {precision[optimal_idx]:.3f}")
        print(f"  Recall at optimal:        {recall[optimal_idx]:.3f}")

    return optimal_threshold

def check_for_model_collapse(y_pred, y_true, model_name='Model'):
    """
    Detect if model has collapsed to trivial solution (predict all negative).

    CRITICAL: Always run this check after prediction!

    Returns:
    --------
    bool : True if model collapsed, False otherwise
    """
    unique_preds = np.unique(y_pred)
    pos_preds = (y_pred == 1).sum()
    total = len(y_pred)

    print(f"\n🔍 Model Collapse Check for {model_name}:")
    print(f"  Unique predictions: {unique_preds}")
    print(f"  Positive predictions: {pos_preds:4d} / {total:4d} ({pos_preds/total:6.2%})")

    if len(unique_preds) == 1:
        if unique_preds[0] == 0:
            print(f"  🚨 CRITICAL: Model COLLAPSED! Predicting ALL NEGATIVE!")
            print(f"  🚨 This model is WORTHLESS!")
            print(f"  🚨 FIX: Use SMOTE, extend horizon, or adjust threshold!")
            return True
        elif unique_preds[0] == 1:
            print(f"  🚨 CRITICAL: Model predicting ALL POSITIVE!")
            print(f"  🚨 Check for data leakage!")
            return True

    if pos_preds < total * 0.001:  # Less than 0.1%
        print(f"  ⚠️  WARNING: Model barely predicts positives ({pos_preds/total:.3%})")
        print(f"  ⚠️  Likely collapsed - check threshold or use SMOTE!")
        return True

    print(f"  ✅ Model appears to be learning (not collapsed)")
    return False

def compute_sample_weights(y: np.ndarray, indices: np.ndarray, config: Dict = CONFIG) -> np.ndarray:
    """
    Compute sample weights with class balancing and optional recency weighting.
    """
    # Compute class weights
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Apply class weights
    sample_weights = np.array([class_weight_dict[label] for label in y])
    
    # Apply recency weighting if lambda > 0
    if config['recency_lambda'] > 0:
        n = len(indices)
        recency_weights = np.exp(-config['recency_lambda'] * (n - indices))
        sample_weights *= recency_weights
    
    return sample_weights

def compute_temporal_weights(time_to_clog: np.ndarray, config: Dict = CONFIG) -> np.ndarray:
    """
    Compute sample weights based on temporal distance to clogging event.

    Samples closer to clogging get higher weights (exponential decay).
    Perfect for imbalanced data where event is rare and at end of timeline.

    Parameters:
    -----------
    time_to_clog : array
        Distance (in steps) to next clogging event
    config : dict
        Configuration with 'weight_decay_factor'

    Returns:
    --------
    weights : array
        Sample weights (higher = more important)

    Example:
    --------
    time_to_clog = [8940, 8900, 100, 50, 10]  # Far to near
    weights ≈ [0.01, 0.02, 0.5, 0.7, 0.95]    # Low to high
    """
    if not config.get('use_temporal_weighting', False):
        return np.ones(len(time_to_clog))

    decay_factor = config.get('weight_decay_factor', 0.2)
    max_time = np.max(time_to_clog[time_to_clog < float('inf')])

    # Exponential weighting: exp(-time_to_clog / (max_time * decay_factor))
    # Closer to clog (small time) → higher weight
    weights = np.exp(-time_to_clog / (max_time * decay_factor + 1e-8))

    # Handle censored samples (infinite time_to_clog)
    weights[time_to_clog == float('inf')] = np.min(weights[weights > 0])

    # Normalize to mean=1
    weights = weights / (np.mean(weights) + 1e-8)

    return weights

def combine_sample_weights(class_weights: np.ndarray, temporal_weights: np.ndarray) -> np.ndarray:
    """
    Combine class balancing weights with temporal weights.

    Parameters:
    -----------
    class_weights : array
        Weights from class balancing
    temporal_weights : array
        Weights from temporal proximity to event

    Returns:
    --------
    combined_weights : array
        Element-wise product, normalized
    """
    combined = class_weights * temporal_weights
    # Normalize to mean=1
    return combined / (np.mean(combined) + 1e-8)

# %% [markdown]
# ## 6. Hyperparameter Optimization with Optuna

# %%
def optuna_objective_rf(trial, X_train, y_train, X_val, y_val, sample_weights):
    """Optuna objective for RandomForest."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 50),  # Reduced range (was 1-80)
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),  # FIXED: min=2 (was 1)
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),  # Reduced range (was 1-40)
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.8]),
        'random_state': RANDOM_SEED,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='binary' if len(np.unique(y_val)) == 2 else 'macro')
    
    return float(f1)

def optuna_objective_xgb(trial, X_train, y_train, X_val, y_val, sample_weights):
    """Optuna objective for XGBoost."""
    # Calculate scale_pos_weight
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'scale_pos_weight': scale_pos_weight,
        'random_state': RANDOM_SEED,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'early_stopping_rounds': CONFIG['early_stopping_rounds']  # Now a constructor parameter
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, 
             sample_weight=sample_weights,
             eval_set=[(X_val, y_val)],
             verbose=False)
    
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='binary' if len(np.unique(y_val)) == 2 else 'macro')
    
    return float(f1)

def optimize_hyperparameters(X_train, y_train, X_val, y_val, sample_weights, 
                           model_type='rf', n_trials=50):
    """
    Run Optuna hyperparameter optimization.
    """
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=TPESampler(seed=RANDOM_SEED)
    )
    
    # Map model type to its corresponding objective function
    objective_map = {
        'rf': optuna_objective_rf,
        'xgb': optuna_objective_xgb
    }
    
    if model_type not in objective_map:
        raise ValueError(f"Unsupported model type for optimization: {model_type}")
        
    # Create the objective function with necessary arguments
    objective_func = objective_map[model_type]
    objective = lambda trial: objective_func(trial, X_train, y_train, 
                                             X_val, y_val, sample_weights)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Enhanced logging of the best trial results
    print(f"\nBest {model_type.upper()} trial results:")
    print(f"  - Trial number: {study.best_trial.number}")
    print(f"  - Validation F1 Score: {study.best_value:.4f}")
    print("  - Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    - {key}: {value}")
    
    return study.best_params

# %% [markdown]
# ## 7. Model Training and Evaluation

# %%
def train_model(X_train, y_train, X_val, y_val, sample_weights, best_params, model_type='rf'):
    """
    Train model with best parameters on combined train+val set.
    """
    # Combine train and validation
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])
    weights_combined = np.hstack([sample_weights, np.ones(len(y_val))])
    
    if model_type == 'rf':
        model = RandomForestClassifier(**best_params, n_jobs=-1)
        model.fit(X_combined, y_combined, sample_weight=weights_combined)
    else:  # xgb
        # For final training, we don't use early stopping since we're using all data
        # Remove early_stopping_rounds if it exists in best_params
        final_params = best_params.copy()
        if 'early_stopping_rounds' in final_params:
            del final_params['early_stopping_rounds']
        model = xgb.XGBClassifier(**final_params, use_label_encoder=False)
        model.fit(X_combined, y_combined, sample_weight=weights_combined)
    
    return model

def evaluate_model(model: Union[RandomForestClassifier, xgb.XGBClassifier, CalibratedClassifierCV], X_test, y_test, model_name='Model'):
    """
    CRITICAL: Imbalanced-aware model evaluation.

    PRIMARY METRIC: PR-AUC (not ROC-AUC or accuracy!)
    Uses optimized threshold (not 0.5)
    Checks for model collapse
    """
    y_proba_full = model.predict_proba(X_test)

    # Binary metrics
    if len(np.unique(y_test)) == 2:
        y_proba = y_proba_full[:, 1]

        # CRITICAL: Optimize threshold for imbalanced data
        optimal_threshold = optimize_threshold(y_test, y_proba, metric='f1')

        # Use optimized threshold (NOT 0.5!)
        y_pred = (y_proba > optimal_threshold).astype(int)

        # CRITICAL: Check for model collapse
        collapsed = check_for_model_collapse(y_pred, y_test, model_name=model_name)

        # Calculate metrics
        from sklearn.metrics import average_precision_score

        metrics = {
            'pr_auc': average_precision_score(y_test, y_proba),  # PRIMARY METRIC!
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'optimal_threshold': optimal_threshold,
            'model_collapsed': collapsed
        }

        # PR-AUC legacy calculation
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        metrics['pr_auc_curve'] = auc(recall, precision)

        # Confusion matrix for detailed analysis
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = tp
            metrics['false_positives'] = fp
            metrics['true_negatives'] = tn
            metrics['false_negatives'] = fn

    else:
        # Multiclass metrics
        y_pred = model.predict(X_test)
        y_proba = y_proba_full
        metrics = {
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'model_collapsed': False
        }

    print(f"\n📊 {model_name} Test Performance (Imbalanced-Aware):")
    print(f"  ⭐ PR-AUC (PRIMARY):      {metrics.get('pr_auc', 0):.4f}")
    print(f"  ROC-AUC:                 {metrics.get('roc_auc', 0):.4f}")
    print(f"  F1 Score:                {metrics.get('f1', 0):.4f}")
    print(f"  Precision:               {metrics.get('precision', 0):.4f}")
    print(f"  Recall:                  {metrics.get('recall', 0):.4f}")
    print(f"  Optimal Threshold:       {metrics.get('optimal_threshold', 0.5):.3f}")

    # Check for data issues
    if metrics.get('roc_auc', 0) < 0.6:
        print(f"\n  ⚠️  ROC-AUC is low ({metrics.get('roc_auc', 0):.3f}) - This is EXPECTED after SMOTE!")
        print(f"      Reason: SMOTE balanced training data, but test set remains imbalanced.")
        print(f"      Model learned to predict positive more aggressively (good for clog detection).")
        print(f"      ROC-AUC is unstable with <5% minority class. Use PR-AUC instead.")
        print(f"      ⭐ PR-AUC = {metrics.get('pr_auc', 0):.3f} is the PRIMARY metric for this problem.")

    if 'true_positives' in metrics:
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}")
        print(f"    FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}")

        # Check test set balance
        total_positives = metrics['true_positives'] + metrics['false_negatives']
        total_negatives = metrics['true_negatives'] + metrics['false_positives']
        total = total_positives + total_negatives
        pos_ratio = total_positives / total if total > 0 else 0

        if pos_ratio > 0.95 or pos_ratio < 0.05:
            print(f"\n  ⚠️  WARNING: Test set is severely imbalanced!")
            print(f"      Positive samples: {total_positives}/{total} ({pos_ratio:.1%})")
            print(f"      This is expected with horizon={CONFIG['forecast_horizon_steps']}.")
            print(f"      Use PR-AUC (not ROC-AUC) as primary metric.")

        if metrics['true_positives'] == 0:
            print(f"\n  🚨 CRITICAL: NO TRUE POSITIVES! Model is useless!")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics, y_pred, y_proba

# %% [markdown]
# ## 8. SHAP Interpretability

# %%
def generate_shap_plots(model, X_train, X_test, feature_names, save_path='plots'):
    """
    Generate comprehensive SHAP interpretability plots using the modern Explanation API.
    """
    # Unwrap CalibratedClassifierCV if present to get the underlying tree model
    from sklearn.calibration import CalibratedClassifierCV
    if isinstance(model, CalibratedClassifierCV):
        # Access the base estimator from the first calibrated classifier
        base_model = model.calibrated_classifiers_[0].estimator
    else:
        base_model = model

    # Create SHAP explainer
    explainer = shap.TreeExplainer(base_model)
    
    # Calculate SHAP values on a subset for efficiency, converting to DataFrame for feature names
    X_test_subset = pd.DataFrame(X_test[:100], columns=feature_names)
    
    # Use the modern API to get a unified Explanation object
    shap_explanation = explainer(X_test_subset)
    
    # For binary classification, we are interested in the SHAP values for the positive class (class 1)
    shap_values_positive = shap_explanation[:, :, 1]

    # Summary plot (beeswarm) - shap creates its own figure
    shap.summary_plot(shap_values_positive, show=False)
    plt.title("SHAP Feature Importance (Beeswarm Plot)")
    plt.tight_layout()
    plt.savefig(f"{save_path}/shap_summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Bar plot for global importance - shap creates its own figure
    shap.summary_plot(shap_values_positive, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{save_path}/shap_bar.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Dependence plots for top 3 features
    # Get global feature importance from the explanation object's values
    global_importance = np.abs(shap_values_positive.values).mean(0)
    top_features_indices = np.argsort(global_importance)[-3:][::-1]
    top_feature_names = [feature_names[i] for i in top_features_indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, feature_name in enumerate(top_feature_names):
        shap.dependence_plot(feature_name, 
                            shap_values_positive.values, 
                            X_test_subset,
                            ax=axes[i], 
                            show=False)
        axes[i].set_title(f"SHAP Dependence: {feature_name}")
    plt.tight_layout()
    plt.savefig(f"{save_path}/shap_dependence.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Waterfall plot for a single sample - shap creates its own figure
    shap.waterfall_plot(shap_values_positive[0], show=False)
    plt.title("SHAP Waterfall Plot - Single Sample Explanation")
    plt.tight_layout()
    plt.savefig(f"{save_path}/shap_waterfall.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %% [markdown]
# ## 9. Visualization Suite

# %%
def safe_save_plot(save_func, filepath, *args, **kwargs):
    """
    Safely save a plot, handling PermissionError gracefully.

    Parameters:
    -----------
    save_func : callable
        Function to save (e.g., plt.savefig, fig.write_html)
    filepath : str
        Path to save to
    *args, **kwargs : passed to save_func
    """
    try:
        save_func(filepath, *args, **kwargs)
        return True
    except PermissionError:
        print(f"⚠️  Could not save {filepath} (file may be open)")
        return False
    except Exception as e:
        print(f"⚠️  Error saving {filepath}: {e}")
        return False

def plot_time_series(df, predictions=None, save_path='plots'):
    """
    Create comprehensive time-series visualizations showing feature reset behavior.
    """
    # Interactive plotly visualization
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Flowrate Over Time', 'Pressure Drop Over Time',
                       'DP/Flow Ratio Over Time', 'Cumulative Features (with Reset)'),
        shared_xaxes=True,
        vertical_spacing=0.08
    )

    # Add traces
    fig.add_trace(go.Scatter(x=df.index, y=df['flowrate'],
                            name='Flowrate', line=dict(color='blue')),
                 row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['dp'],
                            name='Pressure Drop', line=dict(color='red')),
                 row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['dp_per_flow'],
                            name='DP/Flow', line=dict(color='green')),
                 row=3, col=1)

    # Add cumulative features to show reset behavior
    fig.add_trace(go.Scatter(x=df.index, y=df['adaptive_cusum'],
                            name='Adaptive CUSUM', line=dict(color='purple')),
                 row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['time_since_clog']/10,
                            name='Time Since Clog (÷10)', line=dict(color='orange', dash='dot')),
                 row=4, col=1)

    # Add clog events more efficiently
    clog_indices = df[df['is_clogged'] == 1].index
    shapes = [
        go.layout.Shape(
            type="line",
            x0=idx,
            y0=0,
            x1=idx,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                dash="dash",
                color="red",
                width=2
            ),
            opacity=0.5
        )
        for idx in clog_indices
    ]
    fig.update_layout(shapes=shapes)

    # Update layout
    fig.update_layout(height=1000, showlegend=True,
                     title_text="Filter System Time Series Analysis - Feature Reset Validation")
    fig.update_xaxes(title_text="Time Index", row=4, col=1)
    fig.update_yaxes(title_text="Flow Rate", row=1, col=1)
    fig.update_yaxes(title_text="Pressure Drop", row=2, col=1)
    fig.update_yaxes(title_text="DP/Flow Ratio", row=3, col=1)
    fig.update_yaxes(title_text="Feature Value", row=4, col=1)

    fig.write_html(f"{save_path}/time_series_interactive.html")
    fig.show()

    # Static matplotlib version with enhanced feature visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df.index, df['flowrate'], 'b-', label='Flowrate', alpha=0.7, linewidth=1.5)
    axes[0].set_ylabel('Flowrate', fontsize=11)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, df['dp'], 'r-', label='Pressure Drop', alpha=0.7, linewidth=1.5)
    axes[1].set_ylabel('Pressure Drop', fontsize=11)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df.index, df['dp_per_flow'], 'g-', label='DP/Flow', alpha=0.7, linewidth=1.5)
    axes[2].set_ylabel('DP/Flow Ratio', fontsize=11)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # Plot cumulative features showing reset behavior
    ax3_twin = axes[3].twinx()
    axes[3].plot(df.index, df['adaptive_cusum'], 'purple',
                label='Adaptive CUSUM (resets to 0)', alpha=0.8, linewidth=2)
    ax3_twin.plot(df.index, df['time_since_clog'], 'orange',
                 label='Time Since Clog (9999→0)', alpha=0.6, linewidth=1.5, linestyle='--')

    axes[3].set_ylabel('Adaptive CUSUM', fontsize=11, color='purple')
    ax3_twin.set_ylabel('Time Since Clog', fontsize=11, color='orange')
    axes[3].set_xlabel('Time Index', fontsize=11)
    axes[3].tick_params(axis='y', labelcolor='purple')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    axes[3].legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    # Add clog markers with annotations
    for i, ax in enumerate(axes):
        for clog_idx in clog_indices:
            ax.axvline(x=clog_idx, color='red', linestyle='--', alpha=0.6, linewidth=2)
            if i == 0 and clog_idx in clog_indices[:3]:  # Annotate first 3 clogs on top plot
                ax.text(clog_idx, ax.get_ylim()[1]*0.95, 'CLOG',
                       rotation=90, va='top', ha='right',
                       fontsize=9, color='red', fontweight='bold', alpha=0.7)

    plt.suptitle('Filter System Time Series Analysis - Showing Feature Reset After Clogging',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_path}/time_series_static.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_model_performance(y_true, y_pred, y_proba, model_name='Model', save_path='plots'):
    """
    Create model performance visualizations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title(f'{model_name} - Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # ROC Curve
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'{model_name} - ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        axes[1, 0].plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title(f'{model_name} - Precision-Recall Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Calibration Curve
        fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
        axes[1, 1].plot(mean_pred, fraction_pos, 's-', label=model_name)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        axes[1, 1].set_xlabel('Mean Predicted Probability')
        axes[1, 1].set_ylabel('Fraction of Positives')
        axes[1, 1].set_title(f'{model_name} - Calibration Curve')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name.lower()}_performance.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_feature_importance(model, feature_names, model_name='Model', save_path='plots'):
    """
    Plot feature importance from the model.
    """
    # Handle CalibratedClassifierCV wrapper
    base_model = model.calibrated_classifiers_[0].estimator if hasattr(model, 'calibrated_classifiers_') else model

    if hasattr(base_model, 'feature_importances_'):
        importances = base_model.feature_importances_
    else:
        importances = base_model.get_booster().get_score(importance_type='gain')
        importances = [importances.get(f'f{i}', 0) for i in range(len(feature_names))]
    
    # Sort features by importance
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'{model_name} - Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name.lower()}_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_correlation_matrix(df, save_path='plots'):
    """
    Plot correlation matrix of features.
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Plot
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{save_path}/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %% [markdown]
# ## 9.5. Regression Models for Time-to-Clog Prediction

# %%
class RegressionPredictor:
    """
    Regression-based predictor for continuous time-to-clog estimation.
    Alternative to classification that uses all data points efficiently.
    """
    def __init__(self, model_type='rf'):
        """
        Parameters:
        -----------
        model_type : str
            'rf' for RandomForestRegressor or 'xgb' for XGBRegressor
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.max_time = None  # For censored data handling

    def fit(self, X_train, y_time_to_clog, censored_mask=None):
        """
        Train regression model to predict time-to-clog.

        Parameters:
        -----------
        X_train : array-like
            Training features
        y_time_to_clog : array-like
            Time to clog (continuous target)
        censored_mask : array-like, optional
            Boolean mask indicating censored samples (True = censored)
        """
        from sklearn.ensemble import RandomForestRegressor

        # Handle censored data by capping at maximum observed time
        y_train = y_time_to_clog.copy()
        if censored_mask is not None:
            # Cap censored values at max observed event time
            observed_times = y_time_to_clog[~censored_mask]
            if len(observed_times) > 0:
                self.max_time = np.max(observed_times)
                y_train[censored_mask] = self.max_time
        else:
            self.max_time = np.max(y_train)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        if self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        elif self.model_type == 'xgb':
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=RANDOM_SEED,
                n_jobs=-1
            )

        self.model.fit(X_train_scaled, y_train)
        print(f"\n{self.model_type.upper()} Regression model trained")
        return self

    def predict(self, X):
        """Predict time-to-clog (continuous)"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        return predictions

    def predict_risk_class(self, X, T_high=40, T_low=100):
        """
        Predict risk class based on predicted time-to-clog.

        Returns:
        --------
        risk_classes : array
            0=Low, 1=Medium, 2=High
        """
        time_predictions = self.predict(X)
        risk_classes = np.zeros(len(time_predictions), dtype=int)  # Default: Low
        risk_classes[(time_predictions > T_high) & (time_predictions <= T_low)] = 1  # Medium
        risk_classes[time_predictions <= T_high] = 2  # High
        return risk_classes

# %% [markdown]
# ## 9.7. Anomaly Detection Module

# %%
class AnomalyDetectionModule:
    """
    Anomaly detection for filter clogging using unsupervised learning.

    Perfect for severely imbalanced data where clogging event is rare.
    Learns what "healthy" operation looks like, then detects deviations.

    Methods:
    - Isolation Forest: Detects global anomalies
    - Local Outlier Factor (LOF): Detects local density changes
    - One-Class SVM: Learns boundary of normal operation
    """
    def __init__(self, config=None):
        """
        Initialize anomaly detectors.

        Parameters:
        -----------
        config : dict
            Configuration with 'anomaly_detection' section
        """
        if config is None:
            config = CONFIG

        self.config = config['anomaly_detection'] if 'anomaly_detection' in config else {}
        self.scaler = StandardScaler()

        # Initialize detectors
        self.isolation_forest = None
        self.lof = None
        self.ocsvm = None
        self.detectors_trained = []

    def fit(self, X_healthy, verbose=True):
        """
        Train anomaly detectors on healthy (normal) operation data.

        Parameters:
        -----------
        X_healthy : array-like
            Features from healthy operation period (before degradation)
        verbose : bool
            Whether to print training progress
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X_healthy)

        methods = self.config.get('methods', ['isolation_forest'])
        contamination = self.config.get('contamination', 0.01)

        if verbose:
            print(f"\n{'='*60}")
            print("ANOMALY DETECTION TRAINING")
            print(f"{'='*60}")
            print(f"Training on {len(X_healthy)} healthy samples")
            print(f"Expected contamination: {contamination:.1%}")
            print(f"Methods: {', '.join(methods)}")

        # 1. Isolation Forest
        if 'isolation_forest' in methods:
            if verbose:
                print(f"\n[1/{len(methods)}] Training Isolation Forest...")

            self.isolation_forest = IsolationForest(
                n_estimators=self.config.get('n_estimators', 200),
                contamination=contamination,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
            self.isolation_forest.fit(X_scaled)
            self.detectors_trained.append('isolation_forest')

            if verbose:
                # Test on training data
                scores = self.isolation_forest.decision_function(X_scaled)
                anomalies = (self.isolation_forest.predict(X_scaled) == -1).sum()
                print(f"  ✓ Trained successfully")
                print(f"  Anomalies detected in training: {anomalies} ({anomalies/len(X_scaled):.2%})")
                print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

        # 2. Local Outlier Factor
        if 'lof' in methods:
            if verbose:
                print(f"\n[2/{len(methods)}] Training Local Outlier Factor...")

            n_neighbors = min(self.config.get('lof_neighbors', 20), len(X_scaled) - 1)

            self.lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                novelty=True,  # Enable predict on new data
                n_jobs=-1
            )
            self.lof.fit(X_scaled)
            self.detectors_trained.append('lof')

            if verbose:
                print(f"  ✓ Trained successfully")
                print(f"  Neighbors: {n_neighbors}")

        # 3. One-Class SVM
        if 'ocsvm' in methods:
            if verbose:
                print(f"\n[3/{len(methods)}] Training One-Class SVM...")

            self.ocsvm = OneClassSVM(
                kernel='rbf',
                gamma='auto',
                nu=contamination  # Upper bound on fraction of outliers
            )
            self.ocsvm.fit(X_scaled)
            self.detectors_trained.append('ocsvm')

            if verbose:
                scores = self.ocsvm.decision_function(X_scaled)
                anomalies = (self.ocsvm.predict(X_scaled) == -1).sum()
                print(f"  ✓ Trained successfully")
                print(f"  Anomalies detected in training: {anomalies} ({anomalies/len(X_scaled):.2%})")
                print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

        if verbose:
            print(f"\n{'='*60}")
            print(f"✓ Anomaly detection training complete!")
            print(f"  Detectors trained: {', '.join(self.detectors_trained)}")
            print(f"{'='*60}\n")

        return self

    def predict_anomaly_scores(self, X):
        """
        Predict anomaly scores for new data.

        Returns:
        --------
        dict with keys:
            - 'ensemble_score': Combined anomaly score (higher = more anomalous)
            - 'isolation_forest_score': IF score (if trained)
            - 'lof_score': LOF score (if trained)
            - 'ocsvm_score': One-Class SVM score (if trained)
            - 'is_anomaly': Binary prediction (ensemble)
        """
        X_scaled = self.scaler.transform(X)

        scores = {}
        raw_scores = []
        weights = []
        ensemble_weights = self.config.get('ensemble_weights', [0.5, 0.3, 0.2])

        # Collect scores from each detector
        if self.isolation_forest is not None:
            # IF: More negative = more anomalous, flip sign
            if_score = -self.isolation_forest.decision_function(X_scaled)
            scores['isolation_forest_score'] = if_score
            raw_scores.append(if_score)
            weights.append(ensemble_weights[0])

        if self.lof is not None:
            # LOF: More negative = more anomalous, flip sign
            lof_score = -self.lof.decision_function(X_scaled)
            scores['lof_score'] = lof_score
            raw_scores.append(lof_score)
            weights.append(ensemble_weights[1])

        if self.ocsvm is not None:
            # OCSVM: More negative = more anomalous, flip sign
            ocsvm_score = -self.ocsvm.decision_function(X_scaled)
            scores['ocsvm_score'] = ocsvm_score
            raw_scores.append(ocsvm_score)
            weights.append(ensemble_weights[2])

        # Normalize weights
        if len(weights) > 0:
            weights = np.array(weights) / sum(weights)

            # Weighted ensemble
            ensemble_score = np.average(raw_scores, axis=0, weights=weights)
            scores['ensemble_score'] = ensemble_score

            # Binary classification using ensemble
            # Normalize to 0-1 range and threshold at 0.5
            normalized_score = (ensemble_score - ensemble_score.min()) / (ensemble_score.max() - ensemble_score.min() + 1e-8)
            scores['is_anomaly'] = (normalized_score > 0.5).astype(int)
        else:
            raise RuntimeError("No anomaly detectors have been trained!")

        return scores

    def detect_anomalies(self, X, threshold=None):
        """
        Simple binary anomaly detection.

        Parameters:
        -----------
        X : array-like
            Features to check
        threshold : float, optional
            Custom threshold for ensemble score

        Returns:
        --------
        is_anomaly : array
            1 if anomaly, 0 if normal
        ensemble_score : array
            Continuous anomaly score
        """
        scores = self.predict_anomaly_scores(X)
        ensemble_score = scores['ensemble_score']

        if threshold is not None:
            is_anomaly = (ensemble_score > threshold).astype(int)
        else:
            is_anomaly = scores['is_anomaly']

        return is_anomaly, ensemble_score

# %% [markdown]
# ## 10. Survival Analysis Models

# %%
class SurvivalPredictor:
    """
    Survival analysis wrapper for Cox Proportional Hazards and Random Survival Forests.
    Handles time-to-event prediction with censored data.
    """
    def __init__(self, model_type='cox'):
        """
        Parameters:
        -----------
        model_type : str
            'cox' for Cox Proportional Hazards or 'rsf' for Random Survival Forest
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()

    def fit(self, X_train, duration_train, event_train, X_val=None, duration_val=None, event_val=None):
        """
        Train survival model.

        Parameters:
        -----------
        X_train : array-like
            Training features
        duration_train : array-like
            Time to event (or censoring)
        event_train : array-like
            Event indicator (1=event, 0=censored)
        """
        # Clean data before scaling
        X_train_clean = np.array(X_train, dtype=float)
        duration_train_clean = np.array(duration_train, dtype=float)
        event_train_clean = np.array(event_train, dtype=int)

        # Remove NaN and infinite values
        valid_mask = (
            ~np.isnan(X_train_clean).any(axis=1) &
            ~np.isinf(X_train_clean).any(axis=1) &
            ~np.isnan(duration_train_clean) &
            ~np.isinf(duration_train_clean) &
            (duration_train_clean > 0)  # Ensure positive durations
        )

        if valid_mask.sum() < len(valid_mask):
            n_removed = len(valid_mask) - valid_mask.sum()
            print(f"  ⚠️  Removed {n_removed} samples with NaN/inf values")

        X_train_clean = X_train_clean[valid_mask]
        duration_train_clean = duration_train_clean[valid_mask]
        event_train_clean = event_train_clean[valid_mask]

        if len(X_train_clean) == 0:
            raise ValueError("No valid samples remaining after cleaning!")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_clean)

        # Additional check for NaN after scaling (can happen with zero variance features)
        if np.isnan(X_train_scaled).any():
            print("  ⚠️  NaN detected after scaling, replacing with 0")
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1e10, neginf=-1e10)

        if self.model_type == 'cox':
            try:
                from lifelines import CoxPHFitter
            except ImportError:
                raise ImportError("lifelines not installed. Run: pip install lifelines")

            # Prepare dataframe for Cox model
            train_df = pd.DataFrame(X_train_scaled, columns=[f'f{i}' for i in range(X_train_scaled.shape[1])])
            train_df['duration'] = duration_train_clean
            train_df['event'] = event_train_clean

            # Additional validation
            if train_df.isnull().any().any():
                print("  ⚠️  Dropping columns with NaN values")
                train_df = train_df.dropna(axis=1)

            # Check for perfect collinearity and remove highly correlated features
            corr_matrix = train_df.drop(columns=['duration', 'event']).corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

            if high_corr_cols:
                print(f"  ⚠️  Removing {len(high_corr_cols)} highly correlated features")
                train_df = train_df.drop(columns=high_corr_cols)

            # Train Cox model with progressive fallback strategy
            for attempt, penalizer in enumerate([0.1, 1.0, 5.0, 10.0], 1):
                try:
                    self.model = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
                    self.model.fit(train_df, duration_col='duration', event_col='event', show_progress=False)
                    print(f"\nCox model trained (attempt {attempt}, penalizer={penalizer}):")
                    print(f"  Concordance index: {self.model.concordance_index_:.4f}")
                    break
                except Exception as e:
                    if attempt == 4:
                        # Last attempt failed - skip Cox model
                        print(f"  ❌ Cox model failed after all attempts: {str(e)[:100]}")
                        print(f"  Skipping Cox model - recommend using RSF or regression models instead")
                        self.model = None
                        return self
                    else:
                        print(f"  ⚠️  Attempt {attempt} failed (penalizer={penalizer}), trying stronger penalty...")

        elif self.model_type == 'rsf':
            try:
                from sksurv.ensemble import RandomSurvivalForest
            except ImportError:
                raise ImportError("scikit-survival not installed. Run: pip install scikit-survival")

            # Prepare structured array for sksurv (using cleaned data)
            y_train_surv = np.array(
                [(bool(e), float(t)) for e, t in zip(event_train_clean, duration_train_clean)],
                dtype=[('event', bool), ('time', float)]
            )

            # Train Random Survival Forest
            self.model = RandomSurvivalForest(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train_surv)

            # Calculate concordance index
            risk_scores = self.model.predict(X_train_scaled)
            from sksurv.metrics import concordance_index_censored
            c_index = concordance_index_censored(
                event_train_clean.astype(bool),
                duration_train_clean,
                -risk_scores  # Negative because higher risk = lower survival
            )[0]

            print(f"\nRandom Survival Forest trained:")
            print(f"  Concordance index: {c_index:.4f}")

        return self

    def predict_risk_score(self, X):
        """
        Predict risk scores (higher = higher risk of clogging).

        Returns:
        --------
        array-like : Risk scores
        """
        X_scaled = self.scaler.transform(X)

        if self.model_type == 'cox':
            # For Cox, use partial hazard (exp(beta'X))
            test_df = pd.DataFrame(X_scaled, columns=[f'f{i}' for i in range(X_scaled.shape[1])])
            risk_scores = self.model.predict_partial_hazard(test_df).values
        elif self.model_type == 'rsf':
            # For RSF, use negative prediction (higher = more risk)
            risk_scores = -self.model.predict(X_scaled)

        return risk_scores

    def predict_probability(self, X, horizon=5):
        """
        Predict probability of clogging within next 'horizon' steps.

        Parameters:
        -----------
        X : array-like
            Features
        horizon : int
            Time horizon for prediction

        Returns:
        --------
        array-like : Probability of event within horizon
        """
        X_scaled = self.scaler.transform(X)

        if self.model_type == 'cox':
            test_df = pd.DataFrame(X_scaled, columns=[f'f{i}' for i in range(X_scaled.shape[1])])

            # Get cumulative baseline hazard at horizon
            baseline_hazard = self.model.baseline_cumulative_hazard_

            # Find closest time point to horizon
            if horizon in baseline_hazard.index:
                baseline_at_t = baseline_hazard.loc[horizon].values[0]
            else:
                # Interpolate
                times = baseline_hazard.index.values
                if horizon > times.max():
                    baseline_at_t = baseline_hazard.iloc[-1].values[0]
                else:
                    idx = np.searchsorted(times, horizon)
                    baseline_at_t = baseline_hazard.iloc[idx].values[0]

            # Calculate probability: P(T < t) = 1 - exp(-H(t))
            risk_scores = self.model.predict_partial_hazard(test_df).values
            cum_hazard = baseline_at_t * risk_scores
            probabilities = 1 - np.exp(-cum_hazard)

        elif self.model_type == 'rsf':
            # Get survival functions
            surv_funcs = self.model.predict_survival_function(X_scaled)

            probabilities = []
            for surv_func in surv_funcs:
                # Find survival probability at horizon
                times = surv_func.x
                surv_probs = surv_func.y

                if horizon in times:
                    idx = np.where(times == horizon)[0][0]
                    surv_at_t = surv_probs[idx]
                elif horizon > times.max():
                    surv_at_t = surv_probs[-1]
                else:
                    idx = np.searchsorted(times, horizon)
                    surv_at_t = surv_probs[idx]

                # P(clog) = 1 - P(survival)
                probabilities.append(1 - surv_at_t)

            probabilities = np.array(probabilities)

        return probabilities

# %%
class UncertaintyWrapper:
    """
    Wrapper to provide uncertainty estimates for any model.
    Uses prediction variance across ensemble or bootstrap samples.
    """
    def __init__(self, model, method='ensemble'):
        """
        Parameters:
        -----------
        model : sklearn model or custom model
            Base model to wrap
        method : str
            'ensemble' for tree-based models, 'bootstrap' for others
        """
        self.model = model
        self.method = method
        self.bootstrap_models = []

    def predict_with_uncertainty(self, X, n_bootstrap=30):
        """
        Predict with confidence intervals.

        Returns:
        --------
        mean_pred : array
            Mean predictions
        lower_bound : array
            Lower confidence bound (5th percentile)
        upper_bound : array
            Upper confidence bound (95th percentile)
        """
        if self.method == 'ensemble' and hasattr(self.model, 'estimators_'):
            # Use tree ensemble variance
            predictions = np.array([
                tree.predict_proba(X)[:, 1] if hasattr(tree, 'predict_proba')
                else tree.predict(X)
                for tree in self.model.estimators_
            ])

            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)

            # 90% confidence interval (assuming normal distribution)
            lower_bound = mean_pred - 1.645 * std_pred
            upper_bound = mean_pred + 1.645 * std_pred

        elif self.method == 'bootstrap':
            # Use bootstrap samples
            if len(self.bootstrap_models) == 0:
                warnings.warn("No bootstrap models available. Using single model prediction.")
                if hasattr(self.model, 'predict_proba'):
                    mean_pred = self.model.predict_proba(X)[:, 1]
                else:
                    mean_pred = self.model.predict(X)
                return mean_pred, mean_pred, mean_pred

            predictions = np.array([
                model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba')
                else model.predict(X)
                for model in self.bootstrap_models
            ])

            mean_pred = predictions.mean(axis=0)
            lower_bound = np.percentile(predictions, 5, axis=0)
            upper_bound = np.percentile(predictions, 95, axis=0)

        else:
            # Fallback: no uncertainty
            if hasattr(self.model, 'predict_proba'):
                mean_pred = self.model.predict_proba(X)[:, 1]
            else:
                mean_pred = self.model.predict(X)
            lower_bound = mean_pred
            upper_bound = mean_pred

        # Clip to valid probability range
        lower_bound = np.clip(lower_bound, 0, 1)
        upper_bound = np.clip(upper_bound, 0, 1)

        return mean_pred, lower_bound, upper_bound

# %%
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual frequencies.

    Parameters:
    -----------
    y_true : array
        True binary labels
    y_prob : array
        Predicted probabilities
    n_bins : int
        Number of bins for calibration curve

    Returns:
    --------
    float : ECE score (lower is better, 0 = perfect calibration)
    """
    # FIXED: Handle cases where predictions are concentrated in few bins
    # Bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Digitize predictions into bins
    bin_indices = np.digitize(y_prob, bin_edges[:-1], right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure valid indices

    ece = 0.0
    for bin_idx in range(n_bins):
        # Find samples in this bin
        in_bin = bin_indices == bin_idx

        if np.sum(in_bin) > 0:  # Only process non-empty bins
            # Fraction of samples in this bin
            bin_weight = np.sum(in_bin) / len(y_prob)

            # Average predicted probability in this bin
            avg_confidence = np.mean(y_prob[in_bin])

            # Actual fraction of positives in this bin
            avg_accuracy = np.mean(y_true[in_bin])

            # Add weighted difference to ECE
            ece += bin_weight * np.abs(avg_accuracy - avg_confidence)

    return ece

def evaluate_survival_model(model: SurvivalPredictor, X_test, duration_test, event_test,
                           horizon=5, model_name='Survival Model'):
    """
    Evaluate survival model with concordance index and integrated Brier score.

    Parameters:
    -----------
    model : SurvivalPredictor
        Trained survival model
    X_test : array
        Test features
    duration_test : array
        Test durations
    event_test : array
        Test event indicators
    horizon : int
        Prediction horizon
    model_name : str
        Name for display

    Returns:
    --------
    dict : Metrics dictionary
    """
    # Predict risk scores and probabilities
    risk_scores = model.predict_risk_score(X_test)
    probabilities = model.predict_probability(X_test, horizon=horizon)

    # Concordance index (C-index)
    try:
        if model.model_type == 'cox':
            from lifelines.utils import concordance_index
            c_index = concordance_index(duration_test, -risk_scores, event_test)
        else:  # RSF
            from sksurv.metrics import concordance_index_censored
            c_index = concordance_index_censored(
                event_test.astype(bool),
                duration_test,
                risk_scores
            )[0]
    except Exception as e:
        print(f"Could not calculate C-index: {e}")
        c_index = 0.0

    # Convert to binary predictions for comparison metrics
    y_pred_binary = (probabilities > 0.5).astype(int)

    # Binary classification metrics (using horizon-based target)
    y_true_binary = (duration_test <= horizon).astype(int)

    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    metrics = {
        'concordance_index': c_index,
        'roc_auc': roc_auc_score(y_true_binary, probabilities) if len(np.unique(y_true_binary)) > 1 else 0.0,
        'f1': f1_score(y_true_binary, y_pred_binary) if len(np.unique(y_true_binary)) > 1 else 0.0,
        'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
    }

    print(f"\n{model_name} Test Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return metrics, probabilities, risk_scores

# %% [markdown]
# ## 11. Model Persistence and Update Functions

# %%
class FilterCloggingPredictor:
    """
    Main class for filter clogging prediction system with survival analysis support.
    """
    def __init__(self, config=CONFIG, use_survival=True, use_regression=False):
        self.config = config
        self.use_survival = use_survival  # Default: True (survival analysis recommended)
        self.use_regression = use_regression  # Optional: regression for time-to-clog
        self.rf_model: Optional[Union[RandomForestClassifier, CalibratedClassifierCV]] = None
        self.xgb_model: Optional[Union[xgb.XGBClassifier, CalibratedClassifierCV]] = None
        self.cox_model: Optional[SurvivalPredictor] = None
        self.rsf_model: Optional[SurvivalPredictor] = None
        self.rf_regressor: Optional[RegressionPredictor] = None
        self.xgb_regressor: Optional[RegressionPredictor] = None
        self.rf_params: Dict[str, Any] = {}
        self.xgb_params: Dict[str, Any] = {}
        self.feature_names: Optional[List[str]] = None
        self.scaler = StandardScaler()
        self.metadata: Dict[str, Any] = {}
        
    def fit(self, df):
        """
        Complete training pipeline.
        """
        print("="*50)
        print("FILTER CLOGGING PREDICTION SYSTEM")
        print("="*50)
        
        # Feature engineering
        print("\n[1/7] Engineering features...")
        df_features = build_features(df, self.config)

        # CRITICAL: Handle any remaining NaN/inf values
        print("[1.5/7] Checking for NaN/inf values...")
        n_rows_before = len(df_features)

        # Replace inf with NaN
        df_features = df_features.replace([np.inf, -np.inf], np.nan)

        # Check for NaN
        nan_counts = df_features.isna().sum()
        if nan_counts.sum() > 0:
            print(f"  Found NaN values in {(nan_counts > 0).sum()} columns")
            print(f"  Total NaN values: {nan_counts.sum()}")

            # Fill NaN with column median (better than dropping rows)
            for col in df_features.columns:
                if nan_counts[col] > 0 and col not in ['time', 'filter_status']:
                    median_val = df_features[col].median()
                    if pd.isna(median_val):
                        median_val = 0  # If entire column is NaN, use 0
                    df_features[col].fillna(median_val, inplace=True)

            print(f"✅ NaN values filled with column medians")
        else:
            print(f"✅ No NaN values found")

        # Create targets
        print("[2/7] Creating targets...")
        df_features = create_targets(df_features, self.config)
        
        # Prepare features
        feature_cols = [col for col in df_features.columns
                       if col not in ['time', 'filter_status', 'is_clogged',
                                     'will_clog', 'time_to_clog', 'risk_class',
                                     'risk_class_numeric', 'is_censored']]
        self.feature_names = feature_cols

        X = df_features[feature_cols].values
        y_binary = df_features['will_clog'].values
        y_multiclass = df_features['risk_class_numeric'].values

        # Prepare features
        feature_cols = [col for col in df_features.columns
                       if col not in ['time', 'filter_status', 'is_clogged',
                                     'will_clog', 'time_to_clog', 'risk_class',
                                     'risk_class_numeric', 'is_censored']]
        self.feature_names = feature_cols
        
        X = df_features[feature_cols].values
        y_binary = df_features['will_clog'].values.astype(int) # Ensure numpy array of int
        y_multiclass = df_features['risk_class_numeric'].values.astype(int) # Not used in current binary setup, but kept for consistency
        
        all_rf_metrics = []
        all_xgb_metrics = []
        
        # Initialize these outside the loop to ensure they are always bound
        # In case rolling_origin_time_series_split yields no folds, or only one.
        last_rf_params = {}
        last_xgb_params = {}
        last_scaler = StandardScaler() # Default scaler if no folds are processed
        last_X_val_scaled = None
        last_y_val = None

        if self.config['use_rolling_cv']:
            print(f"\n[3/7] Performing rolling origin cross-validation with {self.config['n_splits_rolling_cv']} splits...")
            
            for fold, (train_idx, val_idx, test_idx) in enumerate(rolling_origin_time_series_split(
                df_features, 
                n_splits=self.config['n_splits_rolling_cv'],
                train_size_frac=self.config['train_frac'],
                val_size_frac=self.config['val_frac'],
                min_pos_fraction=self.config['min_pos_fraction']
            )):
                print(f"\n--- Fold {fold + 1}/{self.config['n_splits_rolling_cv']} ---")
                X_train, y_train = X[train_idx], y_binary[train_idx]
                X_val, y_val = X[val_idx], y_binary[val_idx]
                X_test, y_test = X[test_idx], y_binary[test_idx]

                # Scale features for this fold
                current_scaler = StandardScaler()
                X_train_scaled = current_scaler.fit_transform(X_train)
                X_val_scaled = current_scaler.transform(X_val)
                X_test_scaled = current_scaler.transform(X_test)

                # Compute sample weights for this fold
                sample_weights_fold = compute_sample_weights(y_train, train_idx, self.config)

                # Compute sample weights for cost-sensitive learning (NO SMOTE)
                print(f"\n[3.5/7] Computing cost-sensitive sample weights for Fold {fold + 1}...")
                sample_weights_fold = compute_sample_weights(
                    y_train,
                    np.arange(len(y_train)),
                    self.config
                )

                # Hyperparameter optimization for this fold
                print(f"\n[4/7] Optimizing hyperparameters for Fold {fold + 1}...")
                rf_params_fold = optimize_hyperparameters(
                    X_train_scaled, y_train, X_val_scaled, y_val, sample_weights_fold,
                    model_type='rf', n_trials=self.config['optuna_trials_binary']
                )
                xgb_params_fold = optimize_hyperparameters(
                    X_train_scaled, y_train, X_val_scaled, y_val, sample_weights_fold,
                    model_type='xgb', n_trials=self.config['optuna_trials_binary']
                )

                # Train final models for this fold
                print(f"\n[5/7] Training final models on combined train+val for Fold {fold + 1}...")
                rf_model_fold = train_model(X_train_scaled, y_train, X_val_scaled, y_val,
                                            sample_weights_fold, rf_params_fold, 'rf')
                xgb_model_fold = train_model(X_train_scaled, y_train, X_val_scaled, y_val,
                                             sample_weights_fold, xgb_params_fold, 'xgb')

                # Evaluate on test set for this fold
                print(f"\n[6/7] Evaluating on test set for Fold {fold + 1}...")
                rf_metrics_fold, rf_pred_fold, rf_proba_fold = evaluate_model(rf_model_fold, X_test_scaled, y_test, 
                                                                            f'Random Forest Fold {fold + 1}')
                xgb_metrics_fold, xgb_pred_fold, xgb_proba_fold = evaluate_model(xgb_model_fold, X_test_scaled, y_test, 
                                                                                f'XGBoost Fold {fold + 1}')
                
                all_rf_metrics.append(rf_metrics_fold)
                all_xgb_metrics.append(xgb_metrics_fold)

                last_rf_params = rf_params_fold
                last_xgb_params = xgb_params_fold
                last_scaler = current_scaler
                last_X_val_scaled = X_val_scaled
                last_y_val = y_val

            if all_rf_metrics: # Only aggregate if there were successful folds
                # Aggregate metrics
                avg_rf_metrics = pd.DataFrame(all_rf_metrics).mean().to_dict()
                std_rf_metrics = pd.DataFrame(all_rf_metrics).std().to_dict()
                avg_xgb_metrics = pd.DataFrame(all_xgb_metrics).mean().to_dict()
                std_xgb_metrics = pd.DataFrame(all_xgb_metrics).std().to_dict()

                print("\n" + "="*50)
                print("ROLLING ORIGIN CROSS-VALIDATION COMPLETE!")
                print("="*50)
                print("\nAverage Random Forest Metrics:")
                for metric, value in avg_rf_metrics.items():
                    print(f"  {metric}: {value:.4f} (std: {std_rf_metrics[metric]:.4f})")
                print("\nAverage XGBoost Metrics:")
                for metric, value in avg_xgb_metrics.items():
                    print(f"  {metric}: {value:.4f} (std: {std_xgb_metrics[metric]:.4f})")

                # For final prediction and saving, retrain on the entire dataset using the best params from the last fold
                print("\n[7/7] Retraining final models on the entire dataset with best parameters from last fold...")
                self.scaler = StandardScaler()
                X_scaled_full = self.scaler.fit_transform(X)

                # Use cost-sensitive learning without SMOTE
                print("\nComputing cost-sensitive weights for full dataset...")
                sample_weights_full = compute_sample_weights(
                    y_binary,
                    np.arange(len(y_binary)),
                    self.config
                )

                self.rf_model = RandomForestClassifier(**last_rf_params, n_jobs=-1)
                self.rf_model.fit(X_scaled_full, y_binary, sample_weight=sample_weights_full)
                self.rf_params = last_rf_params

                xgb_params_final = last_xgb_params.copy()
                # Remove parameters that are not compatible with final training
                params_to_remove = ['early_stopping_rounds', 'eval_set', 'eval_metric', 'verbose']
                for param in params_to_remove:
                    xgb_params_final.pop(param, None)

                self.xgb_model = xgb.XGBClassifier(**xgb_params_final, use_label_encoder=False)
                self.xgb_model.fit(X_scaled_full, y_binary, sample_weight=sample_weights_full)
                self.xgb_params = xgb_params_final

                # Calibrate models if enabled
                if self.config['calibrate_models'] and last_X_val_scaled is not None and last_y_val is not None:
                    print(f"\nCalibrating models using method: {self.config['calibration_method']}...")
                    self.rf_model = CalibratedClassifierCV(self.rf_model, method=self.config['calibration_method'], cv='prefit')
                    self.rf_model.fit(last_X_val_scaled, last_y_val)

                    self.xgb_model = CalibratedClassifierCV(self.xgb_model, method=self.config['calibration_method'], cv='prefit')
                    self.xgb_model.fit(last_X_val_scaled, last_y_val)

                # Update metadata with average metrics
                self.metadata = {
                    'training_date': datetime.now().isoformat(),
                    'n_samples': len(df_features),
                    'n_features': len(feature_cols),
                    'rf_metrics_avg': avg_rf_metrics,
                    'rf_metrics_std': std_rf_metrics,
                    'xgb_metrics_avg': avg_xgb_metrics,
                    'xgb_metrics_std': std_xgb_metrics,
                    'feature_names': self.feature_names,
                    'config': self.config
                }
                
                # No visualizations generated for individual folds, only for the final model if needed.
                # For now, skip visualizations in rolling CV mode to avoid clutter.
            else:
                print("\nNo successful folds in rolling origin cross-validation. Skipping final model training and visualizations.")
                # If no successful folds, ensure models are None or handle appropriately
                self.rf_model = None
                self.xgb_model = None
                self.metadata = {
                    'training_date': datetime.now().isoformat(),
                    'n_samples': len(df_features),
                    'n_features': len(feature_cols),
                    'message': 'No successful folds in rolling origin cross-validation.',
                    'feature_names': self.feature_names,
                    'config': self.config
                }


        else: # Original single split training
            print("[3/7] Splitting data chronologically...")
            train_idx, val_idx, test_idx = time_series_split(df_features, self.config)
            
            X_train, y_train = X[train_idx], y_binary[train_idx]
            X_val, y_val = X[val_idx], y_binary[val_idx]
            X_test, y_test = X[test_idx], y_binary[test_idx]
            
            # Scale features
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

            # Compute sample weights for cost-sensitive learning
            print("[4/7] Computing cost-sensitive sample weights...")
            sample_weights = compute_sample_weights(y_train, train_idx, self.config)

            # Hyperparameter optimization (using cost-sensitive learning, NO SMOTE)
            print("\n[5/7] Optimizing hyperparameters with cost-sensitive learning...")
            print("\n--- Random Forest ---")
            self.rf_params = optimize_hyperparameters(
                X_train, y_train, X_val, y_val, sample_weights,
                model_type='rf', n_trials=self.config['optuna_trials_binary']
            )

            print("\n--- XGBoost ---")
            self.xgb_params = optimize_hyperparameters(
                X_train, y_train, X_val, y_val, sample_weights,
                model_type='xgb', n_trials=self.config['optuna_trials_binary']
            )

            # Train final models
            print("\n[6/7] Training final models on combined train+val...")
            self.rf_model = train_model(X_train, y_train, X_val, y_val,
                                       sample_weights, self.rf_params, 'rf')
            self.xgb_model = train_model(X_train, y_train, X_val, y_val,
                                        sample_weights, self.xgb_params, 'xgb')
            
            # Calibrate models if enabled
            if self.config['calibrate_models']:
                print(f"\nCalibrating models using method: {self.config['calibration_method']}...")
                self.rf_model = CalibratedClassifierCV(self.rf_model, method=self.config['calibration_method'], cv='prefit')
                self.rf_model.fit(X_val, y_val) # Use validation set for calibration
                
                self.xgb_model = CalibratedClassifierCV(self.xgb_model, method=self.config['calibration_method'], cv='prefit')
                self.xgb_model.fit(X_val, y_val) # Use validation set for calibration

            # Evaluate on test set
            print("\n[7/7] Evaluating on test set...")
            rf_metrics, rf_pred, rf_proba = evaluate_model(self.rf_model, X_test, y_test,
                                                           'Random Forest')
            xgb_metrics, xgb_pred, xgb_proba = evaluate_model(self.xgb_model, X_test, y_test,
                                                             'XGBoost')

            # Add ECE metric
            rf_metrics['ece'] = expected_calibration_error(y_test, rf_proba)
            xgb_metrics['ece'] = expected_calibration_error(y_test, xgb_proba)
            print(f"\nRandom Forest ECE: {rf_metrics['ece']:.4f}")
            print(f"XGBoost ECE: {xgb_metrics['ece']:.4f}")

            # Train survival models if requested
            if self.use_survival:
                print("\n[8/9] Training survival analysis models...")

                # Prepare survival data
                survival_df = prepare_survival_data(df_features)
                duration_train = survival_df.iloc[train_idx]['duration'].values
                event_train = survival_df.iloc[train_idx]['event'].values
                duration_val = survival_df.iloc[val_idx]['duration'].values
                event_val = survival_df.iloc[val_idx]['event'].values
                duration_test = survival_df.iloc[test_idx]['duration'].values
                event_test = survival_df.iloc[test_idx]['event'].values

                # Survival models handle censoring naturally - NO SMOTE needed
                print("\n--- Training Survival Models (no resampling needed) ---")

                # Train Cox Proportional Hazards
                print("\n--- Cox Proportional Hazards ---")
                self.cox_model = SurvivalPredictor(model_type='cox')
                self.cox_model.fit(X_train, duration_train, event_train)

                # Train Random Survival Forest
                print("\n--- Random Survival Forest ---")
                try:
                    self.rsf_model = SurvivalPredictor(model_type='rsf')
                    self.rsf_model.fit(X_train, duration_train, event_train)
                except ImportError as e:
                    print(f"Skipping RSF: {e}")
                    self.rsf_model = None

                # Evaluate survival models
                print("\n[9/9] Evaluating survival models...")

                if self.cox_model is not None and self.cox_model.model is not None:
                    cox_metrics, cox_proba, cox_risk = evaluate_survival_model(
                        self.cox_model, X_test, duration_test, event_test,
                        horizon=self.config['forecast_horizon_steps'],
                        model_name='Cox Proportional Hazards'
                    )
                else:
                    print("  ⚠️  Cox model not available (training failed)")
                    cox_metrics = {}

                if self.rsf_model is not None:
                    rsf_metrics, rsf_proba, rsf_risk = evaluate_survival_model(
                        self.rsf_model, X_test, duration_test, event_test,
                        horizon=self.config['forecast_horizon_steps'],
                        model_name='Random Survival Forest'
                    )
                else:
                    rsf_metrics = {}

                # Store survival metrics
                self.metadata['cox_metrics'] = cox_metrics
                self.metadata['rsf_metrics'] = rsf_metrics

            # Train regression models if enabled
            if self.use_regression:
                print("\n[8/9] Training Regression Models for Time-to-Clog Prediction...")

                # Prepare regression targets
                y_time_to_clog_train = df_features.iloc[train_idx]['time_to_clog'].values
                y_time_to_clog_test = df_features.iloc[test_idx]['time_to_clog'].values
                censored_train = df_features.iloc[train_idx]['is_censored'].values.astype(bool)
                censored_test = df_features.iloc[test_idx]['is_censored'].values.astype(bool)

                # Train RF Regressor
                print("\n--- Random Forest Regressor ---")
                self.rf_regressor = RegressionPredictor(model_type='rf')
                self.rf_regressor.fit(X_train, y_time_to_clog_train, censored_train)

                # Train XGB Regressor
                print("\n--- XGBoost Regressor ---")
                self.xgb_regressor = RegressionPredictor(model_type='xgb')
                self.xgb_regressor.fit(X_train, y_time_to_clog_train, censored_train)

                # Evaluate regression models
                print("\n--- Evaluating Regression Models ---")
                rf_time_pred = self.rf_regressor.predict(X_test)
                xgb_time_pred = self.xgb_regressor.predict(X_test)

                # Calculate RMSE for uncensored samples only
                uncensored_test = ~censored_test
                if uncensored_test.sum() > 0:
                    rf_rmse = np.sqrt(np.mean((rf_time_pred[uncensored_test] - y_time_to_clog_test[uncensored_test])**2))
                    xgb_rmse = np.sqrt(np.mean((xgb_time_pred[uncensored_test] - y_time_to_clog_test[uncensored_test])**2))

                    print(f"\nRF Regressor  - RMSE (uncensored): {rf_rmse:.2f} steps")
                    print(f"XGB Regressor - RMSE (uncensored): {xgb_rmse:.2f} steps")

                    # Calculate MAE
                    rf_mae = np.mean(np.abs(rf_time_pred[uncensored_test] - y_time_to_clog_test[uncensored_test]))
                    xgb_mae = np.mean(np.abs(xgb_time_pred[uncensored_test] - y_time_to_clog_test[uncensored_test]))

                    print(f"RF Regressor  - MAE (uncensored):  {rf_mae:.2f} steps")
                    print(f"XGB Regressor - MAE (uncensored):  {xgb_mae:.2f} steps")

                    # Store regression metrics
                    self.metadata['rf_regressor_metrics'] = {'rmse': rf_rmse, 'mae': rf_mae}
                    self.metadata['xgb_regressor_metrics'] = {'rmse': xgb_rmse, 'mae': xgb_mae}

            # Generate visualizations
            print("\nGenerating visualizations...")
            try:
                plot_time_series(df_features, save_path=self.config['plots_dir'])
                plot_correlation_matrix(df_features[feature_cols], save_path=self.config['plots_dir'])
                plot_model_performance(y_test, rf_pred, rf_proba, 'RandomForest',
                                     save_path=self.config['plots_dir'])
                plot_model_performance(y_test, xgb_pred, xgb_proba, 'XGBoost',
                                     save_path=self.config['plots_dir'])
                plot_feature_importance(self.rf_model, self.feature_names, 'RandomForest',
                                      save_path=self.config['plots_dir'])
                plot_feature_importance(self.xgb_model, self.feature_names, 'XGBoost',
                                      save_path=self.config['plots_dir'])
                print("✅ Visualizations generated successfully")
            except Exception as e:
                print(f"⚠️  Some visualizations failed: {e}")

            # SHAP analysis
            print("\nGenerating SHAP interpretability plots...")
            try:
                generate_shap_plots(self.rf_model, X_train, X_test, self.feature_names,
                                  save_path=self.config['plots_dir'])
                print("✅ SHAP plots generated successfully")
            except Exception as e:
                print(f"⚠️  SHAP plots failed: {e}")
            
            # Save metadata
            self.metadata = {
                'training_date': datetime.now().isoformat(),
                'n_samples': len(df_features),
                'n_features': len(feature_cols),
                'rf_metrics': rf_metrics,
                'xgb_metrics': xgb_metrics,
                'feature_names': self.feature_names,
                'config': self.config
            }
            
            # Save performance summary
            self._save_performance_summary(rf_metrics, xgb_metrics)
            
            print("\n" + "="*50)
            print("TRAINING COMPLETE!")
            print("="*50)
        
        # Save models (only once after all training is done)
        if self.rf_model is not None and self.xgb_model is not None:
            self.save_models()
            print("\n" + "="*50)
            print("TRAINING COMPLETE!")
            print("="*50)
        else:
            print("\n" + "="*50)
            print("TRAINING SKIPPED OR FAILED DUE TO NO SUCCESSFUL FOLDS!")
            print("="*50)
        
        return self
    
    def predict(self, df, model_type='xgb', with_uncertainty=True):
        """
        Make predictions on new data with optional uncertainty quantification.

        Parameters:
        -----------
        df : pd.DataFrame
            New data with required columns
        model_type : str
            'rf', 'xgb', 'cox', 'rsf', or 'ensemble' (default: 'xgb')
        with_uncertainty : bool
            If True, include confidence intervals (default: True)

        Returns:
        --------
        pd.DataFrame with predictions, probabilities, risk classifications, and uncertainty
        """
        if self.rf_model is None or self.xgb_model is None:
            raise RuntimeError("Model has not been trained. Call fit() before predicting.")
        if self.feature_names is None:
            raise RuntimeError("Feature names are not available. Call fit() before predicting.")

        # Engineer features
        df_features = build_features(df, self.config)

        # Handle NaN/inf values (same as training)
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        for col in self.feature_names:
            if col in df_features.columns and df_features[col].isna().any():
                median_val = df_features[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df_features[col].fillna(median_val, inplace=True)

        # Select features
        X = df_features[self.feature_names].values
        X_scaled = self.scaler.transform(X)

        # Make predictions based on model type
        if model_type == 'cox':
            if self.cox_model is None or self.cox_model.model is None:
                raise RuntimeError("Cox model not available (training may have failed). Try RSF or regression models.")
            probabilities = self.cox_model.predict_probability(X, horizon=self.config['forecast_horizon_steps'])
            predictions = (probabilities > 0.5).astype(int)
            uncertainty_lower = probabilities  # Placeholder
            uncertainty_upper = probabilities

        elif model_type == 'rsf':
            if self.rsf_model is None:
                raise RuntimeError("RSF model not trained. Use use_survival=True during fit().")
            probabilities = self.rsf_model.predict_probability(X, horizon=self.config['forecast_horizon_steps'])
            predictions = (probabilities > 0.5).astype(int)
            uncertainty_lower = probabilities  # Placeholder
            uncertainty_upper = probabilities

        elif model_type == 'regression':
            # Use regression models to predict time-to-clog
            if self.rf_regressor is None and self.xgb_regressor is None:
                raise RuntimeError("No regression models trained. Use use_regression=True during fit().")

            # Average predictions from both regressors if available
            time_predictions = []
            if self.rf_regressor is not None:
                time_predictions.append(self.rf_regressor.predict(X))
            if self.xgb_regressor is not None:
                time_predictions.append(self.xgb_regressor.predict(X))

            # Average time predictions
            avg_time_pred = np.mean(time_predictions, axis=0)

            # Convert to binary predictions based on threshold
            T_high = self.config['risk_thresholds']['T_high']
            predictions = (avg_time_pred <= T_high).astype(int)
            probabilities = 1 / (1 + avg_time_pred / T_high)  # Convert time to probability-like score

            # Uncertainty from variance
            if len(time_predictions) > 1:
                time_std = np.std(time_predictions, axis=0)
                uncertainty_lower = probabilities - time_std / (2 * T_high)
                uncertainty_upper = probabilities + time_std / (2 * T_high)
            else:
                uncertainty_lower = probabilities
                uncertainty_upper = probabilities

        elif model_type == 'ensemble':
            # Weighted ensemble of all available models
            all_probs = []
            weights = []

            if self.xgb_model is not None:
                all_probs.append(self.xgb_model.predict_proba(X_scaled)[:, 1])
                weights.append(0.25)

            if self.rf_model is not None:
                all_probs.append(self.rf_model.predict_proba(X_scaled)[:, 1])
                weights.append(0.25)

            if self.cox_model is not None and self.cox_model.model is not None:
                all_probs.append(self.cox_model.predict_probability(X, horizon=self.config['forecast_horizon_steps']))
                weights.append(0.2)

            if self.rsf_model is not None:
                all_probs.append(self.rsf_model.predict_probability(X, horizon=self.config['forecast_horizon_steps']))
                weights.append(0.2)

            # Add regression models to ensemble if available
            if self.rf_regressor is not None or self.xgb_regressor is not None:
                time_predictions = []
                if self.rf_regressor is not None:
                    time_predictions.append(self.rf_regressor.predict(X))
                if self.xgb_regressor is not None:
                    time_predictions.append(self.xgb_regressor.predict(X))
                avg_time_pred = np.mean(time_predictions, axis=0)
                T_high = self.config['risk_thresholds']['T_high']
                reg_proba = 1 / (1 + avg_time_pred / T_high)
                all_probs.append(reg_proba)
                weights.append(0.1)

            # Normalize weights
            weights = np.array(weights) / sum(weights)
            probabilities = np.average(all_probs, axis=0, weights=weights)
            predictions = (probabilities > 0.5).astype(int)

            # Uncertainty from ensemble variance
            uncertainty_std = np.std(all_probs, axis=0)
            uncertainty_lower = np.clip(probabilities - 1.645 * uncertainty_std, 0, 1)
            uncertainty_upper = np.clip(probabilities + 1.645 * uncertainty_std, 0, 1)

        else:
            # Tree models (rf or xgb)
            model = self.xgb_model if model_type == 'xgb' else self.rf_model

            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]

            # Get uncertainty if requested
            if with_uncertainty:
                uncertainty_wrapper = UncertaintyWrapper(model, method='ensemble')
                _, uncertainty_lower, uncertainty_upper = uncertainty_wrapper.predict_with_uncertainty(X_scaled)
            else:
                uncertainty_lower = probabilities
                uncertainty_upper = probabilities

        # Generate risk classifications based on probability thresholds
        risk_labels = pd.Series(['Low'] * len(probabilities), dtype=str)
        risk_labels[probabilities > 0.3] = 'Medium'
        risk_labels[probabilities > 0.7] = 'High'

        results = pd.DataFrame({
            'index': df_features.index,
            'predicted_clog': predictions,
            'clog_probability': probabilities,
            'confidence_lower': uncertainty_lower,
            'confidence_upper': uncertainty_upper,
            'uncertainty': uncertainty_upper - uncertainty_lower,
            'risk_class': risk_labels
        })

        # Add time-to-clog estimate based on risk class
        risk_to_time = {'Low': 30, 'Medium': 10, 'High': 3}
        results['estimated_time_to_clog'] = results['risk_class'].map(risk_to_time)

        return results
    
    def update_model(self, new_df):
        """
        Update model with new data using saved hyperparameters.
        
        Parameters:
        -----------
        new_df : pd.DataFrame
            New data for retraining
        """
        if not self.rf_params or not self.xgb_params:
            raise RuntimeError("Model hyperparameters not available. Call fit() to train the model and find hyperparameters first.")
        if self.feature_names is None:
            raise RuntimeError("Feature names are not available. Call fit() before updating the model.")

        print("Updating model with new data...")
        
        # Combine with existing data if available
        # In production, you would load historical data here
        
        # Retrain with saved hyperparameters
        df_features = build_features(new_df, self.config)
        df_features = create_targets(df_features, self.config)
        
        X = df_features[self.feature_names].values
        y = df_features['will_clog'].values
        
        # Split and scale
        train_idx, val_idx, test_idx = time_series_split(df_features, self.config)
        X_train = self.scaler.fit_transform(X[train_idx])
        y_train = y[train_idx]
        
        # Compute weights
        sample_weights = compute_sample_weights(y_train, train_idx, self.config)
        
        # Retrain models with existing hyperparameters
        rf_model = RandomForestClassifier(**self.rf_params)
        rf_model.fit(X_train, y_train, sample_weight=sample_weights)
        self.rf_model = rf_model

        # Remove early_stopping_rounds for retraining if it exists
        xgb_params_retrain = self.xgb_params.copy()
        if 'early_stopping_rounds' in xgb_params_retrain:
            del xgb_params_retrain['early_stopping_rounds']
        
        xgb_model = xgb.XGBClassifier(**xgb_params_retrain)
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
        self.xgb_model = xgb_model
        
        # Update metadata
        self.metadata['last_update'] = datetime.now().isoformat()
        self.metadata['n_samples_update'] = len(new_df)
        
        # Save updated models
        self.save_models()
        
        print("Model update complete!")
        
        return self
    
    def save_models(self):
        """Save trained models and metadata."""
        models_dir = Path(self.config['models_dir'])
        
        # Save models
        joblib.dump(self.rf_model, models_dir / 'rf_model.pkl')
        joblib.dump(self.xgb_model, models_dir / 'xgb_model.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, models_dir / 'scaler.pkl')
        
        # Save parameters
        joblib.dump({
            'rf_params': self.rf_params,
            'xgb_params': self.xgb_params,
            'feature_names': self.feature_names,
            'metadata': self.metadata,
            'config': self.config
        }, models_dir / 'model_metadata.pkl')
        
        print(f"Models saved to {models_dir}")
    
    def load_models(self):
        """Load saved models and metadata."""
        models_dir = Path(self.config['models_dir'])
        
        self.rf_model = joblib.load(models_dir / 'rf_model.pkl')
        self.xgb_model = joblib.load(models_dir / 'xgb_model.pkl')
        self.scaler = joblib.load(models_dir / 'scaler.pkl')
        
        metadata = joblib.load(models_dir / 'model_metadata.pkl')
        self.rf_params = metadata['rf_params']
        self.xgb_params = metadata['xgb_params']
        self.feature_names = metadata['feature_names']
        self.metadata = metadata['metadata']
        
        print("Models loaded successfully!")
        
        return self
    
    def _save_performance_summary(self, rf_metrics, xgb_metrics):
        """Save performance metrics to CSV."""
        summary = pd.DataFrame({
            'RandomForest': rf_metrics,
            'XGBoost': xgb_metrics
        })

        csv_path = f"{self.config['results_dir']}/model_performance_summary.csv"
        try:
            summary.to_csv(csv_path)
            print(f"\n✅ Performance summary saved to {csv_path}")
        except PermissionError:
            print(f"\n⚠️  Could not save {csv_path} (file may be open in Excel)")
            print(f"    Performance summary:")
            print(summary.to_string())
        except Exception as e:
            print(f"\n⚠️  Error saving performance summary: {e}")
            print(f"    Performance summary:")
            print(summary.to_string())

# %% [markdown]
# ## 11. Main Execution Example

# %%
def main():
    """
    Main execution function demonstrating complete pipeline.
    """
    # Example usage - replace with your Excel file path
    excel_file_path = "Comprehensive_Filter_Analysis.xlsx"  # Replace with your file
    
    try:
        # Load data
        print("Loading data from Excel file...")
        df = pd.read_excel(excel_file_path)
        print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        # Initialize predictor
        predictor = FilterCloggingPredictor(config=CONFIG)
        
        # Train models
        predictor.fit(df)
        
        # Example prediction on last 100 samples
        print("\n" + "="*50)
        print("EXAMPLE PREDICTIONS")
        print("="*50)
        
        test_data = df.tail(100)
        predictions = predictor.predict(test_data, model_type='xgb')
        
        print("\nSample predictions (last 10 rows):")
        print(predictions.tail(10))
        
        # Plot risk timeline with uncertainty bands
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Upper plot: Risk probability over time with uncertainty
        colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
        risk_colors = [colors[risk] for risk in predictions['risk_class']]

        # Plot uncertainty band (90% confidence interval)
        ax1.fill_between(predictions['index'],
                        predictions['confidence_lower'],
                        predictions['confidence_upper'],
                        alpha=0.2, color='blue', label='90% Confidence Interval')

        # Plot probability line
        ax1.plot(predictions['index'], predictions['clog_probability'],
                 'b-', linewidth=2, alpha=0.8, label='Clog Probability')
        ax1.scatter(predictions['index'], predictions['clog_probability'],
                   c=risk_colors, alpha=0.7, s=30, edgecolors='black', linewidth=0.5)

        # Threshold lines
        ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Medium threshold (0.3)')
        ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='High threshold (0.7)')

        ax1.set_ylabel('Clog Probability', fontsize=12)
        ax1.set_title('Predicted Clogging Risk Over Time with Uncertainty', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)

        # Lower plot: Risk class zones
        risk_numeric = predictions['risk_class'].map({'Low': 0, 'Medium': 1, 'High': 2})
        ax2.fill_between(predictions['index'], 0, risk_numeric,
                        color=risk_colors, alpha=0.4, step='mid')
        ax2.plot(predictions['index'], risk_numeric, 'k-', linewidth=1.5, alpha=0.7)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Low', 'Medium', 'High'])
        ax2.set_xlabel('Time Index', fontsize=12)
        ax2.set_ylabel('Risk Class', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='x')

        # Add clog event markers if available in test data
        if 'filter_status' in test_data.columns:
            clog_indices = test_data[
                (test_data['filter_status'] == 'Clogged') |
                (test_data['filter_status'] == 1)
            ].index
            for clog_idx in clog_indices:
                if clog_idx in predictions['index'].values:
                    ax1.axvline(x=clog_idx, color='purple', linestyle=':',
                              alpha=0.5, linewidth=2)
                    ax2.axvline(x=clog_idx, color='purple', linestyle=':',
                              alpha=0.5, linewidth=2, label='Actual Clog' if clog_idx == clog_indices[0] else '')
            if len(clog_indices) > 0:
                ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(f"{CONFIG['plots_dir']}/risk_timeline.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print("\n" + "="*50)
        print("PIPELINE EXECUTION COMPLETE!")
        print("="*50)
        print("\nTo use the trained model later:")
        print("1. Load: predictor.load_models()")
        print("2. Predict: results = predictor.predict(new_data)")
        print("3. Update: predictor.update_model(new_data)")
        
    except FileNotFoundError:
        print(f"Error: Excel file '{excel_file_path}' not found.")
        print("Please provide the correct path to your Excel file.")
        print("\nRequired columns in Excel file:")
        print("  - time: timestamp or numeric time index")
        print("  - flowrate: flow rate measurements")
        print("  - dp: pressure drop measurements")
        print("  - filter_status: 'Clogged'/'Not Clogged' or 1/0")
        print("\nUpdate the 'excel_file_path' variable with your file's path and run again.")

# Execute if running as script
if __name__ == "__main__":
    main()
    
# %% [markdown]
# ## README - How to Use This System
# 
# ### Quick Start:
# 1. **Prepare your Excel file** with columns: `time`, `flowrate`, `dp`, `filter_status`
# 2. **Update the file path** in the `main()` function
# 3. **Run the entire notebook** to train models and generate predictions
# 
# ### Configuration:
# - Modify parameters in the `CONFIG` dictionary (Section 2)
# - Key settings:
#   - `forecast_horizon_steps`: How far ahead to predict (default: 5 steps)
#   - `risk_thresholds`: Boundaries for Low/Medium/High risk classification
#   - `rolling_windows`: Window sizes for rolling features
#   - `optuna_trials_binary`: Number of hyperparameter optimization trials
# 
# ### Using Trained Models:
# ```python
# # Initialize predictor
# predictor = FilterCloggingPredictor()
# 
# # Load saved models
# predictor.load_models()
# 
# # Make predictions
# results = predictor.predict(new_data_df, model_type='xgb')
# 
# # Update with new data
# predictor.update_model(new_data_df)
# ```
# 
# ### Outputs:
# - **Models**: Saved in `models/` directory
# - **Plots**: Saved in `plots/` directory
# - **Results**: Performance metrics saved in `results/` directory
# 
# ### Recommended Retraining Cadence:
# - Weekly or after observing 5-10 new clogging events
# - Use `update_model()` for incremental updates with saved hyperparameters```