"""
Data processing and preparation functions for filter clogging prediction.
"""

import numpy as np
import pandas as pd
from config import CONFIG, RANDOM_SEED
from utils import safe_divide


def load_and_prepare_data(filepath, config=CONFIG):
    """
    Load and prepare filter data from CSV or Excel file.

    Parameters:
    -----------
    filepath : str
        Path to CSV or Excel file (.csv, .xlsx, .xls)
    config : dict
        Configuration dictionary

    Returns:
    --------
    df : pd.DataFrame
        Prepared dataframe with basic features
    """
    # Detect file type and load accordingly
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format. Use .csv, .xlsx, or .xls")

    # Standardize column names (handle different naming conventions)
    column_mapping = {}

    # Map differential pressure column
    if 'dp' in df.columns:
        column_mapping['dp'] = 'differential_pressure'
    elif 'Differential Pressure' in df.columns:
        column_mapping['Differential Pressure'] = 'differential_pressure'
    elif 'pressure' in df.columns:
        column_mapping['pressure'] = 'differential_pressure'
    elif 'differential_pressure' not in df.columns:
        raise ValueError(f"Missing pressure column. Expected: 'dp', 'differential_pressure', or 'Differential Pressure'. Found: {df.columns.tolist()}")

    # Map flow rate column
    if 'flowrate' in df.columns:
        column_mapping['flowrate'] = 'flow_rate'
    elif 'Flow Rate' in df.columns:
        column_mapping['Flow Rate'] = 'flow_rate'
    elif 'flow' in df.columns:
        column_mapping['flow'] = 'flow_rate'
    elif 'flow_rate' not in df.columns:
        raise ValueError(f"Missing flow rate column. Expected: 'flowrate', 'flow_rate', or 'Flow Rate'. Found: {df.columns.tolist()}")

    # Map filter status/clogged column if exists
    if 'filter_status' in df.columns:
        # Convert 'Not Clogged'/'Clogged' to 0/1
        df['is_clogged'] = (df['filter_status'] == 'Clogged').astype(int)
    elif 'Filter Status' in df.columns:
        df['is_clogged'] = (df['Filter Status'] == 'Clogged').astype(int)
    elif 'status' in df.columns:
        df['is_clogged'] = (df['status'] == 'Clogged').astype(int)

    # Apply column renaming
    if column_mapping:
        df = df.rename(columns=column_mapping)

    if config.get('verbose', True):
        print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {df.columns.tolist()}")

    return df


def compute_target_labels(df, config=CONFIG):
    """
    Compute target labels for classification and survival analysis.

    Creates:
    - is_clogged: binary label (0=healthy, 1=clogged)
    - time_to_clog: continuous time until clogging event
    - event_occurred: binary indicator for survival analysis

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    config : dict
        Configuration dictionary

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added target columns
    """
    forecast_horizon = config.get('forecast_horizon_steps', 120)

    # Find clogging point if 'is_clogged' exists
    if 'is_clogged' not in df.columns:
        df['is_clogged'] = 0  # All healthy by default

    # Find first clogging index
    clog_indices = df.index[df['is_clogged'] == 1].tolist()
    clog_start_idx = clog_indices[0] if len(clog_indices) > 0 else len(df)

    # Compute time_to_clog for each sample
    time_to_clog = []
    for i in range(len(df)):
        if i >= clog_start_idx:
            time_to_clog.append(0)  # Already clogged
        else:
            time_to_clog.append(clog_start_idx - i)

    df['time_to_clog'] = time_to_clog

    # Binary classification target: clogging within forecast_horizon
    df['will_clog_soon'] = (df['time_to_clog'] <= forecast_horizon).astype(int)

    # Survival analysis targets
    # Duration: use actual time_to_clog, but clip at horizon for censored samples
    # Event: 1 if clog within horizon, 0 if censored (beyond horizon)
    df['duration'] = df['time_to_clog'].copy()
    df['event_occurred'] = (df['time_to_clog'] <= forecast_horizon).astype(int)

    # For censored samples (time_to_clog > horizon), cap duration at horizon
    # This properly indicates "at least this long, but we don't know exact time"
    censored_mask = df['time_to_clog'] > forecast_horizon
    df.loc[censored_mask, 'duration'] = forecast_horizon

    if config.get('verbose', True):
        print(f"\nTarget Distribution:")
        print(f"  Clogging starts at index: {clog_start_idx}")
        print(f"  Samples with clog within {forecast_horizon} steps: {df['will_clog_soon'].sum()}")
        print(f"  Positive class ratio: {100*df['will_clog_soon'].mean():.2f}%")

    return df


def create_ratio_features(df):
    """
    Create ratio-based features with safe division.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added ratio features
    """
    # Differential pressure / flow rate ratio (resistance indicator)
    df['dp_flowrate_ratio'] = safe_divide(
        df['differential_pressure'],
        df['flow_rate'],
        fill_value=0.0
    )

    # Flow rate / differential pressure ratio (efficiency indicator)
    df['flowrate_dp_ratio'] = safe_divide(
        df['flow_rate'],
        df['differential_pressure'],
        fill_value=0.0
    )

    return df


def handle_missing_values(df, strategy='forward_fill'):
    """
    Handle missing values in dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy for handling missing values:
        - 'forward_fill': Forward fill then backward fill
        - 'interpolate': Linear interpolation
        - 'drop': Drop rows with missing values

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with handled missing values
    """
    if strategy == 'forward_fill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif strategy == 'interpolate':
        df = df.interpolate(method='linear').fillna(method='bfill')
    elif strategy == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df


def remove_outliers(df, columns, method='iqr', threshold=3.0):
    """
    Remove outliers from specified columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to check for outliers
    method : str
        Method for outlier detection:
        - 'iqr': Interquartile range method
        - 'zscore': Z-score method
    threshold : float
        Threshold for outlier detection

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with outliers removed
    n_removed : int
        Number of rows removed
    """
    n_before = len(df)

    if method == 'iqr':
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    elif method == 'zscore':
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < threshold]

    n_removed = n_before - len(df)

    return df, n_removed


def prepare_features_and_targets(df, config=CONFIG):
    """
    Prepare feature matrix and target arrays for modeling.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with all features and targets
    config : dict
        Configuration dictionary

    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y_class : np.ndarray
        Binary classification target
    y_time : np.ndarray
        Time-to-clog target for regression
    y_duration : np.ndarray
        Duration for survival analysis
    y_event : np.ndarray
        Event indicator for survival analysis
    feature_names : list
        List of feature column names
    """
    # Define columns to exclude from features
    exclude_cols = [
        'is_clogged', 'will_clog_soon', 'time_to_clog',
        'duration', 'event_occurred', 'timestamp',
        'date', 'time', 'datetime', 'filter_status'
    ]

    # Select feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y_class = df['will_clog_soon'].values
    y_time = df['time_to_clog'].values
    y_duration = df['duration'].values
    y_event = df['event_occurred'].values

    if config.get('verbose', True):
        print(f"\nFeature Matrix:")
        print(f"  Shape: {X.shape}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Target distribution: {np.bincount(y_class)}")

    return X, y_class, y_time, y_duration, y_event, feature_cols
