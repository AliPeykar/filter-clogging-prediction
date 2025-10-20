"""
Feature engineering functions for filter clogging prediction.
"""

import numpy as np
import pandas as pd
from config import CONFIG


def create_rolling_features(df, columns, window_sizes=None, config=CONFIG):
    """
    Create rolling window statistics features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to create rolling features for
    window_sizes : list, optional
        Window sizes for rolling statistics
    config : dict
        Configuration dictionary

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added rolling features
    """
    if window_sizes is None:
        window_sizes = config.get('rolling_window_sizes', [5, 10, 20, 50])

    for col in columns:
        for window in window_sizes:
            # Rolling mean
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(
                window=window, min_periods=1
            ).mean()

            # Rolling standard deviation
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(
                window=window, min_periods=1
            ).std().fillna(0)

            # Rolling min/max
            df[f'{col}_rolling_min_{window}'] = df[col].rolling(
                window=window, min_periods=1
            ).min()

            df[f'{col}_rolling_max_{window}'] = df[col].rolling(
                window=window, min_periods=1
            ).max()

    return df


def create_lag_features(df, columns, lag_steps=None, config=CONFIG):
    """
    Create lagged features for time-series modeling.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to create lag features for
    lag_steps : list, optional
        Lag steps to create
    config : dict
        Configuration dictionary

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added lag features
    """
    if lag_steps is None:
        lag_steps = config.get('lag_features', [1, 5, 10, 20, 50])

    for col in columns:
        for lag in lag_steps:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag).bfill()

    return df


def create_ewm_features(df, columns, spans=None, config=CONFIG):
    """
    Create exponentially weighted moving average features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to create EWM features for
    spans : list, optional
        Span values for EWM
    config : dict
        Configuration dictionary

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added EWM features
    """
    if spans is None:
        spans = config.get('ewm_spans', [10, 30, 50])

    for col in columns:
        for span in spans:
            df[f'{col}_ewm_{span}'] = df[col].ewm(span=span, adjust=False).mean()

    return df


def create_rate_of_change_features(df, columns):
    """
    Create rate of change (derivative) features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to create rate of change features for

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added rate of change features
    """
    for col in columns:
        # First derivative (rate of change)
        df[f'{col}_rate_of_change'] = df[col].diff().fillna(0)

        # Second derivative (acceleration)
        df[f'{col}_acceleration'] = df[f'{col}_rate_of_change'].diff().fillna(0)

    return df


def create_interaction_features(df, column_pairs):
    """
    Create interaction features between column pairs.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column_pairs : list of tuples
        Pairs of columns to create interactions for
        Example: [('col1', 'col2'), ('col3', 'col4')]

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added interaction features
    """
    for col1, col2 in column_pairs:
        if col1 in df.columns and col2 in df.columns:
            # Multiplicative interaction
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]

            # Additive interaction
            df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]

            # Difference
            df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]

    return df


def create_degradation_index(df):
    """
    Create filter degradation index based on differential pressure and flow rate.

    Degradation Index = (Current DP / Initial DP) * (Initial Flow / Current Flow)
    Higher values indicate more degradation.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'differential_pressure' and 'flow_rate'

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added degradation index
    """
    # Get initial (baseline) values from first few samples
    initial_dp = df['differential_pressure'].iloc[:10].mean()
    initial_flow = df['flow_rate'].iloc[:10].mean()

    # Avoid division by zero
    initial_dp = initial_dp if initial_dp > 0 else 1.0
    initial_flow = initial_flow if initial_flow > 0 else 1.0

    # Compute degradation index
    df['degradation_index'] = (
        (df['differential_pressure'] / initial_dp) *
        (initial_flow / df['flow_rate'].replace(0, 1.0))
    )

    return df


def create_cumulative_features(df, columns):
    """
    Create cumulative sum features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to create cumulative features for

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added cumulative features
    """
    for col in columns:
        df[f'{col}_cumsum'] = df[col].cumsum()

    return df


def engineer_all_features(df, config=CONFIG):
    """
    Apply all feature engineering transformations.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    config : dict
        Configuration dictionary

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with all engineered features
    """
    # Core columns for feature engineering
    core_columns = ['differential_pressure', 'flow_rate']

    if config.get('verbose', True):
        print(f"\nFeature Engineering:")
        print(f"  Starting features: {len(df.columns)}")

    # 1. Ratio features
    from data_processing import create_ratio_features
    df = create_ratio_features(df)

    # 2. Rolling statistics
    df = create_rolling_features(df, core_columns, config=config)

    # 3. Lag features
    df = create_lag_features(df, core_columns, config=config)

    # 4. Exponentially weighted features
    df = create_ewm_features(df, core_columns, config=config)

    # 5. Rate of change
    df = create_rate_of_change_features(df, core_columns)

    # 6. Degradation index
    df = create_degradation_index(df)

    # 7. Interaction features
    df = create_interaction_features(df, [
        ('differential_pressure', 'flow_rate'),
        ('differential_pressure', 'degradation_index'),
    ])

    # 8. Cumulative features
    df = create_cumulative_features(df, ['dp_flowrate_ratio'])

    if config.get('verbose', True):
        print(f"  Final features: {len(df.columns)}")
        print(f"  New features created: {len(df.columns) - 2}")  # Subtract original columns

    return df
