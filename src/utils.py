"""
Utility functions for filter clogging prediction system.
"""

import numpy as np
import pandas as pd
from config import CONFIG, RANDOM_SEED


def time_series_split_imbalanced(df, clog_index=None, config=CONFIG):
    """
    Smart train/val/test split for severely imbalanced time-series data.

    When clogging occurs late (e.g., at point 8940/9000), standard splitting
    would leave almost no positive samples in train/val sets. This function
    creates timeline-aware splits that ensure test set captures degradation phase.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with 'is_clogged' column
    clog_index : int, optional
        Index where clogging starts. If None, auto-detected.
    config : dict
        Configuration dictionary

    Returns:
    --------
    train_idx, val_idx, test_idx, healthy_idx : np.ndarray
        Index arrays for each split
    """
    n = len(df)

    # Auto-detect clogging start point
    if clog_index is None:
        clog_indices = df.index[df['is_clogged'] == 1]
        if len(clog_indices) > 0:
            clog_index = int(clog_indices[0])
        else:
            clog_index = n

    if config.get('verbose', True):
        print(f"\n{'='*60}")
        print(f"TIMELINE-AWARE DATA SPLITTING")
        print(f"{'='*60}")
        print(f"Total samples: {n}")
        print(f"Clogging starts at index: {clog_index}")
        print(f"Clogged samples: {n - clog_index} ({100*(n-clog_index)/n:.2f}%)")

    # Define splits based on degradation timeline
    # Strategy: Ensure all splits have BOTH classes by splitting relative to clog_index

    # For anomaly detection: Use early healthy data (first 60% before clogging)
    healthy_fraction = config.get('anomaly_detection', {}).get('healthy_data_fraction', 0.60)
    healthy_end = int(clog_index * healthy_fraction)

    # For train/val/test: Split to ensure mixed classes
    # Train: 0 to 70% of pre-clog + 0-30% of post-clog
    # Val: 70-85% of pre-clog + 30-50% of post-clog
    # Test: 85-100% of pre-clog + 50-100% of post-clog

    pre_clog_samples = clog_index
    post_clog_samples = n - clog_index

    # Training gets most of pre-clog + some early post-clog
    train_preclog_end = int(pre_clog_samples * 0.70)
    train_postclog_count = int(post_clog_samples * 0.30)
    train_end = train_preclog_end + train_postclog_count

    # Validation gets middle portion
    val_preclog_count = int(pre_clog_samples * 0.15)  # 70-85% = 15%
    val_postclog_count = int(post_clog_samples * 0.20)  # 30-50% = 20%
    val_end = train_end + val_preclog_count + val_postclog_count

    # Create index arrays combining pre and post clog samples
    train_idx = np.concatenate([
        np.arange(0, train_preclog_end),  # Pre-clog training
        np.arange(clog_index, clog_index + train_postclog_count)  # Post-clog training
    ])

    val_idx = np.concatenate([
        np.arange(train_preclog_end, train_preclog_end + val_preclog_count),  # Pre-clog val
        np.arange(clog_index + train_postclog_count,
                 clog_index + train_postclog_count + val_postclog_count)  # Post-clog val
    ])

    # Test gets remainder
    test_idx = np.concatenate([
        np.arange(train_preclog_end + val_preclog_count, clog_index),  # Remaining pre-clog
        np.arange(clog_index + train_postclog_count + val_postclog_count, n)  # Remaining post-clog
    ])

    healthy_idx = np.arange(0, healthy_end)  # Pure healthy for anomaly detection

    if config.get('verbose', True):
        print(f"\nSplit Strategy:")
        print(f"  Healthy set (for anomaly detection): 0 to {healthy_end} ({len(healthy_idx)} samples)")
        print(f"  Training set: 0 to {train_end} ({len(train_idx)} samples)")
        print(f"  Validation set: {train_end} to {val_end} ({len(val_idx)} samples)")
        print(f"  Test set: {val_end} to {n} ({len(test_idx)} samples)")

        # Show class distribution in each split
        for name, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
            n_clogged = df.iloc[idx]['is_clogged'].sum()
            pct = 100 * n_clogged / len(idx) if len(idx) > 0 else 0
            print(f"  {name} clogged: {n_clogged}/{len(idx)} ({pct:.2f}%)")

        print(f"{'='*60}\n")

    return train_idx, val_idx, test_idx, healthy_idx


def compute_temporal_weights(time_to_clog, config=CONFIG):
    """
    Compute temporal weights using exponential decay.

    Samples closer to clogging event get higher weights.
    This helps models focus on critical degradation phase.

    Parameters:
    -----------
    time_to_clog : np.ndarray
        Time remaining until clogging (in steps)
    config : dict
        Configuration dictionary

    Returns:
    --------
    weights : np.ndarray
        Normalized temporal weights
    """
    decay_factor = config.get('weight_decay_factor', 0.2)

    # Compute maximum observed time (excluding infinity)
    finite_times = time_to_clog[time_to_clog < float('inf')]
    if len(finite_times) == 0:
        return np.ones_like(time_to_clog)

    max_time = np.max(finite_times)

    # Exponential decay: w(t) = exp(-t / (max_time * decay_factor))
    weights = np.exp(-time_to_clog / (max_time * decay_factor + 1e-8))

    # Handle censored samples (time_to_clog = inf)
    min_weight = np.min(weights[weights > 0])
    weights[time_to_clog == float('inf')] = min_weight

    # Normalize to mean = 1.0
    weights = weights / (np.mean(weights) + 1e-8)

    return weights


def combine_sample_weights(class_weights, temporal_weights):
    """
    Combine class balancing weights with temporal weights.

    Parameters:
    -----------
    class_weights : np.ndarray
        Weights from class balancing (e.g., inverse frequency)
    temporal_weights : np.ndarray
        Weights from temporal decay

    Returns:
    --------
    combined_weights : np.ndarray
        Normalized combined weights
    """
    combined = class_weights * temporal_weights
    # Normalize to mean = 1.0
    combined = combined / (np.mean(combined) + 1e-8)
    return combined


def optimize_threshold_by_cost(y_true, y_proba, cost_fn=100, cost_fp=1):
    """
    Find optimal classification threshold by minimizing operational cost.

    In filter clogging prediction:
    - False Negative (missed clog) = DISASTER = high cost (e.g., 100)
    - False Positive (unnecessary maintenance) = minor inconvenience = low cost (e.g., 1)

    Parameters:
    -----------
    y_true : np.ndarray
        True labels (0 = healthy, 1 = clogged)
    y_proba : np.ndarray
        Predicted probabilities for positive class
    cost_fn : float
        Cost of false negative (default: 100)
    cost_fp : float
        Cost of false positive (default: 1)

    Returns:
    --------
    optimal_threshold : float
        Threshold that minimizes total cost
    optimal_cost : float
        Minimum total cost achieved
    results : dict
        Full results including costs at all thresholds
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    metrics = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        # Confusion matrix components
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        # Total operational cost
        total_cost = cost_fn * fn + cost_fp * fp

        costs.append(total_cost)
        metrics.append({
            'threshold': threshold,
            'cost': total_cost,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        })

    # Find optimal threshold
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = costs[optimal_idx]

    results = {
        'optimal_threshold': optimal_threshold,
        'optimal_cost': optimal_cost,
        'all_metrics': metrics,
        'thresholds': thresholds,
        'costs': np.array(costs),
    }

    return optimal_threshold, optimal_cost, results


def safe_divide(numerator, denominator, fill_value=0.0):
    """
    Safely divide arrays, replacing inf/nan with fill_value.

    Parameters:
    -----------
    numerator : np.ndarray or pd.Series
        Numerator values
    denominator : np.ndarray or pd.Series
        Denominator values
    fill_value : float
        Value to use for division by zero (default: 0.0)

    Returns:
    --------
    result : np.ndarray or pd.Series
        Safe division result
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        result = np.where(np.isfinite(result), result, fill_value)
    return result


# ============================================================
# Risk Level Utilities
# ============================================================

def compute_risk_level_confusion_matrix(y_true_levels, y_pred_levels, num_levels=4):
    """
    Compute confusion matrix for multi-level risk predictions.

    Parameters:
    -----------
    y_true_levels : np.ndarray
        True risk levels [0, 1, 2, 3]
    y_pred_levels : np.ndarray
        Predicted risk levels [0, 1, 2, 3]
    num_levels : int
        Number of risk levels (default: 4)

    Returns:
    --------
    confusion_matrix : np.ndarray, shape (num_levels, num_levels)
        Confusion matrix where element [i, j] is count of samples
        with true level i predicted as level j
    """
    cm = np.zeros((num_levels, num_levels), dtype=int)

    for i in range(num_levels):
        for j in range(num_levels):
            cm[i, j] = np.sum((y_true_levels == i) & (y_pred_levels == j))

    return cm


def compute_risk_level_metrics(y_true_levels, y_pred_levels, config=CONFIG):
    """
    Compute comprehensive metrics for risk level predictions.

    Parameters:
    -----------
    y_true_levels : np.ndarray
        True risk levels [0, 1, 2, 3]
    y_pred_levels : np.ndarray
        Predicted risk levels [0, 1, 2, 3]
    config : dict
        Configuration dictionary with risk level costs

    Returns:
    --------
    metrics : dict
        Dictionary containing:
        - 'accuracy': Overall accuracy
        - 'per_level_accuracy': Accuracy for each level
        - 'confusion_matrix': Confusion matrix
        - 'mean_absolute_error': MAE in risk levels (0-3)
        - 'operational_cost': Total operational cost based on misclassifications
        - 'critical_recall': Recall for CRITICAL level (most important)
        - 'critical_precision': Precision for CRITICAL level
    """
    num_levels = config.get('risk_levels', {}).get('num_levels', 4)
    level_costs = config.get('risk_levels', {}).get('level_costs', {0: 0, 1: 5, 2: 25, 3: 100})

    # Confusion matrix
    cm = compute_risk_level_confusion_matrix(y_true_levels, y_pred_levels, num_levels)

    # Overall accuracy
    accuracy = np.sum(y_true_levels == y_pred_levels) / len(y_true_levels)

    # Per-level accuracy
    per_level_accuracy = {}
    for level in range(num_levels):
        mask = y_true_levels == level
        if np.sum(mask) > 0:
            per_level_accuracy[level] = np.sum(y_pred_levels[mask] == level) / np.sum(mask)
        else:
            per_level_accuracy[level] = 0.0

    # Mean absolute error in levels
    mae = np.mean(np.abs(y_true_levels - y_pred_levels))

    # Operational cost
    # Cost depends on severity of misclassification
    operational_cost = 0
    for true_level in range(num_levels):
        for pred_level in range(num_levels):
            count = cm[true_level, pred_level]
            if true_level != pred_level:
                # Cost = difference in levels × base cost
                # Under-prediction is more costly than over-prediction
                if pred_level < true_level:
                    # Underestimating risk (dangerous)
                    cost_multiplier = level_costs[true_level] * 2
                else:
                    # Overestimating risk (wasteful but safe)
                    cost_multiplier = level_costs[pred_level] * 0.5

                operational_cost += count * cost_multiplier

    # Critical level metrics (most important)
    critical_level = num_levels - 1  # Level 3
    critical_mask_true = y_true_levels == critical_level
    critical_mask_pred = y_pred_levels == critical_level

    if np.sum(critical_mask_true) > 0:
        critical_recall = np.sum(critical_mask_true & critical_mask_pred) / np.sum(critical_mask_true)
    else:
        critical_recall = 0.0

    if np.sum(critical_mask_pred) > 0:
        critical_precision = np.sum(critical_mask_true & critical_mask_pred) / np.sum(critical_mask_pred)
    else:
        critical_precision = 0.0

    metrics = {
        'accuracy': accuracy,
        'per_level_accuracy': per_level_accuracy,
        'confusion_matrix': cm,
        'mean_absolute_error': mae,
        'operational_cost': operational_cost,
        'critical_recall': critical_recall,
        'critical_precision': critical_precision,
    }

    return metrics


def map_binary_to_risk_level(y_binary, y_proba=None, config=CONFIG):
    """
    Map binary classification labels to 4-level risk scores.

    Useful for converting legacy binary predictions to risk levels.

    Parameters:
    -----------
    y_binary : np.ndarray
        Binary labels (0 = healthy, 1 = clogging)
    y_proba : np.ndarray, optional
        Predicted probabilities for positive class
        If None, uses simple mapping (0 → LOW, 1 → CRITICAL)
    config : dict
        Configuration dictionary

    Returns:
    --------
    risk_levels : np.ndarray
        Risk levels [0, 1, 2, 3]
    """
    if y_proba is None:
        # Simple mapping without probabilities
        risk_levels = y_binary * 3  # 0 → 0 (LOW), 1 → 3 (CRITICAL)
    else:
        # Map using probability thresholds
        risk_config = config.get('risk_levels', {})
        thresholds = risk_config.get('thresholds', {
            'low': 0.25, 'moderate': 0.50, 'high': 0.75, 'critical': 1.00
        })

        risk_levels = np.zeros(len(y_proba), dtype=int)
        risk_levels[y_proba >= thresholds['low']] = 1
        risk_levels[y_proba >= thresholds['moderate']] = 2
        risk_levels[y_proba >= thresholds['high']] = 3

    return risk_levels


def get_risk_level_color(risk_level, config=CONFIG):
    """
    Get color code for a risk level.

    Parameters:
    -----------
    risk_level : int
        Risk level [0, 1, 2, 3]
    config : dict
        Configuration dictionary

    Returns:
    --------
    color : str
        Hex color code
    """
    colors = config.get('risk_levels', {}).get('colors', {
        0: '#28a745', 1: '#ffc107', 2: '#fd7e14', 3: '#dc3545'
    })
    return colors.get(risk_level, '#6c757d')


def get_risk_level_label(risk_level, config=CONFIG):
    """
    Get human-readable label for a risk level.

    Parameters:
    -----------
    risk_level : int
        Risk level [0, 1, 2, 3]
    config : dict
        Configuration dictionary

    Returns:
    --------
    label : str
        Human-readable label
    """
    labels = config.get('risk_levels', {}).get('labels', {
        0: 'LOW', 1: 'MODERATE', 2: 'HIGH', 3: 'CRITICAL'
    })
    return labels.get(risk_level, 'UNKNOWN')
