"""
Model evaluation metrics and visualization functions.
Enhanced with improved styling, additional plots, and interactive capabilities.
"""

import numpy as np
import pandas as pd

# Set matplotlib backend to Agg (non-interactive) to avoid tkinter cleanup warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve
from config import CONFIG

# Try importing plotly for interactive plots (optional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Try importing SHAP for model interpretability (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# Try importing LIME for local interpretability (optional)
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

# Set global plot styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


def evaluate_classification_model(y_true, y_pred, y_proba=None, model_name='Model', verbose=True):
    """
    Comprehensive evaluation for binary classification.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities for positive class
    model_name : str
        Model name for reporting
    verbose : bool
        Print detailed report

    Returns:
    --------
    metrics : dict
        Dictionary with all evaluation metrics
    """
    metrics = {}

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['classification_report'] = report

    # Confusion matrix - ensure it's always 2x2 by specifying labels
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Handle edge case where cm might not be 2x2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        # Only one class present in both y_true and y_pred
        if y_true[0] == 0:  # All negative
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:  # All positive
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        # Fallback for unexpected shapes
        tn, fp, fn, tp = 0, 0, 0, 0

    metrics['confusion_matrix'] = cm
    metrics['TP'] = tp
    metrics['FP'] = fp
    metrics['TN'] = tn
    metrics['FN'] = fn

    # Core metrics
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC-AUC and PR-AUC (if probabilities provided)
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        metrics['precision_curve'] = precision
        metrics['recall_curve'] = recall

    # Operational cost (assuming cost_fn=100, cost_fp=1)
    metrics['operational_cost'] = 100 * fn + 1 * fp

    if verbose:
        print(f"\n{'='*60}")
        print(f"{model_name.upper()} - CLASSIFICATION METRICS")
        print(f"{'='*60}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp:4d}  |  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  |  TN: {tn:4d}")
        print(f"\nCore Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")

        if y_proba is not None:
            print(f"\nProbabilistic Metrics:")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")

        print(f"\nOperational Cost:")
        print(f"  Total Cost: {metrics['operational_cost']:.0f}")
        print(f"  (FN×100 + FP×1)")
        print(f"{'='*60}\n")

    return metrics


def evaluate_regression_model(y_true, y_pred, model_name='Model', verbose=True):
    """
    Evaluate regression model for time-to-clog prediction.

    Parameters:
    -----------
    y_true : array-like
        True time-to-clog values
    y_pred : array-like
        Predicted time-to-clog values
    model_name : str
        Model name for reporting
    verbose : bool
        Print detailed report

    Returns:
    --------
    metrics : dict
        Dictionary with regression metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    metrics = {}

    # Core regression metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    # Additional metrics
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    metrics['median_ae'] = np.median(np.abs(y_true - y_pred))

    # Residuals
    residuals = y_true - y_pred
    metrics['residuals'] = residuals
    metrics['residuals_mean'] = np.mean(residuals)
    metrics['residuals_std'] = np.std(residuals)

    if verbose:
        print(f"\n{'='*60}")
        print(f"{model_name.upper()} - REGRESSION METRICS")
        print(f"{'='*60}")
        print(f"  MAE:        {metrics['mae']:.2f} steps")
        print(f"  RMSE:       {metrics['rmse']:.2f} steps")
        print(f"  MAPE:       {metrics['mape']:.2f}%")
        print(f"  Median AE:  {metrics['median_ae']:.2f} steps")
        print(f"\nResidual Statistics:")
        print(f"  Mean:       {metrics['residuals_mean']:.2f}")
        print(f"  Std:        {metrics['residuals_std']:.2f}")
        print(f"\nNote: R² is not shown as it's not appropriate for censored")
        print(f"      time-to-event data. Focus on F1-Score for model quality.")
        print(f"{'='*60}\n")

    return metrics


def plot_confusion_matrix(cm, model_name='Model', save_path='plots'):
    """
    Plot confusion matrix heatmap with enhanced styling.

    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    model_name : str
        Model name for title
    save_path : str
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate percentages (handle division by zero)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Replace 0 with 1 to avoid division by zero
    cm_percent = cm.astype('float') / row_sums * 100

    # Create annotations with both counts and percentages
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(
        cm, annot=annot, fmt='', cmap='RdYlGn_r',
        xticklabels=['Healthy', 'Clogging'],
        yticklabels=['Healthy', 'Clogging'],
        cbar_kws={'label': 'Count'},
        linewidths=2, linecolor='white',
        vmin=0, square=True, ax=ax
    )

    plt.title(f'{model_name}\nConfusion Matrix', fontweight='bold', pad=20)
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()

    import os
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{model_name.lower()}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, model_name='Model', save_path='plots'):
    """
    Plot ROC curve with enhanced styling and annotations.

    Parameters:
    -----------
    fpr : array-like
        False positive rates
    tpr : array-like
        True positive rates
    roc_auc : float
        Area under ROC curve
    model_name : str
        Model name for title
    save_path : str
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ROC curve with gradient effect
    # Ensure markevery is at least 1 to avoid zero step error
    markevery = max(1, len(fpr)//10) if len(fpr) > 0 else None
    ax.plot(fpr, tpr, color='#2E86AB', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})',
            marker='o', markevery=markevery, markersize=6)
    ax.plot([0, 1], [0, 1], color='#E63946', lw=2, linestyle='--', label='Random Classifier', alpha=0.7)

    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB', label=f'AUC Area')

    # Add optimal threshold point (closest to top-left)
    optimal_idx = np.argmax(tpr - fpr)
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'r*', markersize=20,
            label=f'Optimal Threshold (FPR={fpr[optimal_idx]:.3f}, TPR={tpr[optimal_idx]:.3f})')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=13)
    ax.set_title(f'{model_name}\nReceiver Operating Characteristic (ROC) Curve',
                 fontweight='bold', pad=20, fontsize=15)
    ax.legend(loc='lower right', framealpha=0.95, shadow=True)
    ax.grid(alpha=0.4, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal')
    plt.tight_layout()

    import os
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{model_name.lower()}_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(precision, recall, pr_auc, model_name='Model', save_path='plots'):
    """
    Plot Precision-Recall curve with enhanced styling.

    Parameters:
    -----------
    precision : array-like
        Precision values
    recall : array-like
        Recall values
    pr_auc : float
        Area under PR curve
    model_name : str
        Model name for title
    save_path : str
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Ensure markevery is at least 1 to avoid zero step error
    markevery = max(1, len(recall)//10) if len(recall) > 0 else None
    ax.plot(recall, precision, color='#06A77D', lw=3,
            label=f'PR Curve (AUC = {pr_auc:.4f})',
            marker='s', markevery=markevery, markersize=6)

    # Fill area under curve
    ax.fill_between(recall, precision, alpha=0.2, color='#06A77D')

    # Find and mark F1-optimal point
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    f1_optimal_idx = np.argmax(f1_scores)
    ax.plot(recall[f1_optimal_idx], precision[f1_optimal_idx], 'r*', markersize=20,
            label=f'Optimal F1 (P={precision[f1_optimal_idx]:.3f}, R={recall[f1_optimal_idx]:.3f})')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Precision (PPV)', fontweight='bold', fontsize=13)
    ax.set_title(f'{model_name}\nPrecision-Recall Curve',
                 fontweight='bold', pad=20, fontsize=15)
    ax.legend(loc='lower left', framealpha=0.95, shadow=True)
    ax.grid(alpha=0.4, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    import os
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{model_name.lower()}_pr_curve.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_timeline(y_true, y_pred, title='Predictions Over Time', save_path='plots'):
    """
    Plot predictions vs actual values over time with enhanced visualization.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str
        Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    time_steps = np.arange(len(y_true))

    # Plot actual values
    ax.plot(time_steps, y_true, label='Actual', alpha=0.8, linewidth=2.5,
            color='#264653', marker='o', markevery=max(1, len(y_true)//50), markersize=4)

    # Plot predicted values
    ax.plot(time_steps, y_pred, label='Predicted', alpha=0.8, linewidth=2.5,
            color='#E76F51', marker='s', markevery=max(1, len(y_pred)//50), markersize=4)

    # Highlight clogging regions (where y_true == 1)
    if np.any(y_true == 1):
        clog_regions = y_true == 1
        ax.fill_between(time_steps, 0, 1, where=clog_regions,
                        alpha=0.2, color='red', label='True Clogging Events')

    # Highlight predicted clogging regions
    if np.any(y_pred > 0.5):
        pred_clog_regions = y_pred > 0.5
        ax.fill_between(time_steps, 0, 1, where=pred_clog_regions,
                        alpha=0.15, color='orange', label='Predicted Clogging Events')

    ax.set_xlabel('Time Step', fontweight='bold', fontsize=13)
    ax.set_ylabel('Risk Score / Class Label', fontweight='bold', fontsize=13)
    ax.set_title(title, fontweight='bold', pad=20, fontsize=15)
    ax.legend(loc='best', framealpha=0.95, shadow=True)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim([-0.05, 1.05])
    plt.tight_layout()

    import os
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/prediction_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals(y_true, y_pred, model_name='Model', save_path='plots'):
    """
    Plot residual analysis for regression.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    model_name : str
        Model name for title
    save_path : str
        Directory to save plot
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{model_name} - Residual Plot')
    axes[0].grid(alpha=0.3)

    # Residual histogram
    axes[1].hist(residuals, bins=50, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{model_name} - Residual Distribution')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    import os
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{model_name.lower()}_residuals.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances, feature_names, model_name='Model', top_n=20, save_path='plots'):
    """
    Plot feature importance.

    Parameters:
    -----------
    importances : array-like
        Feature importance values
    feature_names : list
        Feature names
    model_name : str
        Model name for title
    top_n : int
        Number of top features to display
    save_path : str
        Directory to save plot
    """
    # Sort features by importance
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'{model_name} - Top {top_n} Feature Importances')
    plt.tight_layout()

    import os
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{model_name.lower()}_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()


def compare_models(metrics_dict, metric_name='f1_score', save_path='plots'):
    """
    Compare multiple models on a specific metric with enhanced visuals.

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary mapping model names to their metrics dictionaries
    metric_name : str
        Name of metric to compare
    save_path : str
        Directory to save plot
    """
    model_names = list(metrics_dict.keys())
    scores = [metrics_dict[name].get(metric_name, 0) for name in model_names]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create gradient colors
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
    bars = ax.bar(model_names, scores, color=colors, edgecolor='black', linewidth=1.5)

    # Highlight best model
    best_idx = np.argmax(scores)
    bars[best_idx].set_color('#FFD700')
    bars[best_idx].set_edgecolor('darkred')
    bars[best_idx].set_linewidth(3)

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontweight='bold', fontsize=13)
    ax.set_title(f'Model Comparison - {metric_name.replace("_", " ").title()}',
                 fontweight='bold', pad=20, fontsize=15)
    ax.set_xlabel('Model Name', fontweight='bold', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    import os
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/model_comparison_{metric_name}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration_curve(y_true, y_proba, model_name='Model', n_bins=10, save_path='plots'):
    """
    Plot calibration curve to assess probability calibration.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    model_name : str
        Model name for title
    n_bins : int
        Number of bins for calibration
    save_path : str
        Directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')

    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
    ax1.plot(prob_pred, prob_true, 's-', lw=3, color='#E63946',
             markersize=10, label=f'{model_name}')
    ax1.fill_between(prob_pred, prob_true, alpha=0.2, color='#E63946')

    ax1.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Fraction of Positives', fontweight='bold', fontsize=13)
    ax1.set_title('Calibration Curve', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left', framealpha=0.95, shadow=True)
    ax1.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_aspect('equal')

    # Prediction distribution
    ax2.hist(y_proba[y_true == 0], bins=30, alpha=0.6, label='Negative Class',
             color='#457B9D', edgecolor='black', density=True)
    ax2.hist(y_proba[y_true == 1], bins=30, alpha=0.6, label='Positive Class',
             color='#E63946', edgecolor='black', density=True)
    ax2.set_xlabel('Predicted Probability', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Density', fontweight='bold', fontsize=13)
    ax2.set_title('Prediction Distribution', fontweight='bold', fontsize=14)
    ax2.legend(loc='upper center', framealpha=0.95, shadow=True)
    ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    fig.suptitle(f'{model_name} - Calibration Analysis', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    import os
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{model_name.lower()}_calibration.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curves(estimator, X, y, cv=5, scoring='f1', model_name='Model', save_path='plots'):
    """
    Plot learning curves to diagnose bias/variance.

    Parameters:
    -----------
    estimator : sklearn estimator
        Model to evaluate
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv : int
        Cross-validation folds
    scoring : str
        Scoring metric
    model_name : str
        Model name for title
    save_path : str
        Directory to save plot
    """
    train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=train_sizes, n_jobs=-1, random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot training scores
    ax.plot(train_sizes_abs, train_mean, 'o-', color='#2A9D8F', lw=3,
            markersize=8, label='Training Score')
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='#2A9D8F')

    # Plot validation scores
    ax.plot(train_sizes_abs, val_mean, 's-', color='#E76F51', lw=3,
            markersize=8, label='Validation Score')
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color='#E76F51')

    ax.set_xlabel('Training Set Size', fontweight='bold', fontsize=13)
    ax.set_ylabel(f'{scoring.upper()} Score', fontweight='bold', fontsize=13)
    ax.set_title(f'{model_name} - Learning Curves', fontweight='bold', pad=20, fontsize=15)
    ax.legend(loc='lower right', framealpha=0.95, shadow=True, fontsize=12)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    # Add annotation
    final_gap = train_mean[-1] - val_mean[-1]
    if final_gap > 0.1:
        ax.text(0.5, 0.05, f'High variance detected (gap = {final_gap:.3f})',
                transform=ax.transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()

    import os
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{model_name.lower()}_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_threshold_analysis(y_true, y_proba, model_name='Model', save_path='plots'):
    """
    Analyze model performance across different classification thresholds.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    model_name : str
        Model name for title
    save_path : str
        Directory to save plot
    """
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(thresholds, precisions, lw=3, label='Precision', color='#2A9D8F', marker='o', markevery=10)
    ax.plot(thresholds, recalls, lw=3, label='Recall', color='#E76F51', marker='s', markevery=10)
    ax.plot(thresholds, f1_scores, lw=3, label='F1-Score', color='#264653', marker='^', markevery=10)
    ax.plot(thresholds, accuracies, lw=3, label='Accuracy', color='#F4A261', marker='d', markevery=10)

    # Mark optimal F1 threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    ax.axvline(optimal_threshold, color='red', linestyle='--', lw=2,
               label=f'Optimal Threshold = {optimal_threshold:.3f}')
    ax.plot(optimal_threshold, f1_scores[optimal_idx], 'r*', markersize=20)

    ax.set_xlabel('Classification Threshold', fontweight='bold', fontsize=13)
    ax.set_ylabel('Score', fontweight='bold', fontsize=13)
    ax.set_title(f'{model_name} - Threshold Analysis', fontweight='bold', pad=20, fontsize=15)
    ax.legend(loc='best', framealpha=0.95, shadow=True, fontsize=11)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    import os
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{model_name.lower()}_threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    return optimal_threshold


def create_evaluation_dashboard(metrics, y_true, y_pred, y_proba, model_name='Model', save_path='plots'):
    """
    Create comprehensive evaluation dashboard with multiple subplots.

    Parameters:
    -----------
    metrics : dict
        Metrics dictionary from evaluate_classification_model
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like
        Predicted probabilities
    model_name : str
        Model name for title
    save_path : str
        Directory to save plot
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = metrics['confusion_matrix']
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax1, cbar=False,
                xticklabels=['Healthy', 'Clog'], yticklabels=['Healthy', 'Clog'])
    ax1.set_title('Confusion Matrix', fontweight='bold')

    # 2. ROC Curve
    ax2 = fig.add_subplot(gs[0, 1])
    if 'fpr' in metrics and 'tpr' in metrics:
        ax2.plot(metrics['fpr'], metrics['tpr'], 'b-', lw=2,
                 label=f"AUC = {metrics['roc_auc']:.3f}")
        ax2.plot([0, 1], [0, 1], 'r--', lw=1)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

    # 3. Precision-Recall Curve
    ax3 = fig.add_subplot(gs[0, 2])
    if 'precision_curve' in metrics and 'recall_curve' in metrics:
        ax3.plot(metrics['recall_curve'], metrics['precision_curve'], 'g-', lw=2,
                 label=f"AUC = {metrics['pr_auc']:.3f}")
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)

    # 4. Metrics Bar Chart
    ax4 = fig.add_subplot(gs[1, 0])
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [metrics['accuracy'], metrics['precision'],
                     metrics['recall'], metrics['f1_score']]
    colors_bar = ['#2A9D8F', '#E76F51', '#F4A261', '#264653']
    bars = ax4.barh(metric_names, metric_values, color=colors_bar, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, metric_values):
        ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                 va='center', fontweight='bold')
    ax4.set_xlim([0, 1.1])
    ax4.set_title('Key Metrics', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    # 5. Prediction Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(y_proba[y_true == 0], bins=30, alpha=0.6, label='Negative',
             color='blue', edgecolor='black', density=True)
    ax5.hist(y_proba[y_true == 1], bins=30, alpha=0.6, label='Positive',
             color='red', edgecolor='black', density=True)
    ax5.axvline(0.5, color='green', linestyle='--', lw=2, label='Threshold')
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Prediction Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. Cost Analysis
    ax6 = fig.add_subplot(gs[1, 2])
    cost_components = ['False Negatives\n(100x)', 'False Positives\n(1x)', 'Total Cost']
    costs = [metrics['FN'] * 100, metrics['FP'] * 1, metrics['operational_cost']]
    colors_cost = ['#E63946', '#F4A261', '#264653']
    bars = ax6.bar(cost_components, costs, color=colors_cost, edgecolor='black', linewidth=1.5)
    for bar, cost in zip(bars, costs):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.02,
                 f'{cost:.0f}', ha='center', fontweight='bold')
    ax6.set_ylabel('Cost')
    ax6.set_title('Operational Cost Breakdown', fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)

    # 7. Performance Summary Text
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    summary_text = f"""
    MODEL PERFORMANCE SUMMARY

    Classification Metrics:
    • Accuracy:  {metrics['accuracy']:.4f}  |  Precision: {metrics['precision']:.4f}  |  Recall: {metrics['recall']:.4f}  |  F1-Score: {metrics['f1_score']:.4f}

    Confusion Matrix:
    • True Positives: {metrics['TP']}  |  False Positives: {metrics['FP']}  |  True Negatives: {metrics['TN']}  |  False Negatives: {metrics['FN']}

    Operational Cost:
    • Total Cost: {metrics['operational_cost']:.0f}  (FN × 100 + FP × 1)
    """

    if 'roc_auc' in metrics:
        summary_text += f"\n    Probabilistic Metrics:\n    • ROC-AUC: {metrics['roc_auc']:.4f}  |  PR-AUC: {metrics['pr_auc']:.4f}"

    ax7.text(0.5, 0.5, summary_text, ha='center', va='center',
             fontsize=12, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'{model_name} - Comprehensive Evaluation Dashboard',
                 fontweight='bold', fontsize=18, y=0.995)

    import os
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{model_name.lower()}_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# INTERACTIVE VISUALIZATION FUNCTIONS (using Plotly)
# ============================================================================

def plot_interactive_roc_curve(y_true, y_proba, model_name='Model', save_path='plots'):
    """
    Create interactive ROC curve using Plotly.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    model_name : str
        Model name for title
    save_path : str
        Directory to save HTML file
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)

    fig = go.Figure()

    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines+markers',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=4),
        hovertemplate='<b>FPR:</b> %{x:.4f}<br><b>TPR:</b> %{y:.4f}<br><b>Threshold:</b> %{text:.4f}<extra></extra>',
        text=thresholds
    ))

    # Add random classifier line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#E63946', width=2, dash='dash')
    ))

    # Add optimal threshold point
    fig.add_trace(go.Scatter(
        x=[fpr[optimal_idx]], y=[tpr[optimal_idx]],
        mode='markers',
        name=f'Optimal (Threshold={thresholds[optimal_idx]:.3f})',
        marker=dict(color='red', size=15, symbol='star'),
        hovertemplate=f'<b>Optimal Point</b><br>FPR: {fpr[optimal_idx]:.4f}<br>TPR: {tpr[optimal_idx]:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'{model_name} - Interactive ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        hovermode='closest',
        width=900,
        height=700,
        template='plotly_white',
        font=dict(size=12),
        showlegend=True,
        legend=dict(x=0.6, y=0.1)
    )

    import os
    os.makedirs(save_path, exist_ok=True)
    fig.write_html(f"{save_path}/{model_name.lower()}_interactive_roc.html")
    print(f"Interactive ROC curve saved to: {save_path}/{model_name.lower()}_interactive_roc.html")


def plot_interactive_prediction_timeline(y_true, y_pred, y_proba=None, title='Interactive Prediction Timeline', save_path='plots'):
    """
    Create interactive timeline visualization using Plotly.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    y_proba : array-like, optional
        Predicted probabilities
    title : str
        Plot title
    save_path : str
        Directory to save HTML file
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return

    time_steps = np.arange(len(y_true))

    fig = go.Figure()

    # Add actual values
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=y_true,
        mode='lines+markers',
        name='Actual',
        line=dict(color='#264653', width=3),
        marker=dict(size=5),
        hovertemplate='<b>Time:</b> %{x}<br><b>Actual:</b> %{y}<extra></extra>'
    ))

    # Add predicted values
    if y_proba is not None:
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=y_proba,
            mode='lines+markers',
            name='Predicted Probability',
            line=dict(color='#E76F51', width=3),
            marker=dict(size=5),
            hovertemplate='<b>Time:</b> %{x}<br><b>Probability:</b> %{y:.4f}<extra></extra>'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=y_pred,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#E76F51', width=3),
            marker=dict(size=5),
            hovertemplate='<b>Time:</b> %{x}<br><b>Predicted:</b> %{y}<extra></extra>'
        ))

    # Highlight clogging events
    clog_events = time_steps[y_true == 1]
    if len(clog_events) > 0:
        fig.add_trace(go.Scatter(
            x=clog_events,
            y=y_true[y_true == 1],
            mode='markers',
            name='Clogging Events',
            marker=dict(color='red', size=12, symbol='x'),
            hovertemplate='<b>Clogging Event at Time:</b> %{x}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Time Step',
        yaxis_title='Value / Probability',
        hovermode='x unified',
        width=1400,
        height=600,
        template='plotly_white',
        font=dict(size=12),
        showlegend=True
    )

    import os
    os.makedirs(save_path, exist_ok=True)
    fig.write_html(f"{save_path}/interactive_timeline.html")
    print(f"Interactive timeline saved to: {save_path}/interactive_timeline.html")


def plot_interactive_confusion_matrix(cm, model_name='Model', save_path='plots'):
    """
    Create interactive confusion matrix heatmap using Plotly.

    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    model_name : str
        Model name for title
    save_path : str
        Directory to save HTML file
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    labels = ['Healthy', 'Clogging']

    # Create text annotations
    text = [[f'{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)' for j in range(cm.shape[1])]
            for i in range(cm.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=16),
        colorscale='Blues',
        showscale=True,
        hovertemplate='<b>True:</b> %{y}<br><b>Predicted:</b> %{x}<br><b>Count:</b> %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=f'{model_name} - Interactive Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=700,
        height=700,
        template='plotly_white',
        font=dict(size=13)
    )

    import os
    os.makedirs(save_path, exist_ok=True)
    fig.write_html(f"{save_path}/{model_name.lower()}_interactive_confusion_matrix.html")
    print(f"Interactive confusion matrix saved to: {save_path}/{model_name.lower()}_interactive_confusion_matrix.html")


def plot_interactive_threshold_analysis(y_true, y_proba, model_name='Model', save_path='plots'):
    """
    Create interactive threshold analysis using Plotly.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    model_name : str
        Model name
    save_path : str
        Directory to save HTML file
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return

    thresholds = np.linspace(0, 1, 200)
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode='lines', name='Precision',
                             line=dict(color='#2A9D8F', width=3)))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls, mode='lines', name='Recall',
                             line=dict(color='#E76F51', width=3)))
    fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode='lines', name='F1-Score',
                             line=dict(color='#264653', width=3)))
    fig.add_trace(go.Scatter(x=thresholds, y=accuracies, mode='lines', name='Accuracy',
                             line=dict(color='#F4A261', width=3)))

    # Add optimal threshold line
    fig.add_vline(x=optimal_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Optimal: {optimal_threshold:.3f}",
                  annotation_position="top")

    fig.update_layout(
        title=f'{model_name} - Interactive Threshold Analysis',
        xaxis_title='Classification Threshold',
        yaxis_title='Score',
        hovermode='x unified',
        width=1200,
        height=700,
        template='plotly_white',
        font=dict(size=12),
        showlegend=True
    )

    import os
    os.makedirs(save_path, exist_ok=True)
    fig.write_html(f"{save_path}/{model_name.lower()}_interactive_threshold_analysis.html")
    print(f"Interactive threshold analysis saved to: {save_path}/{model_name.lower()}_interactive_threshold_analysis.html")

    return optimal_threshold


def create_interactive_dashboard(metrics, y_true, y_pred, y_proba, model_name='Model', save_path='plots'):
    """
    Create comprehensive interactive dashboard using Plotly.

    Parameters:
    -----------
    metrics : dict
        Metrics dictionary
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like
        Predicted probabilities
    model_name : str
        Model name
    save_path : str
        Directory to save HTML file
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return

    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve',
                       'Key Metrics', 'Prediction Distribution', 'Cost Breakdown'),
        specs=[[{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'bar'}, {'type': 'histogram'}, {'type': 'bar'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. Confusion Matrix
    cm = metrics['confusion_matrix']
    labels = ['Healthy', 'Clogging']
    fig.add_trace(go.Heatmap(z=cm, x=labels, y=labels, colorscale='Blues', showscale=False),
                  row=1, col=1)

    # 2. ROC Curve
    if 'fpr' in metrics and 'tpr' in metrics:
        fig.add_trace(go.Scatter(x=metrics['fpr'], y=metrics['tpr'], mode='lines',
                                name=f"ROC (AUC={metrics['roc_auc']:.3f})",
                                line=dict(color='blue', width=2)),
                     row=1, col=2)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                                line=dict(color='red', width=1, dash='dash')),
                     row=1, col=2)

    # 3. PR Curve
    if 'precision_curve' in metrics and 'recall_curve' in metrics:
        fig.add_trace(go.Scatter(x=metrics['recall_curve'], y=metrics['precision_curve'],
                                mode='lines', name=f"PR (AUC={metrics['pr_auc']:.3f})",
                                line=dict(color='green', width=2)),
                     row=1, col=3)

    # 4. Key Metrics
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [metrics['accuracy'], metrics['precision'],
                    metrics['recall'], metrics['f1_score']]
    fig.add_trace(go.Bar(y=metric_names, x=metric_values, orientation='h',
                        marker_color=['#2A9D8F', '#E76F51', '#F4A261', '#264653'],
                        showlegend=False),
                 row=2, col=1)

    # 5. Prediction Distribution
    fig.add_trace(go.Histogram(x=y_proba[y_true == 0], name='Negative',
                              marker_color='blue', opacity=0.6, nbinsx=30),
                 row=2, col=2)
    fig.add_trace(go.Histogram(x=y_proba[y_true == 1], name='Positive',
                              marker_color='red', opacity=0.6, nbinsx=30),
                 row=2, col=2)

    # 6. Cost Breakdown
    cost_labels = ['FN (100x)', 'FP (1x)', 'Total']
    cost_values = [metrics['FN'] * 100, metrics['FP'] * 1, metrics['operational_cost']]
    fig.add_trace(go.Bar(x=cost_labels, y=cost_values,
                        marker_color=['#E63946', '#F4A261', '#264653'],
                        showlegend=False),
                 row=2, col=3)

    # Update layout
    fig.update_layout(
        title_text=f'{model_name} - Interactive Evaluation Dashboard',
        title_font_size=20,
        showlegend=True,
        height=900,
        width=1600,
        template='plotly_white'
    )

    import os
    os.makedirs(save_path, exist_ok=True)
    fig.write_html(f"{save_path}/{model_name.lower()}_interactive_dashboard.html")
    print(f"Interactive dashboard saved to: {save_path}/{model_name.lower()}_interactive_dashboard.html")


# ============================================================================
# MODEL INTERPRETABILITY FUNCTIONS (SHAP, LIME, PDP)
# ============================================================================

def plot_shap_summary(model, X, feature_names=None, model_name='Model', max_display=20, save_path='plots'):
    """
    Create SHAP summary plot showing global feature importance.

    Parameters:
    -----------
    model : sklearn model
        Trained model (tree-based models work best)
    X : array-like
        Feature matrix (preferably test set)
    feature_names : list, optional
        Names of features
    model_name : str
        Model name for title
    max_display : int
        Maximum number of features to display
    save_path : str
        Directory to save plot
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Install with: pip install shap")
        return

    try:
        # Convert to DataFrame if feature names provided
        if feature_names is not None:
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Create SHAP explainer
        # Use TreeExplainer for tree-based models, otherwise use KernelExplainer
        try:
            explainer = shap.TreeExplainer(model)
        except:
            # Fallback to KernelExplainer (slower but works for any model)
            # Sample background data for efficiency
            background = shap.sample(X_df, min(100, len(X_df)))
            explainer = shap.KernelExplainer(model.predict_proba, background)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_df)

        # Handle multi-class output (take positive class for binary)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        # Create summary plot
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, X_df, max_display=max_display, show=False)
        plt.title(f'{model_name} - SHAP Feature Importance Summary',
                  fontweight='bold', fontsize=16, pad=20)
        plt.tight_layout()

        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{model_name.lower()}_shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"SHAP summary plot saved to: {save_path}/{model_name.lower()}_shap_summary.png")

        return explainer, shap_values

    except Exception as e:
        print(f"Error creating SHAP summary plot: {str(e)}")
        return None, None


def plot_shap_dependence(shap_values, X, feature_names, feature_idx=0,
                         interaction_idx='auto', model_name='Model', save_path='plots'):
    """
    Create SHAP dependence plot showing how a feature affects predictions.

    Parameters:
    -----------
    shap_values : array-like
        SHAP values from explainer
    X : array-like
        Feature matrix
    feature_names : list
        Names of features
    feature_idx : int or str
        Index or name of feature to plot
    interaction_idx : int, str, or 'auto'
        Feature to color by for interactions
    model_name : str
        Model name for title
    save_path : str
        Directory to save plot
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Install with: pip install shap")
        return

    try:
        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)

        # Get feature name and index
        if isinstance(feature_idx, (int, np.integer)):
            # Convert numpy integers to Python int
            feature_idx = int(feature_idx)
            feature_name = feature_names[feature_idx]
        elif isinstance(feature_idx, str):
            feature_name = feature_idx
            feature_idx = feature_names.index(feature_name)
        else:
            # Handle any other type (e.g., np.int64)
            feature_idx = int(feature_idx)
            feature_name = feature_names[feature_idx]

        # Create dependence plot
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(feature_idx, shap_values, X_df,
                            interaction_index=interaction_idx, show=False)
        plt.title(f'{model_name} - SHAP Dependence: {feature_name}',
                  fontweight='bold', fontsize=15, pad=20)
        plt.tight_layout()

        import os
        os.makedirs(save_path, exist_ok=True)
        safe_feature_name = feature_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        plt.savefig(f"{save_path}/{model_name.lower()}_shap_dependence_{safe_feature_name}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"SHAP dependence plot saved for feature: {feature_name}")

    except Exception as e:
        print(f"Error creating SHAP dependence plot: {str(e)}")


def plot_shap_waterfall(explainer, shap_values, X, feature_names, sample_idx=0,
                        model_name='Model', save_path='plots'):
    """
    Create SHAP waterfall plot explaining a single prediction.

    Parameters:
    -----------
    explainer : shap.Explainer
        SHAP explainer object
    shap_values : array-like
        SHAP values from explainer
    X : array-like
        Feature matrix
    feature_names : list
        Names of features
    sample_idx : int
        Index of sample to explain
    model_name : str
        Model name for title
    save_path : str
        Directory to save plot
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Install with: pip install shap")
        return

    try:
        # Create waterfall plot for single prediction
        plt.figure(figsize=(12, 8))

        # Handle multi-output SHAP values (e.g., binary classification)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # Already handled - should be class 1 (positive class)
            shap_vals_sample = shap_values[sample_idx]
        elif shap_values.ndim == 3:
            # Shape: (n_samples, n_features, n_classes) - take positive class
            shap_vals_sample = shap_values[sample_idx, :, 1]
        elif shap_values.ndim == 2:
            # Shape: (n_samples, n_features) - normal case
            shap_vals_sample = shap_values[sample_idx]
        else:
            shap_vals_sample = shap_values[sample_idx]

        # Convert to Explanation object if needed
        if hasattr(shap, 'Explanation'):
            X_df = pd.DataFrame(X, columns=feature_names)

            # Get base value
            base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]

            explanation = shap.Explanation(
                values=shap_vals_sample,
                base_values=base_value,
                data=X_df.iloc[sample_idx] if sample_idx < len(X_df) else X_df.iloc[0],
                feature_names=feature_names
            )
            shap.plots.waterfall(explanation, show=False)
        else:
            # Fallback for older SHAP versions
            base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]

            shap.plots._waterfall.waterfall_legacy(
                base_value,
                shap_vals_sample,
                feature_names=feature_names,
                show=False
            )

        plt.title(f'{model_name} - SHAP Waterfall (Sample {sample_idx})',
                  fontweight='bold', fontsize=15, pad=20)
        plt.tight_layout()

        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{model_name.lower()}_shap_waterfall_sample_{sample_idx}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"SHAP waterfall plot saved for sample {sample_idx}")

    except Exception as e:
        print(f"Error creating SHAP waterfall plot: {str(e)}")


def plot_shap_force(explainer, shap_values, X, feature_names, sample_idx=0,
                    model_name='Model', save_path='plots'):
    """
    Create SHAP force plot (interactive HTML) for a single prediction.

    Parameters:
    -----------
    explainer : shap.Explainer
        SHAP explainer object
    shap_values : array-like
        SHAP values from explainer
    X : array-like
        Feature matrix
    feature_names : list
        Names of features
    sample_idx : int
        Index of sample to explain
    model_name : str
        Model name for title
    save_path : str
        Directory to save HTML file
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Install with: pip install shap")
        return

    try:
        # Create force plot
        X_df = pd.DataFrame(X, columns=feature_names)

        base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        force_plot = shap.force_plot(
            base_value,
            shap_values[sample_idx],
            X_df.iloc[sample_idx],
            feature_names=feature_names
        )

        import os
        os.makedirs(save_path, exist_ok=True)
        shap.save_html(f"{save_path}/{model_name.lower()}_shap_force_sample_{sample_idx}.html", force_plot)

        print(f"SHAP force plot saved for sample {sample_idx}")

    except Exception as e:
        print(f"Error creating SHAP force plot: {str(e)}")


def plot_shap_decision(explainer, shap_values, X, feature_names, sample_indices=None,
                       model_name='Model', save_path='plots'):
    """
    Create SHAP decision plot showing prediction paths for multiple samples.

    Parameters:
    -----------
    explainer : shap.Explainer
        SHAP explainer object
    shap_values : array-like
        SHAP values from explainer
    X : array-like
        Feature matrix
    feature_names : list
        Names of features
    sample_indices : list, optional
        Indices of samples to include (default: first 10)
    model_name : str
        Model name for title
    save_path : str
        Directory to save plot
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Install with: pip install shap")
        return

    try:
        if sample_indices is None:
            sample_indices = list(range(min(10, len(X))))

        # Ensure sample indices are within bounds
        sample_indices = [idx for idx in sample_indices if idx < len(X)]
        if len(sample_indices) == 0:
            print("No valid sample indices for decision plot")
            return

        plt.figure(figsize=(14, 10))

        base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        # Handle multi-output SHAP values
        if shap_values.ndim == 3:
            # Shape: (n_samples, n_features, n_classes) - take positive class
            shap_vals_subset = shap_values[sample_indices, :, 1]
        elif shap_values.ndim == 2:
            # Shape: (n_samples, n_features) - normal case
            shap_vals_subset = shap_values[sample_indices, :]
        else:
            shap_vals_subset = shap_values[sample_indices]

        # Get corresponding feature values
        X_subset = X[sample_indices] if isinstance(X, np.ndarray) else X.iloc[sample_indices].values

        shap.decision_plot(
            base_value,
            shap_vals_subset,
            features=X_subset,
            feature_names=feature_names,
            show=False
        )

        plt.title(f'{model_name} - SHAP Decision Plot',
                  fontweight='bold', fontsize=15, pad=20)
        plt.tight_layout()

        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{model_name.lower()}_shap_decision.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"SHAP decision plot saved")

    except Exception as e:
        print(f"Error creating SHAP decision plot: {str(e)}")


def plot_lime_explanation(model, X_train, X_test, feature_names, sample_idx=0,
                          class_names=['Healthy', 'Clogging'], model_name='Model',
                          save_path='plots', num_features=20):
    """
    Create LIME explanation for a single prediction.

    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_train : array-like
        Training data (used as background)
    X_test : array-like
        Test data
    feature_names : list
        Names of features
    sample_idx : int
        Index of sample to explain
    class_names : list
        Names of classes
    model_name : str
        Model name for title
    save_path : str
        Directory to save plot
    num_features : int
        Number of features to show
    """
    if not LIME_AVAILABLE:
        print("LIME not available. Install with: pip install lime")
        return

    try:
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            random_state=42
        )

        # Explain prediction
        exp = explainer.explain_instance(
            X_test[sample_idx],
            model.predict_proba,
            num_features=num_features
        )

        # Create visualization
        fig = exp.as_pyplot_figure()
        plt.title(f'{model_name} - LIME Explanation (Sample {sample_idx})',
                  fontweight='bold', fontsize=15, pad=20)
        plt.tight_layout()

        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{model_name.lower()}_lime_sample_{sample_idx}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Save HTML version
        exp.save_to_file(f"{save_path}/{model_name.lower()}_lime_sample_{sample_idx}.html")

        print(f"LIME explanation saved for sample {sample_idx}")

        return exp

    except Exception as e:
        print(f"Error creating LIME explanation: {str(e)}")
        return None


def plot_partial_dependence(model, X, feature_names, features_to_plot=None,
                            model_name='Model', save_path='plots', grid_resolution=50):
    """
    Create Partial Dependence Plots (PDP) showing marginal effect of features.

    Parameters:
    -----------
    model : sklearn model
        Trained model
    X : array-like
        Feature matrix
    feature_names : list
        Names of features
    features_to_plot : list, optional
        Indices or names of features to plot (default: top 4)
    model_name : str
        Model name for title
    save_path : str
        Directory to save plot
    grid_resolution : int
        Number of grid points for PDP
    """
    try:
        from sklearn.inspection import PartialDependenceDisplay

        if features_to_plot is None:
            # Plot top 4 most important features
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[-4:][::-1]
                features_to_plot = [int(i) for i in top_indices]
            else:
                features_to_plot = list(range(min(4, len(feature_names))))

        # Convert feature names to indices if needed
        feature_indices = []
        for f in features_to_plot:
            if isinstance(f, str):
                feature_indices.append(feature_names.index(f))
            else:
                feature_indices.append(f)

        # Create PDP
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        display = PartialDependenceDisplay.from_estimator(
            model, X, feature_indices[:4],
            feature_names=feature_names,
            grid_resolution=grid_resolution,
            ax=axes
        )

        fig.suptitle(f'{model_name} - Partial Dependence Plots',
                    fontweight='bold', fontsize=16, y=0.995)
        plt.tight_layout()

        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{model_name.lower()}_partial_dependence.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Partial dependence plots saved")

    except Exception as e:
        print(f"Error creating partial dependence plots: {str(e)}")


def plot_permutation_importance(model, X, y, feature_names, model_name='Model',
                                n_repeats=10, top_n=20, save_path='plots'):
    """
    Create permutation importance plot (model-agnostic feature importance).

    Parameters:
    -----------
    model : sklearn model
        Trained model
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    feature_names : list
        Names of features
    model_name : str
        Model name for title
    n_repeats : int
        Number of times to permute each feature
    top_n : int
        Number of top features to display
    save_path : str
        Directory to save plot
    """
    try:
        from sklearn.inspection import permutation_importance

        # Calculate permutation importance
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

        # Sort features by importance
        sorted_idx = result.importances_mean.argsort()[-top_n:]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot with error bars
        ax.barh(range(len(sorted_idx)),
                result.importances_mean[sorted_idx],
                xerr=result.importances_std[sorted_idx],
                color='steelblue',
                alpha=0.8,
                edgecolor='black',
                linewidth=1.5)

        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Permutation Importance (decrease in score)', fontweight='bold', fontsize=13)
        ax.set_title(f'{model_name} - Permutation Feature Importance (Top {top_n})',
                    fontweight='bold', fontsize=15, pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        plt.tight_layout()

        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{model_name.lower()}_permutation_importance.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Permutation importance plot saved")

        return result

    except Exception as e:
        print(f"Error creating permutation importance plot: {str(e)}")
        return None


def create_interpretability_dashboard(model, X_train, X_test, y_test, feature_names,
                                     model_name='Model', save_path='plots',
                                     sample_indices=[0, 1, 2]):
    """
    Create comprehensive interpretability dashboard with multiple explanation methods.

    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_train : array-like
        Training data
    X_test : array-like
        Test data
    y_test : array-like
        Test labels
    feature_names : list
        Names of features
    model_name : str
        Model name for title
    save_path : str
        Directory to save plots
    sample_indices : list
        Indices of samples to explain in detail
    """
    print(f"\n{'='*60}")
    print(f"CREATING INTERPRETABILITY DASHBOARD FOR {model_name.upper()}")
    print(f"{'='*60}\n")

    results = {}

    # 1. SHAP Analysis
    if SHAP_AVAILABLE:
        print("[1/5] Generating SHAP visualizations...")
        explainer, shap_values = plot_shap_summary(
            model, X_test, feature_names, model_name, save_path=save_path
        )

        if explainer is not None and shap_values is not None:
            results['shap_explainer'] = explainer
            results['shap_values'] = shap_values

            # SHAP dependence for top features
            if hasattr(model, 'feature_importances_'):
                top_feature_idx = np.argmax(model.feature_importances_)
                plot_shap_dependence(
                    shap_values, X_test, feature_names,
                    feature_idx=top_feature_idx,
                    model_name=model_name, save_path=save_path
                )

            # SHAP waterfall for sample predictions
            for idx in sample_indices[:3]:
                if idx < len(X_test):
                    plot_shap_waterfall(
                        explainer, shap_values, X_test, feature_names,
                        sample_idx=idx, model_name=model_name, save_path=save_path
                    )

            # SHAP decision plot
            plot_shap_decision(
                explainer, shap_values, X_test, feature_names,
                sample_indices=sample_indices[:10],
                model_name=model_name, save_path=save_path
            )

    # 2. LIME Analysis
    if LIME_AVAILABLE:
        print("[2/5] Generating LIME explanations...")
        for idx in sample_indices[:2]:
            if idx < len(X_test):
                plot_lime_explanation(
                    model, X_train, X_test, feature_names,
                    sample_idx=idx, model_name=model_name, save_path=save_path
                )

    # 3. Partial Dependence Plots
    print("[3/5] Creating Partial Dependence Plots...")
    plot_partial_dependence(
        model, X_test, feature_names,
        model_name=model_name, save_path=save_path
    )

    # 4. Permutation Importance
    print("[4/5] Calculating Permutation Importance...")
    perm_importance = plot_permutation_importance(
        model, X_test, y_test, feature_names,
        model_name=model_name, save_path=save_path
    )
    results['permutation_importance'] = perm_importance

    # 5. Feature Importance Comparison
    if hasattr(model, 'feature_importances_'):
        print("[5/5] Creating Feature Importance Comparison...")
        plot_feature_importance(
            model.feature_importances_, feature_names,
            model_name=f"{model_name}_TreeBased",
            save_path=save_path
        )

    print(f"\n{'='*60}")
    print(f"INTERPRETABILITY DASHBOARD COMPLETE")
    print(f"All visualizations saved to: {save_path}/")
    print(f"{'='*60}\n")

    return results


def explain_prediction(model, X_sample, feature_names, explainer=None, shap_values=None,
                      X_train=None, model_name='Model', save_path='plots', sample_idx=0):
    """
    Comprehensive explanation of a single prediction using multiple methods.

    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_sample : array-like
        Single sample to explain (1D or 2D with shape (1, n_features))
    feature_names : list
        Names of features
    explainer : shap.Explainer, optional
        Pre-computed SHAP explainer
    shap_values : array-like, optional
        Pre-computed SHAP values
    X_train : array-like, optional
        Training data for LIME
    model_name : str
        Model name for title
    save_path : str
        Directory to save explanation
    sample_idx : int
        Sample index for naming

    Returns:
    --------
    explanation : dict
        Dictionary with explanation results from different methods
    """
    explanation = {}

    # Ensure X_sample is 2D
    if len(X_sample.shape) == 1:
        X_sample = X_sample.reshape(1, -1)

    # Get prediction
    pred_proba = model.predict_proba(X_sample)[0]
    pred_class = model.predict(X_sample)[0]

    explanation['prediction'] = pred_class
    explanation['probability'] = pred_proba

    print(f"\n{'='*60}")
    print(f"EXPLAINING PREDICTION FOR SAMPLE {sample_idx}")
    print(f"{'='*60}")
    print(f"Predicted Class: {pred_class} ({'Clogging' if pred_class == 1 else 'Healthy'})")
    print(f"Probability: {pred_proba[1]:.4f} (Clogging) | {pred_proba[0]:.4f} (Healthy)")
    print(f"{'='*60}\n")

    # SHAP explanation
    if SHAP_AVAILABLE and (explainer is not None or shap_values is not None):
        print("Generating SHAP explanation...")
        if shap_values is None:
            X_df = pd.DataFrame(X_sample, columns=feature_names)
            shap_values = explainer.shap_values(X_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        plot_shap_waterfall(explainer, shap_values, X_sample, feature_names,
                          sample_idx=0, model_name=model_name, save_path=save_path)
        explanation['shap_values'] = shap_values[0] if shap_values.ndim > 1 else shap_values

    # LIME explanation
    if LIME_AVAILABLE and X_train is not None:
        print("Generating LIME explanation...")
        lime_exp = plot_lime_explanation(
            model, X_train, X_sample, feature_names,
            sample_idx=0, model_name=model_name, save_path=save_path
        )
        explanation['lime'] = lime_exp

    # Feature contributions (if tree-based model)
    if hasattr(model, 'feature_importances_'):
        explanation['feature_importances'] = model.feature_importances_

    print(f"\nExplanation saved to: {save_path}/\n")

    return explanation


# ============================================================
# Risk Level Visualization Functions
# ============================================================

def plot_risk_level_distribution(risk_levels, risk_labels=None, save_path='plots', model_name='Risk_Level'):
    """
    Plot distribution of risk levels.

    Parameters:
    -----------
    risk_levels : np.ndarray
        Risk levels [0, 1, 2, 3]
    risk_labels : list, optional
        Human-readable labels for each level
    save_path : str
        Directory to save plot
    model_name : str
        Model name for filename
    """
    from config import CONFIG

    if risk_labels is None:
        labels_dict = CONFIG.get('risk_levels', {}).get('labels', {
            0: 'LOW', 1: 'MODERATE', 2: 'HIGH', 3: 'CRITICAL'
        })
        risk_labels = [labels_dict[i] for i in range(4)]

    colors_dict = CONFIG.get('risk_levels', {}).get('colors', {
        0: '#28a745', 1: '#ffc107', 2: '#fd7e14', 3: '#dc3545'
    })
    colors = [colors_dict[i] for i in range(4)]

    # Count occurrences
    unique, counts = np.unique(risk_levels, return_counts=True)
    percentages = 100 * counts / len(risk_levels)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(unique, counts, color=[colors[i] for i in unique], alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Risk Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Risk Level Distribution - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(risk_labels)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, f'{model_name}_risk_distribution.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"[OK] Risk distribution plot saved: {filepath}")


def plot_risk_level_confusion_matrix(cm, save_path='plots', model_name='Risk_Level'):
    """
    Plot confusion matrix for risk level predictions.

    Parameters:
    -----------
    cm : np.ndarray
        Confusion matrix, shape (num_levels, num_levels)
    save_path : str
        Directory to save plot
    model_name : str
        Model name for filename
    """
    from config import CONFIG

    labels_dict = CONFIG.get('risk_levels', {}).get('labels', {
        0: 'LOW', 1: 'MODERATE', 2: 'HIGH', 3: 'CRITICAL'
    })
    labels = [labels_dict[i] for i in range(4)]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='YlOrRd')
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           xlabel='Predicted Risk Level',
           ylabel='True Risk Level',
           title=f'Risk Level Confusion Matrix - {model_name}')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, f'{model_name}_risk_confusion_matrix.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"[OK] Risk confusion matrix saved: {filepath}")


def plot_risk_level_timeline(risk_levels, risk_scores, y_true=None, save_path='plots', model_name='Risk_Level'):
    """
    Plot risk levels and scores over time.

    Parameters:
    -----------
    risk_levels : np.ndarray
        Predicted risk levels [0, 1, 2, 3]
    risk_scores : np.ndarray
        Continuous risk scores [0-1]
    y_true : np.ndarray, optional
        True binary labels for background shading
    save_path : str
        Directory to save plot
    model_name : str
        Model name for filename
    """
    from config import CONFIG

    colors_dict = CONFIG.get('risk_levels', {}).get('colors', {
        0: '#28a745', 1: '#ffc107', 2: '#fd7e14', 3: '#dc3545'
    })
    labels_dict = CONFIG.get('risk_levels', {}).get('labels', {
        0: 'LOW', 1: 'MODERATE', 2: 'HIGH', 3: 'CRITICAL'
    })

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    time_steps = np.arange(len(risk_levels))

    # Top plot: Risk levels (discrete)
    ax1_colors = [colors_dict[level] for level in risk_levels]
    ax1.scatter(time_steps, risk_levels, c=ax1_colors, s=20, alpha=0.6)

    # Add background shading for true labels if provided
    if y_true is not None:
        clogged_regions = np.where(y_true == 1)[0]
        if len(clogged_regions) > 0:
            ax1.axvspan(clogged_regions[0], len(y_true), alpha=0.15, color='red', label='True Clogging')

    ax1.set_ylabel('Risk Level', fontsize=12, fontweight='bold')
    ax1.set_title(f'Risk Level Timeline - {model_name}', fontsize=14, fontweight='bold')
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([labels_dict[i] for i in range(4)])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Bottom plot: Risk scores (continuous)
    ax2.plot(time_steps, risk_scores, color='darkblue', linewidth=1.5, alpha=0.7, label='Risk Score')

    # Add threshold lines
    thresholds = CONFIG.get('risk_levels', {}).get('thresholds', {
        'low': 0.25, 'moderate': 0.50, 'high': 0.75, 'critical': 1.00
    })
    ax2.axhline(thresholds['low'], color=colors_dict[0], linestyle='--', linewidth=1, alpha=0.5, label='LOW threshold')
    ax2.axhline(thresholds['moderate'], color=colors_dict[1], linestyle='--', linewidth=1, alpha=0.5, label='MODERATE threshold')
    ax2.axhline(thresholds['high'], color=colors_dict[2], linestyle='--', linewidth=1, alpha=0.5, label='HIGH threshold')

    # Add background shading
    if y_true is not None:
        if len(clogged_regions) > 0:
            ax2.axvspan(clogged_regions[0], len(y_true), alpha=0.15, color='red', label='True Clogging')

    ax2.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Risk Score (Probability)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Continuous Risk Scores - {model_name}', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    plt.tight_layout()

    # Save
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, f'{model_name}_risk_timeline.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"[OK] Risk timeline plot saved: {filepath}")


def plot_risk_calibration_curve(y_true_binary, risk_scores, n_bins=10, save_path='plots', model_name='Risk_Level'):
    """
    Plot calibration curve for risk scores.

    Shows how well predicted probabilities match actual outcomes.

    Parameters:
    -----------
    y_true_binary : np.ndarray
        True binary labels (0=healthy, 1=clogging)
    risk_scores : np.ndarray
        Continuous risk scores [0-1]
    n_bins : int
        Number of bins for calibration
    save_path : str
        Directory to save plot
    model_name : str
        Model name for filename
    """
    from sklearn.calibration import calibration_curve

    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true_binary, risk_scores, n_bins=n_bins, strategy='uniform'
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=2,
            label='Model Calibration', color='darkblue')

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Perfect Calibration')

    ax.set_xlabel('Mean Predicted Risk Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Positives (True Clogging Rate)', fontsize=12, fontweight='bold')
    ax.set_title(f'Risk Score Calibration Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, f'{model_name}_risk_calibration.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"[OK] Risk calibration curve saved: {filepath}")


def evaluate_risk_level_model(y_true_binary, risk_results, verbose=True):
    """
    Comprehensive evaluation of risk level predictions.

    Parameters:
    -----------
    y_true_binary : np.ndarray
        True binary labels (0=healthy, 1=clogging)
    risk_results : dict
        Results from predictor.predict_risk_level()
        Must contain: 'risk_levels', 'risk_scores', 'risk_labels'
    verbose : bool
        Print detailed metrics

    Returns:
    --------
    metrics : dict
        Comprehensive metrics including:
        - Binary classification metrics
        - Risk level metrics
        - Calibration metrics
    """
    from utils import compute_risk_level_metrics, map_binary_to_risk_level
    from config import CONFIG

    risk_levels = risk_results['risk_levels']
    risk_scores = risk_results['risk_scores']

    # Convert true binary labels to risk levels
    y_true_levels = map_binary_to_risk_level(y_true_binary, y_proba=None, config=CONFIG)

    # Compute risk level metrics
    risk_metrics = compute_risk_level_metrics(y_true_levels, risk_levels, config=CONFIG)

    # Also compute binary classification metrics using threshold
    threshold = CONFIG.get('risk_levels', {}).get('thresholds', {}).get('high', 0.75)
    y_pred_binary = (risk_scores >= threshold).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    binary_metrics = {
        'accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
        'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
        'roc_auc': roc_auc_score(y_true_binary, risk_scores) if len(np.unique(y_true_binary)) > 1 else 0,
    }

    # Combine metrics
    metrics = {
        'binary_metrics': binary_metrics,
        'risk_level_metrics': risk_metrics,
        'risk_levels': risk_levels,
        'risk_scores': risk_scores,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"RISK LEVEL EVALUATION")
        print(f"{'='*60}")
        print(f"\n=== Binary Classification (using HIGH threshold={threshold:.2f}) ===")
        print(f"  Accuracy:  {binary_metrics['accuracy']:.4f}")
        print(f"  Precision: {binary_metrics['precision']:.4f}")
        print(f"  Recall:    {binary_metrics['recall']:.4f}")
        print(f"  F1-Score:  {binary_metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {binary_metrics['roc_auc']:.4f}")

        print(f"\n=== Risk Level Metrics ===")
        print(f"  Overall Accuracy: {risk_metrics['accuracy']:.4f}")
        print(f"  Mean Absolute Error (levels): {risk_metrics['mean_absolute_error']:.2f}")
        print(f"  Operational Cost: {risk_metrics['operational_cost']:.0f}")

        print(f"\n=== Per-Level Accuracy ===")
        labels_dict = CONFIG.get('risk_levels', {}).get('labels', {
            0: 'LOW', 1: 'MODERATE', 2: 'HIGH', 3: 'CRITICAL'
        })
        for level, acc in risk_metrics['per_level_accuracy'].items():
            print(f"  {labels_dict[level]:10s}: {acc:.4f}")

        print(f"\n=== Critical Level Performance ===")
        print(f"  Critical Recall:    {risk_metrics['critical_recall']:.4f}")
        print(f"  Critical Precision: {risk_metrics['critical_precision']:.4f}")

        print(f"{'='*60}\n")

    return metrics


def create_risk_level_dashboard(y_true_binary, risk_results, save_path='plots/risk_levels', model_name='Risk_Level'):
    """
    Create comprehensive dashboard of risk level visualizations.

    Parameters:
    -----------
    y_true_binary : np.ndarray
        True binary labels (0=healthy, 1=clogging)
    risk_results : dict
        Results from predictor.predict_risk_level()
    save_path : str
        Directory to save plots
    model_name : str
        Model name for plots

    Returns:
    --------
    metrics : dict
        Comprehensive evaluation metrics
    """
    print(f"\n{'#'*60}")
    print(f"# RISK LEVEL DASHBOARD GENERATION")
    print(f"{'#'*60}\n")

    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    risk_levels = risk_results['risk_levels']
    risk_scores = risk_results['risk_scores']

    # 1. Distribution plot
    print("[1/5] Creating risk level distribution plot...")
    plot_risk_level_distribution(risk_levels, save_path=save_path, model_name=model_name)

    # 2. Confusion matrix
    print("[2/5] Creating risk level confusion matrix...")
    from utils import compute_risk_level_metrics, map_binary_to_risk_level
    from config import CONFIG

    y_true_levels = map_binary_to_risk_level(y_true_binary, y_proba=None, config=CONFIG)
    risk_metrics = compute_risk_level_metrics(y_true_levels, risk_levels, config=CONFIG)
    cm = risk_metrics['confusion_matrix']
    plot_risk_level_confusion_matrix(cm, save_path=save_path, model_name=model_name)

    # 3. Timeline plot
    print("[3/5] Creating risk level timeline...")
    plot_risk_level_timeline(risk_levels, risk_scores, y_true=y_true_binary,
                             save_path=save_path, model_name=model_name)

    # 4. Calibration curve
    print("[4/5] Creating calibration curve...")
    plot_risk_calibration_curve(y_true_binary, risk_scores, save_path=save_path, model_name=model_name)

    # 5. Evaluation metrics
    print("[5/5] Computing comprehensive metrics...")
    metrics = evaluate_risk_level_model(y_true_binary, risk_results, verbose=True)

    print(f"\n{'#'*60}")
    print(f"# RISK LEVEL DASHBOARD COMPLETE")
    print(f"{'#'*60}")
    print(f"  Visualizations saved to: {save_path}/")
    print(f"{'#'*60}\n")

    return metrics
