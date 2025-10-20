"""
Main script for filter clogging prediction system.
Demonstrates complete workflow from data loading to prediction with enhanced visualizations.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import modular components
from config import CONFIG
from data_processing import (
    load_and_prepare_data,
    compute_target_labels,
    create_ratio_features,
    prepare_features_and_targets
)
from feature_engineering import engineer_all_features
from utils import time_series_split_imbalanced
from predictor import FilterCloggingPredictor
from evaluation import (
    evaluate_classification_model,
    evaluate_regression_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_prediction_timeline,
    plot_calibration_curve,
    plot_threshold_analysis,
    create_evaluation_dashboard,
    plot_interactive_roc_curve,
    plot_interactive_prediction_timeline,
    plot_interactive_confusion_matrix,
    plot_interactive_threshold_analysis,
    create_interactive_dashboard
)


def print_banner(text, char='#', width=80):
    """Print a formatted banner."""
    print(f"\n{char * width}")
    print(f"{char} {text.center(width-4)} {char}")
    print(f"{char * width}\n")


def print_step(step_num, total_steps, description):
    """Print a step header."""
    print(f"\n{'='*80}")
    print(f"[STEP {step_num}/{total_steps}] {description}")
    print(f"{'='*80}")


def main(data_filepath, clog_index=None, output_dir='results', interactive=True):
    """
    Complete workflow for filter clogging prediction with enhanced visualizations.

    Parameters:
    -----------
    data_filepath : str
        Path to CSV/Excel file with filter data
    clog_index : int, optional
        Index where clogging starts (auto-detected if None)
    output_dir : str
        Directory for saving results and plots
    interactive : bool
        Whether to generate interactive visualizations (requires plotly)
    """
    print_banner("FILTER CLOGGING PREDICTION SYSTEM")

    # Create output directories
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. Load and Prepare Data
    # ============================================================
    print_step(1, 7, "Loading and Preparing Data")
    print(f"  Data file: {data_filepath}")

    try:
        df = load_and_prepare_data(data_filepath, config=CONFIG)
    except Exception as e:
        print(f"\n[ERROR] Failed to load data: {str(e)}")
        print("\nPlease ensure your data file contains:")
        print("  - 'differential_pressure' or 'dp' column")
        print("  - 'flow_rate' or 'flowrate' column")
        sys.exit(1)

    # Create basic ratio features
    df = create_ratio_features(df)

    # Compute target labels
    df = compute_target_labels(df, config=CONFIG)

    print(f"  ✓ Data loaded: {len(df)} samples, {len(df.columns)} columns")

    # ============================================================
    # 2. Feature Engineering
    # ============================================================
    print_step(2, 7, "Engineering Features")

    df = engineer_all_features(df, config=CONFIG)

    print(f"  ✓ Feature engineering complete: {len(df.columns)} total features")

    # ============================================================
    # 3. Prepare Features and Targets
    # ============================================================
    print_step(3, 7, "Preparing Features and Targets")

    X, y_class, y_time, y_duration, y_event, feature_names = prepare_features_and_targets(
        df, config=CONFIG
    )

    print(f"  ✓ Feature matrix: {X.shape}")
    print(f"  ✓ Positive samples: {y_class.sum()} ({100*y_class.mean():.2f}%)")

    # ============================================================
    # 4. Create Train/Val/Test Splits
    # ============================================================
    print_step(4, 7, "Creating Data Splits")

    train_idx, val_idx, test_idx, healthy_idx = time_series_split_imbalanced(
        df, clog_index=clog_index, config=CONFIG
    )

    # Split data
    X_train, y_train_class = X.iloc[train_idx], y_class[train_idx]
    X_val, y_val_class = X.iloc[val_idx], y_class[val_idx]
    X_test, y_test_class = X.iloc[test_idx], y_class[test_idx]
    X_healthy = X.iloc[healthy_idx]

    y_train_time = y_time[train_idx]
    y_train_duration = y_duration[train_idx]
    y_train_event = y_event[train_idx]
    y_test_time = y_time[test_idx]

    print(f"  ✓ Train:   {len(train_idx):5d} samples (positive: {y_train_class.sum()})")
    print(f"  ✓ Val:     {len(val_idx):5d} samples (positive: {y_val_class.sum()})")
    print(f"  ✓ Test:    {len(test_idx):5d} samples (positive: {y_test_class.sum()})")
    print(f"  ✓ Healthy: {len(healthy_idx):5d} samples")

    # ============================================================
    # 5. Train Predictor
    # ============================================================
    print_step(5, 7, "Training Prediction Models")

    predictor = FilterCloggingPredictor(config=CONFIG)

    predictor.fit(
        X_train=X_train,
        y_class=y_train_class,
        y_time=y_train_time,
        y_duration=y_train_duration,
        y_event=y_train_event,
        X_val=X_val,
        y_class_val=y_val_class,
        X_healthy=X_healthy,
        verbose=True
    )

    # ============================================================
    # 6. Evaluate on Test Set
    # ============================================================
    print_step(6, 7, "Evaluating on Test Set")

    # Scale test features
    X_test_scaled = predictor.scaler.transform(X_test)

    # Classification predictions
    y_pred_class = predictor.predict(X_test_scaled, use_anomaly=True)
    y_pred_proba = predictor.predict_proba(X_test_scaled, use_anomaly=True)

    # Evaluate classification
    class_metrics = evaluate_classification_model(
        y_test_class, y_pred_class, y_pred_proba[:, 1],
        model_name='Filter Clogging Predictor',
        verbose=True
    )

    # ============================================================
    # 7. Generate Visualizations
    # ============================================================
    print_step(7, 7, "Generating Visualizations")

    print("\n  Creating static plots...")

    # Confusion Matrix
    plot_confusion_matrix(
        class_metrics['confusion_matrix'],
        model_name='Filter_Clogging_Predictor',
        save_path=str(plots_dir)
    )
    print(f"    ✓ Confusion matrix saved")

    # ROC Curve
    if 'roc_auc' in class_metrics:
        plot_roc_curve(
            class_metrics['fpr'],
            class_metrics['tpr'],
            class_metrics['roc_auc'],
            model_name='Filter_Clogging_Predictor',
            save_path=str(plots_dir)
        )
        print(f"    ✓ ROC curve saved")

    # Precision-Recall Curve
    if 'pr_auc' in class_metrics:
        plot_precision_recall_curve(
            class_metrics['precision_curve'],
            class_metrics['recall_curve'],
            class_metrics['pr_auc'],
            model_name='Filter_Clogging_Predictor',
            save_path=str(plots_dir)
        )
        print(f"    ✓ Precision-Recall curve saved")

    # Calibration Curve
    plot_calibration_curve(
        y_test_class,
        y_pred_proba[:, 1],
        model_name='Filter_Clogging_Predictor',
        save_path=str(plots_dir)
    )
    print(f"    ✓ Calibration curve saved")

    # Threshold Analysis
    optimal_threshold = plot_threshold_analysis(
        y_test_class,
        y_pred_proba[:, 1],
        model_name='Filter_Clogging_Predictor',
        save_path=str(plots_dir)
    )
    print(f"    ✓ Threshold analysis saved (optimal: {optimal_threshold:.3f})")

    # Comprehensive risk scores
    risk_scores = predictor.predict_risk_scores(X_test_scaled, use_all_models=True)

    # Timeline plot
    if 'ensemble' in risk_scores:
        plot_prediction_timeline(
            y_test_class, risk_scores['ensemble'],
            title='Ensemble Risk Score Over Time',
            save_path=str(plots_dir)
        )
        print(f"    ✓ Prediction timeline saved")

    # Comprehensive Dashboard
    create_evaluation_dashboard(
        class_metrics,
        y_test_class,
        y_pred_class,
        y_pred_proba[:, 1],
        model_name='Filter_Clogging_Predictor',
        save_path=str(plots_dir)
    )
    print(f"    ✓ Evaluation dashboard saved")

    # Interactive visualizations (if enabled)
    if interactive:
        print("\n  Creating interactive plots...")
        try:
            # Interactive ROC Curve
            plot_interactive_roc_curve(
                y_test_class,
                y_pred_proba[:, 1],
                model_name='Filter_Clogging_Predictor',
                save_path=str(plots_dir)
            )

            # Interactive Confusion Matrix
            plot_interactive_confusion_matrix(
                class_metrics['confusion_matrix'],
                model_name='Filter_Clogging_Predictor',
                save_path=str(plots_dir)
            )

            # Interactive Threshold Analysis
            plot_interactive_threshold_analysis(
                y_test_class,
                y_pred_proba[:, 1],
                model_name='Filter_Clogging_Predictor',
                save_path=str(plots_dir)
            )

            # Interactive Timeline
            if 'ensemble' in risk_scores:
                plot_interactive_prediction_timeline(
                    y_test_class,
                    y_pred_class,
                    y_proba=risk_scores['ensemble'],
                    title='Interactive Ensemble Risk Score Timeline',
                    save_path=str(plots_dir)
                )

            # Interactive Dashboard
            create_interactive_dashboard(
                class_metrics,
                y_test_class,
                y_pred_class,
                y_pred_proba[:, 1],
                model_name='Filter_Clogging_Predictor',
                save_path=str(plots_dir)
            )

        except Exception as e:
            print(f"    [!] Interactive plots skipped: {str(e)}")
            print(f"    [!] Install plotly for interactive visualizations: pip install plotly")

    # Regression evaluation (if available)
    if len(predictor.regression_models) > 0:
        print("\n  Evaluating regression models...")
        y_pred_time = predictor.predict_time_to_clog(X_test_scaled)

        reg_metrics = evaluate_regression_model(
            y_test_time, y_pred_time,
            model_name='Time-to-Clog Regression',
            verbose=True
        )

    # ============================================================
    # Summary Report
    # ============================================================
    print_banner("EVALUATION SUMMARY")

    print("Classification Performance:")
    print(f"  Accuracy:  {class_metrics['accuracy']:.4f}")
    print(f"  Precision: {class_metrics['precision']:.4f}")
    print(f"  Recall:    {class_metrics['recall']:.4f}")
    print(f"  F1-Score:  {class_metrics['f1_score']:.4f}")

    if 'roc_auc' in class_metrics:
        print(f"  ROC-AUC:   {class_metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {class_metrics['pr_auc']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {class_metrics['TP']:4d}")
    print(f"  False Positives: {class_metrics['FP']:4d}")
    print(f"  True Negatives:  {class_metrics['TN']:4d}")
    print(f"  False Negatives: {class_metrics['FN']:4d}")

    print(f"\nOperational Cost:")
    print(f"  Total Cost: {class_metrics['operational_cost']:.0f}")
    print(f"  (Missed clogs × 100 + False alarms × 1)")

    print(f"\nModel Components:")
    print(f"  Classification models: {len(predictor.classification_models)}")
    print(f"  Survival models:       {len(predictor.survival_models)}")
    print(f"  Regression models:     {len(predictor.regression_models)}")
    print(f"  Anomaly detector:      {'✓' if predictor.anomaly_detector else '✗'}")

    print(f"\nResults saved to: {output_dir}/")
    print(f"  • Static plots:       {plots_dir}/")
    if interactive:
        print(f"  • Interactive plots:  {plots_dir}/ (HTML files)")

    print("\n" + "="*80)
    print("Pipeline complete! ✓")
    print("="*80 + "\n")

    return predictor, class_metrics


if __name__ == '__main__':
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description='Filter Clogging Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data.csv
  python main.py filter_data.xlsx --clog-index 8940
  python main.py data.csv --output results --no-interactive

For more information, see README.md
        """
    )

    parser.add_argument(
        'data_file',
        type=str,
        help='Path to data file (CSV or Excel format)'
    )

    parser.add_argument(
        '--clog-index',
        type=int,
        default=None,
        help='Index where clogging starts (auto-detected if not specified)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )

    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive visualizations'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom configuration file (JSON)'
    )

    args = parser.parse_args()

    # Load custom config if provided
    if args.config:
        import json
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            CONFIG.update(custom_config)
            print(f"Loaded custom configuration from: {args.config}")

    # Check if data file exists
    if not Path(args.data_file).exists():
        print(f"[ERROR] Data file not found: {args.data_file}")
        sys.exit(1)

    # Run main pipeline
    try:
        predictor, metrics = main(
            data_filepath=args.data_file,
            clog_index=args.clog_index,
            output_dir=args.output,
            interactive=not args.no_interactive
        )
    except KeyboardInterrupt:
        print("\n\n[!] Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
