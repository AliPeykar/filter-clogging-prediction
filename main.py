"""
Main script for filter clogging prediction system.
Demonstrates complete workflow from data loading to prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    plot_prediction_timeline,
    create_interpretability_dashboard,
    create_risk_level_dashboard
)


def main(data_filepath, clog_index=None):
    """
    Complete workflow for filter clogging prediction.

    Parameters:
    -----------
    data_filepath : str
        Path to CSV file with filter data
    clog_index : int, optional
        Index where clogging starts (auto-detected if None)
    """
    print(f"\n{'#'*80}")
    print(f"# FILTER CLOGGING PREDICTION SYSTEM")
    print(f"{'#'*80}\n")

    # ============================================================
    # 1. Load and Prepare Data
    # ============================================================
    print(f"\n[STEP 1/6] Loading and preparing data...")
    print(f"  Data file: {data_filepath}")

    df = load_and_prepare_data(data_filepath, config=CONFIG)

    # Create basic ratio features
    df = create_ratio_features(df)

    # Compute target labels
    df = compute_target_labels(df, config=CONFIG)

    print(f"  [OK] Data loaded: {len(df)} samples, {len(df.columns)} columns")

    # ============================================================
    # 2. Feature Engineering
    # ============================================================
    print(f"\n[STEP 2/6] Engineering features...")

    df = engineer_all_features(df, config=CONFIG)

    print(f"  [OK] Feature engineering complete: {len(df.columns)} total features")

    # ============================================================
    # 3. Prepare Features and Targets
    # ============================================================
    print(f"\n[STEP 3/6] Preparing features and targets...")

    X, y_class, y_time, y_duration, y_event, feature_names = prepare_features_and_targets(
        df, config=CONFIG
    )

    print(f"  [OK] Feature matrix: {X.shape}")
    print(f"  [OK] Positive samples: {y_class.sum()} ({100*y_class.mean():.2f}%)")

    # ============================================================
    # 4. Create Train/Val/Test Splits
    # ============================================================
    print(f"\n[STEP 4/6] Creating data splits...")

    train_idx, val_idx, test_idx, healthy_idx = time_series_split_imbalanced(
        df, clog_index=clog_index, config=CONFIG
    )

    # Split data
    X_train, y_train_class = X.iloc[train_idx], y_class[train_idx]
    X_val, y_val_class = X.iloc[val_idx], y_class[val_idx]
    X_test, y_test_class = X.iloc[test_idx], y_class[test_idx]
    X_healthy = X.iloc[healthy_idx]  # Healthy data for anomaly detection

    y_train_time = y_time[train_idx]
    y_train_duration = y_duration[train_idx]
    y_train_event = y_event[train_idx]

    y_test_time = y_time[test_idx]

    print(f"  [OK] Train: {len(train_idx)} samples")
    print(f"  [OK] Val:   {len(val_idx)} samples")
    print(f"  [OK] Test:  {len(test_idx)} samples")
    print(f"  [OK] Healthy: {len(healthy_idx)} samples")

    # ============================================================
    # 5. Train Predictor
    # ============================================================
    print(f"\n[STEP 5/6] Training prediction models...")

    predictor = FilterCloggingPredictor(config=CONFIG)

    predictor.fit(
        X_train=X_train,
        y_class=y_train_class,
        y_time=y_train_time,
        y_duration=y_train_duration,
        y_event=y_train_event,
        X_val=X_val,
        y_class_val=y_val_class,
        X_healthy=X_healthy,  # Pass healthy data, not indices
        verbose=True
    )

    # ============================================================
    # 6. Evaluate on Test Set
    # ============================================================
    print(f"\n[STEP 6/6] Evaluating on test set...")

    # Scale test features (convert to DataFrame to preserve feature names)
    scaler = predictor.scaler
    if isinstance(X_test, pd.DataFrame):
        X_test_scaled = scaler.transform(X_test.values)
    else:
        X_test_scaled = scaler.transform(X_test)

    # Classification predictions
    y_pred_class = predictor.predict(X_test_scaled, use_anomaly=True)
    y_pred_proba = predictor.predict_proba(X_test_scaled, use_anomaly=True)

    # Evaluate classification
    class_metrics = evaluate_classification_model(
        y_test_class, y_pred_class, y_pred_proba[:, 1],
        model_name='Filter Clogging Predictor',
        verbose=True
    )

    # Plot classification results
    plot_confusion_matrix(
        class_metrics['confusion_matrix'],
        model_name='Filter_Clogging_Predictor',
        save_path='plots'
    )

    if 'roc_auc' in class_metrics:
        plot_roc_curve(
            class_metrics['fpr'],
            class_metrics['tpr'],
            class_metrics['roc_auc'],
            model_name='Filter_Clogging_Predictor',
            save_path='plots'
        )

    # Regression predictions (if available)
    if len(predictor.regression_models) > 0:
        y_pred_time = predictor.predict_time_to_clog(X_test_scaled)

        reg_metrics = evaluate_regression_model(
            y_test_time, y_pred_time,
            model_name='Time-to-Clog Regression',
            verbose=True
        )

    # Comprehensive risk scores
    risk_scores = predictor.predict_risk_scores(X_test_scaled, use_all_models=True)

    # Plot timeline
    if 'ensemble' in risk_scores:
        plot_prediction_timeline(
            y_test_class, risk_scores['ensemble'],
            title='Ensemble Risk Score Over Time',
            save_path='plots'
        )

    # ============================================================
    # 6b. Risk Level Prediction (4-Level System)
    # ============================================================
    if CONFIG.get('risk_levels', {}).get('enabled', True):
        print(f"\n[STEP 6b/8] Generating 4-Level Risk Predictions...")

        try:
            # Get risk level predictions
            risk_level_results = predictor.predict_risk_level(
                X_test_scaled,
                use_anomaly=True,
                return_details=True
            )

            # Create comprehensive risk level dashboard
            risk_level_metrics = create_risk_level_dashboard(
                y_true_binary=y_test_class,
                risk_results=risk_level_results,
                save_path='plots/risk_levels',
                model_name='Filter_Clogging_Predictor'
            )

            print(f"  [OK] Risk level analysis complete!")
            print(f"  [OK] 4-level visualizations saved to: plots/risk_levels/")

        except Exception as e:
            print(f"  [!] Risk level analysis failed: {str(e)}")
            print(f"  [!] Continuing with pipeline...")
    else:
        print(f"\n[INFO] Risk level prediction disabled in config")

    # ============================================================
    # 7. Model Interpretability Analysis
    # ============================================================
    if CONFIG.get('interpretability', {}).get('enabled', True):
        print(f"\n[STEP 7/7] Generating Model Interpretability Analysis...")

        # Get a representative model for interpretability (use first classification model)
        if len(predictor.classification_models) > 0:
            # Get the first available model (unwrap from CalibratedClassifierCV if needed)
            model_name = list(predictor.classification_models.keys())[0]
            model = predictor.classification_models[model_name]

            # Unwrap calibrated model to get base estimator
            if hasattr(model, 'calibrated_classifiers_'):
                base_model = model.calibrated_classifiers_[0].estimator
            else:
                base_model = model

            # Get sample indices from config
            interp_config = CONFIG.get('interpretability', {})
            sample_indices = interp_config.get('shap', {}).get('sample_indices', [0, 1, 2])

            # Create interpretability plots subdirectory
            interp_save_path = 'plots/interpretability'

            try:
                # Create comprehensive interpretability dashboard
                interp_results = create_interpretability_dashboard(
                    model=base_model,
                    X_train=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
                    X_test=X_test_scaled,
                    y_test=y_test_class,
                    feature_names=feature_names,
                    model_name=f'{model_name.upper()}_Ensemble',
                    save_path=interp_save_path,
                    sample_indices=sample_indices
                )

                print(f"  [OK] Interpretability analysis complete!")
                print(f"  [OK] Visualizations saved to: {interp_save_path}/")

            except Exception as e:
                print(f"  [!] Interpretability analysis failed: {str(e)}")
                print(f"  [!] Continuing with pipeline...")
        else:
            print(f"  [!] No classification models available for interpretability analysis")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'#'*80}")
    print(f"# EVALUATION SUMMARY")
    print(f"{'#'*80}")
    print(f"\nBinary Classification Performance:")
    print(f"  Accuracy:  {class_metrics['accuracy']:.4f}")
    print(f"  Precision: {class_metrics['precision']:.4f}")
    print(f"  Recall:    {class_metrics['recall']:.4f}")
    print(f"  F1-Score:  {class_metrics['f1_score']:.4f}")

    if 'roc_auc' in class_metrics:
        print(f"  ROC-AUC:   {class_metrics['roc_auc']:.4f}")

    print(f"\nOperational Cost (Binary):")
    print(f"  Total Cost: {class_metrics['operational_cost']:.0f}")
    print(f"  (Missed clogs × 100 + False alarms × 1)")

    # Add risk level summary if available
    if CONFIG.get('risk_levels', {}).get('enabled', True) and 'risk_level_results' in locals():
        print(f"\n4-Level Risk System:")
        print(f"  Risk levels: LOW (0-25%), MODERATE (25-50%), HIGH (50-75%), CRITICAL (75-100%)")
        print(f"  Visualizations: plots/risk_levels/")

    print(f"\nAll plots saved to: plots/")
    print(f"{'#'*80}\n")

    return predictor, class_metrics


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py Comprehensive_Filter_Analysis.xlsx [clog_index]")
        print("\nExample:")
        print("  python main.py filter_data.csv 8940")
        sys.exit(1)

    data_filepath = sys.argv[1]
    clog_index = int(sys.argv[2]) if len(sys.argv) > 2 else None

    predictor, metrics = main(data_filepath, clog_index=clog_index)

    print("\n[OK] Pipeline complete!")
