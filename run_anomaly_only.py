"""
Simplified script for extreme imbalance cases.
Uses only anomaly detection since training set has no clogged samples.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG
from data_processing import load_and_prepare_data, compute_target_labels, create_ratio_features
from feature_engineering import engineer_all_features
from utils import time_series_split_imbalanced
from anomaly_detection import AnomalyDetectionModule

print("="*80)
print("FILTER CLOGGING PREDICTION - ANOMALY DETECTION MODE")
print("(For extreme imbalance where training set has no clogged samples)")
print("="*80)

# Load data
print("\n[1/4] Loading data...")
df = load_and_prepare_data('Comprehensive_Filter_Analysis.xlsx', config=CONFIG)
df = create_ratio_features(df)
df = compute_target_labels(df, config=CONFIG)

# Engineer features
print("\n[2/4] Engineering features...")
df = engineer_all_features(df, config=CONFIG)

# Prepare data
from data_processing import prepare_features_and_targets
X, y_class, y_time, y_duration, y_event, feature_names = prepare_features_and_targets(df, config=CONFIG)

# Split data
print("\n[3/4] Splitting data...")
train_idx, val_idx, test_idx, healthy_idx = time_series_split_imbalanced(df, clog_index=8942, config=CONFIG)

X_healthy = X.iloc[healthy_idx].values
X_test = X.iloc[test_idx].values
y_test = y_class[test_idx]

print(f"Healthy samples for training: {len(X_healthy)}")
print(f"Test samples: {len(X_test)}")
print(f"Test clogged ratio: {y_test.mean()*100:.2f}%")

# Train anomaly detector
print("\n[4/4] Training anomaly detector...")
detector = AnomalyDetectionModule(config=CONFIG)
detector.fit(X_healthy, verbose=True)

# Predict on test set
print("\nEvaluating on test set...")
results = detector.predict_anomaly_scores(X_test)

anomaly_scores = results['ensemble_score']
anomaly_labels = results['is_anomaly']

# Evaluate
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("\n" + "="*80)
print("RESULTS")
print("="*80)

# Convert anomaly scores to predictions (higher score = more likely to clog)
# Use threshold of 0.5
y_pred = (anomaly_scores > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Clogging']))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
print(f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")

# ROC-AUC
roc_auc = roc_auc_score(y_test, anomaly_scores)
print(f"\nROC-AUC: {roc_auc:.4f}")

# Operational cost
cost_fn = 100  # Cost of missing a clog
cost_fp = 1    # Cost of false alarm
total_cost = cost_fn * cm[1,0] + cost_fp * cm[0,1]
print(f"\nOperational Cost: {total_cost:.0f}")
print(f"  (Missed clogs × {cost_fn} + False alarms × {cost_fp})")

# Plot timeline
plt.figure(figsize=(14, 6))
plt.plot(anomaly_scores, label='Anomaly Score', alpha=0.7)
plt.plot(y_test, label='Actual Status (0=healthy, 1=clogged)', alpha=0.7)
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold=0.5')
plt.xlabel('Test Sample Index')
plt.ylabel('Value')
plt.title('Anomaly Detection Over Time')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/anomaly_detection_timeline.png', dpi=300, bbox_inches='tight')
print("\n[OK] Plot saved to: plots/anomaly_detection_timeline.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nKey Insights:")
print(f"  - Anomaly detector trained on {len(X_healthy)} healthy samples")
print(f"  - Detected {np.sum(anomaly_labels)} anomalies out of {len(X_test)} test samples")
print(f"  - ROC-AUC: {roc_auc:.4f}")
print(f"  - Operational Cost: {total_cost:.0f}")
print("\nRecommendation:")
if roc_auc > 0.75:
    print("  [OK] Good performance! Anomaly detection works well for this data.")
else:
    print("  [!] Consider collecting more diverse healthy operational data.")
