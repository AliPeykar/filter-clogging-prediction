"""
Quick diagnostic to understand the ROC-AUC < 0.5 issue.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

# Simulate what we saw in the output:
# Test set: 2267 positives, 25 negatives
# Model predictions: 2266 predicted positive, 26 predicted negative
# TP=2266, FP=25, FN=1, TN=0

n_test = 2292
y_test = np.array([1] * 2267 + [0] * 25)  # 2267 positives, 25 negatives

# Model predicts 2291 positives (based on 99.96%)
y_pred = np.array([1] * 2291 + [0] * 1)

# Create probabilities that would lead to these predictions
# Model is predicting almost everything as positive
y_proba = np.random.uniform(0.8, 0.99, size=n_test)  # High probabilities for positives

# Set 25 false positives to have high probability
# Set 1 false negative to have low probability
# Set 2266 true positives to have high probability
# Set 0 true negatives

# Actually, let's think about this differently
# If TP=2266, FP=25, FN=1, TN=0, that means:
# - 2267 actual positives: 2266 predicted as 1, 1 predicted as 0
# - 25 actual negatives: 25 predicted as 1, 0 predicted as 0

# This is weird - the model predicts everything as positive
# Let's check if the confusion matrix makes sense

print("Expected confusion matrix:")
print(f"TN=0, FP=25")
print(f"FN=1, TP=2266")
print()

# Let's create realistic probabilities
y_proba = np.zeros(n_test)
# For the 2267 true positives:
#   - 2266 of them should have prob > threshold (correctly predicted)
y_proba[:2266] = np.random.uniform(0.80, 0.99, 2266)
#   - 1 of them should have prob < threshold (false negative)
y_proba[2266] = 0.1
# For the 25 true negatives:
#   - All 25 should have prob > threshold (false positives)
y_proba[2267:] = np.random.uniform(0.80, 0.99, 25)

# Now calculate metrics
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print()

# Check confusion matrix
y_pred_from_proba = (y_proba > 0.795).astype(int)  # Using threshold from output
cm = confusion_matrix(y_test, y_pred_from_proba)
print("Confusion matrix from simulated probabilities:")
print(cm)
print()

# The issue: if the model gives high probabilities to EVERYTHING (both pos and neg),
# then ROC-AUC will be close to 0.5 or even < 0.5 because it can't discriminate!

# Let's test: what if model gives LOWER probs to negatives?
print("=== CORRECT MODEL (should have high ROC-AUC) ===")
y_proba_correct = np.zeros(n_test)
y_proba_correct[:2267] = np.random.uniform(0.7, 0.99, 2267)  # Positives get high prob
y_proba_correct[2267:] = np.random.uniform(0.01, 0.3, 25)    # Negatives get LOW prob

roc_auc_correct = roc_auc_score(y_test, y_proba_correct)
pr_auc_correct = average_precision_score(y_test, y_proba_correct)
print(f"ROC-AUC: {roc_auc_correct:.4f} (should be high)")
print(f"PR-AUC: {pr_auc_correct:.4f}")
print()

print("=== BROKEN MODEL (like ours - predicts all high) ===")
y_proba_broken = np.zeros(n_test)
y_proba_broken[:2267] = np.random.uniform(0.7, 0.99, 2267)   # Positives get high prob
y_proba_broken[2267:] = np.random.uniform(0.7, 0.99, 25)     # Negatives ALSO get high prob!

roc_auc_broken = roc_auc_score(y_test, y_proba_broken)
pr_auc_broken = average_precision_score(y_test, y_proba_broken)
print(f"ROC-AUC: {roc_auc_broken:.4f} (should be ~0.5 or worse)")
print(f"PR-AUC: {pr_auc_broken:.4f}")
print()

print("=== DIAGNOSIS ===")
print("If ROC-AUC < 0.5, it means the model is giving HIGH probabilities to negatives")
print("and possibly LOWER probabilities to positives (inverted learning).")
print()
print("OR it means the model cannot discriminate between classes at all.")
print("With only 25 negatives out of 2292 samples, small random variations")
print("in how the model scores those 25 samples can drastically affect ROC-AUC.")
print()
print("PR-AUC is more stable because it focuses on positive class performance,")
print("which is what we actually care about (detecting clogs).")
