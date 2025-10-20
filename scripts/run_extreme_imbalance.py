
from main import main
from config import CONFIG

# Disable classification and survival (need both classes)
# Keep anomaly detection and regression
CONFIG['models_to_use'] = []  # Disable traditional classification
CONFIG['use_survival'] = False  # Disable survival models
CONFIG['use_regression'] = True  # Keep regression
CONFIG['anomaly_detection']['enabled'] = True  # Keep anomaly detection

print("Running with configuration optimized for extreme imbalance...")
print("Using: Anomaly Detection + Regression only")
print("-" * 60)

predictor, metrics = main(
    data_filepath="Comprehensive_Filter_Analysis.xlsx",
    clog_index=8942
)

print("
[OK] Training complete!")
