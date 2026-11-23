import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from scipy.optimize import curve_fit

# File loading - replace 'filter_data.xlsx' with your actual file name
file_path = 'data.xlsx'
try:
    # Load the data
    data = pd.read_excel(file_path)
    print("Data loaded successfully.")
    print(f"Columns in the dataset: {data.columns.tolist()}")
    print(f"Number of data points: {len(data)}")
    print("\nFirst few rows of the data:")
    print(data.head())
except Exception as e:
    print(f"Error loading the file: {e}")
    print("Please ensure the file exists and contains 'time', 'flowrate', and 'dp' columns.")
    exit()

# Check for missing values
if data.isnull().sum().sum() > 0:
    print("\nWarning: Dataset contains missing values.")
    print(data.isnull().sum())
    # Fill missing values or drop rows as appropriate
    data = data.dropna()
    print(f"Dataset size after handling missing values: {len(data)}")

# Basic statistics of the data
print("\nBasic statistics of the data:")
print(data.describe())

# Exploratory Data Visualization
plt.figure(figsize=(18, 6))

# Plot 1: Pressure Drop vs Time
plt.subplot(131)
plt.scatter(data['time'], data['dp'], alpha=0.6)
plt.title('Pressure Drop vs Time')
plt.xlabel('Time (sec)')
plt.ylabel('Pressure Drop (psi)')
plt.grid(True)

# Plot 2: Pressure Drop vs Flow Rate
plt.subplot(132)
plt.scatter(data['flowrate'], data['dp'], alpha=0.6)
plt.title('Pressure Drop vs Flow Rate')
plt.xlabel('Flow Rate')
plt.ylabel('Pressure Drop (psi)')
plt.grid(True)

# Plot 3: Flow Rate vs Time
plt.subplot(133)
plt.scatter(data['time'], data['flowrate'], alpha=0.6)
plt.title('Flow Rate vs Time')
plt.xlabel('Time (sec)')
plt.ylabel('Flow Rate')
plt.grid(True)

plt.tight_layout()
plt.savefig('exploratory_plots.png')
plt.close()
print("\nExploratory plots generated and saved.")

# Calculate filter characteristics using Forchheimer equation: ΔP = (μ/k)·v + β·ρ·v²
# where:
# - ΔP is pressure drop (dp in our dataset)
# - μ is dynamic viscosity of air (1.81e-5 Pa·s at 20°C)
# - k is permeability (what we want to find)
# - v is superficial velocity (proportional to flowrate)
# - β is Forchheimer coefficient (what we want to find)
# - ρ is density of air (1.2 kg/m³ at 20°C)

# For simplicity, we'll assume the flowrate in our dataset is already in terms of superficial velocity
# If it's not, you'd need to convert it based on filter area

def forchheimer_equation(flowrate, permeability, forchheimer_coef):
    """
    Forchheimer equation for pressure drop through a porous medium.
    
    Parameters:
    - flowrate: superficial velocity (or proportional to it)
    - permeability: filter permeability
    - forchheimer_coef: Forchheimer coefficient
    
    Returns:
    - pressure drop (dp)
    """
    # Constants for air at standard conditions
    viscosity = 1.81e-5  # Pa·s
    density = 1.2  # kg/m³
    
    # Forchheimer equation
    dp = (viscosity/permeability) * flowrate + forchheimer_coef * density * flowrate**2
    return dp

# Filter data for steady-state conditions
# Assuming initial part of data might have transient effects, we use latter portion
steady_state_data = data[int(len(data) * 0.2):]  # Using 80% of data, skipping first 20%

# Fit Forchheimer equation to the steady-state data
try:
    # Initial parameter guesses
    p0 = [1e-10, 1e-3]  # Initial guesses for [permeability, forchheimer_coef]
    
    # Curve fitting
    params, params_covariance = curve_fit(
        forchheimer_equation, 
        steady_state_data['flowrate'], 
        steady_state_data['dp'],
        p0=p0,
        bounds=([1e-15, 1e-6], [1e-8, 1.0])  # Bounds for parameters
    )
    
    permeability = params[0]
    forchheimer_coef = params[1]
    
    print("\nFilter Characterization Results:")
    print(f"Permeability (k): {permeability:.4e} m²")
    print(f"Forchheimer coefficient (β): {forchheimer_coef:.4e}")
    
    # Calculate R² to assess fit quality
    dp_predicted = forchheimer_equation(steady_state_data['flowrate'], permeability, forchheimer_coef)
    r2 = r2_score(steady_state_data['dp'], dp_predicted)
    print(f"R² of Forchheimer equation fit: {r2:.4f}")
    
    # Plot the Forchheimer equation fit
    plt.figure(figsize=(8, 6))
    plt.scatter(steady_state_data['flowrate'], steady_state_data['dp'], alpha=0.6, label='Actual Data')
    
    # Generate points for the fitted curve
    flow_range = np.linspace(min(data['flowrate']), max(data['flowrate']), 100)
    dp_fit = forchheimer_equation(flow_range, permeability, forchheimer_coef)
    
    plt.plot(flow_range, dp_fit, 'r-', linewidth=2, label='Forchheimer Fit')
    plt.title('Forchheimer Equation Fit')
    plt.xlabel('Flow Rate')
    plt.ylabel('Pressure Drop (psi)')
    plt.legend()
    plt.grid(True)
    plt.savefig('forchheimer_fit.png')
    plt.close()
    print("Forchheimer fit plot generated and saved.")
    
except Exception as e:
    print(f"Error in Forchheimer equation fitting: {e}")
    print("Proceeding with XGBoost modeling only.")
    permeability = None
    forchheimer_coef = None

# XGBoost Model Training
print("\nPreparing data for XGBoost model training...")

# Features and target
X = data[['time', 'flowrate']]
y = data['dp']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

# Train a basic XGBoost model first
basic_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

basic_model.fit(X_train, y_train)

# Predict on test set
y_pred_basic = basic_model.predict(X_test)
mse_basic = mean_squared_error(y_test, y_pred_basic)
r2_basic = r2_score(y_test, y_pred_basic)

print("\nBasic XGBoost Model Performance:")
print(f"Mean Squared Error: {mse_basic:.4f}")
print(f"R² Score: {r2_basic:.4f}")

# Fine-tune the XGBoost model
print("\nFine-tuning XGBoost model...")
fine_tuned_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,  # Increased from 100
    max_depth=7,       # Increased from 3
    learning_rate=0.05,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42
)

fine_tuned_model.fit(X_train, y_train)

# Predict on test set
y_pred_fine_tuned = fine_tuned_model.predict(X_test)
mse_fine_tuned = mean_squared_error(y_test, y_pred_fine_tuned)
r2_fine_tuned = r2_score(y_test, y_pred_fine_tuned)

print("\nFine-tuned XGBoost Model Performance:")
print(f"Mean Squared Error: {mse_fine_tuned:.4f}")
print(f"R² Score: {r2_fine_tuned:.4f}")

# Evaluate model improvement
improvement_percentage = ((mse_basic - mse_fine_tuned) / mse_basic) * 100
print(f"\nModel Improvement: {improvement_percentage:.2f}% reduction in MSE")

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_fine_tuned, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Actual vs. Predicted Pressure Drop (Fine-tuned XGBoost)')
plt.xlabel('Actual Pressure Drop (psi)')
plt.ylabel('Predicted Pressure Drop (psi)')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.close()
print("Actual vs. predicted plot generated and saved.")

# Feature importance
plt.figure(figsize=(8, 4))
xgb.plot_importance(fine_tuned_model)
plt.title('Feature Importance in XGBoost Model')
plt.savefig('feature_importance.png')
plt.close()
print("Feature importance plot generated and saved.")

# Predict clogging time (when dp reaches 15 psi)
print("\nPredicting filter clogging time (when dp reaches 15 psi)...")

# Define flow rate ranges to analyze
min_flow = min(data['flowrate'])
max_flow = max(data['flowrate'])
flow_rates = np.linspace(min_flow, max_flow, 5)  # 5 different flow rates to analyze

# Maximum time to consider in prediction (adjust based on your data)
max_time_to_consider = max(data['time']) * 2  # Extending beyond the dataset time range

# Create a dataframe to store results
clogging_predictions = []

for flow_rate in flow_rates:
    # Create a time series from 0 to max_time_to_consider
    prediction_times = np.arange(0, max_time_to_consider, 10)  # 10-second intervals
    prediction_df = pd.DataFrame({
        'time': prediction_times,
        'flowrate': [flow_rate] * len(prediction_times)
    })
    
    # Predict pressure drop for each time point
    predicted_dp = fine_tuned_model.predict(prediction_df[['time', 'flowrate']])
    
    # Find the first time when dp >= 15 psi
    clogging_indices = np.where(predicted_dp >= 15)[0]
    
    if len(clogging_indices) > 0:
        clogging_time = prediction_times[clogging_indices[0]]
        clogging_predictions.append({
            'flow_rate': flow_rate,
            'clogging_time': clogging_time,
            'projected_dp': predicted_dp[clogging_indices[0]]
        })
    else:
        clogging_predictions.append({
            'flow_rate': flow_rate,
            'clogging_time': None,
            'projected_dp': max(predicted_dp)
        })

# Print clogging predictions
print("\nFilter Clogging Time Predictions:")
print("--------------------------------")
for pred in clogging_predictions:
    if pred['clogging_time'] is not None:
        print(f"At flow rate {pred['flow_rate']:.4f}: Clogging predicted at {pred['clogging_time']:.1f} seconds (dp = {pred['projected_dp']:.2f} psi)")
    else:
        print(f"At flow rate {pred['flow_rate']:.4f}: No clogging predicted within time range. Max dp = {pred['projected_dp']:.2f} psi")

# Plot clogging time vs. flow rate
clogging_flow_rates = [pred['flow_rate'] for pred in clogging_predictions if pred['clogging_time'] is not None]
clogging_times = [pred['clogging_time'] for pred in clogging_predictions if pred['clogging_time'] is not None]

if len(clogging_flow_rates) > 0:
    plt.figure(figsize=(8, 6))
    plt.scatter(clogging_flow_rates, clogging_times, marker='o', s=80)
    plt.title('Filter Clogging Time vs. Flow Rate')
    plt.xlabel('Flow Rate')
    plt.ylabel('Clogging Time (seconds)')
    plt.grid(True)
    plt.savefig('clogging_prediction.png')
    plt.close()
    print("Clogging prediction plot generated and saved.")

# Save the models
import pickle
with open('xgboost_basic_model.pkl', 'wb') as f:
    pickle.dump(basic_model, f)
with open('xgboost_fine_tuned_model.pkl', 'wb') as f:
    pickle.dump(fine_tuned_model, f)
print("\nModels saved as 'xgboost_basic_model.pkl' and 'xgboost_fine_tuned_model.pkl'")

# Comprehensive summary
print("\n========== COMPREHENSIVE ANALYSIS SUMMARY ==========")
print("Filter Characteristics:")
if permeability and forchheimer_coef:
    print(f"- Permeability (k): {permeability:.4e} m²")
    print(f"- Forchheimer coefficient (β): {forchheimer_coef:.4e}")
    print(f"- R² of Forchheimer equation fit: {r2:.4f}")
else:
    print("- Filter characteristics could not be determined from the data.")

print("\nXGBoost Model Performance:")
print(f"- Basic Model R² Score: {r2_basic:.4f}")
print(f"- Fine-tuned Model R² Score: {r2_fine_tuned:.4f}")
print(f"- Improvement: {improvement_percentage:.2f}% reduction in MSE")

print("\nFilter Clogging Predictions:")
for pred in clogging_predictions:
    if pred['clogging_time'] is not None:
        print(f"- At flow rate {pred['flow_rate']:.4f}: Clogging at {pred['clogging_time']:.1f} seconds")
    else:
        print(f"- At flow rate {pred['flow_rate']:.4f}: No clogging predicted within time range.")

print("\nConclusion:")
if permeability and forchheimer_coef:
    perm_range = "low" if permeability < 1e-12 else "moderate" if permeability < 1e-10 else "high"
    forch_range = "low" if forchheimer_coef < 1e-4 else "moderate" if forchheimer_coef < 1e-2 else "high"
    print(f"The filter exhibits {perm_range} permeability and {forch_range} inertial resistance (Forchheimer coefficient).")

avg_clogging_time = None
if len(clogging_times) > 0:
    avg_clogging_time = sum(clogging_times) / len(clogging_times)
    print(f"On average, filter clogging (15 psi) is predicted at {avg_clogging_time:.1f} seconds across tested flow rates.")
    
    # Additional insights based on flow rates
    if len(clogging_flow_rates) > 1 and len(clogging_times) > 1:
        if clogging_times[0] > clogging_times[-1]:
            print("Higher flow rates lead to faster clogging, suggesting particle deposition is flow-dependent.")
        else:
            print("Interestingly, lower flow rates lead to faster clogging, suggesting complex clogging mechanisms.")
else:
    print("No clogging was predicted within the analyzed time range for any flow rate.")

print("\nRecommendations:")
if avg_clogging_time:
    print(f"- Plan for filter replacement or cleaning before {avg_clogging_time:.1f} seconds of operation.")
    
    # Get optimal flow rate (one with longest clogging time)
    optimal_idx = clogging_times.index(max(clogging_times))
    optimal_flow = clogging_flow_rates[optimal_idx]
    print(f"- Consider operating at flow rate around {optimal_flow:.4f} to maximize filter life.")
else:
    print("- Current operating conditions appear to be below the clogging threshold.")
    print("- Consider extending the time range of analysis or using higher flow rates to determine clogging limits.")

print("\nSaved Files:")
print("- 'xgboost_basic_model.pkl': Basic XGBoost model")
print("- 'xgboost_fine_tuned_model.pkl': Fine-tuned XGBoost model")
print("- 'exploratory_plots.png': Initial data visualization")
if permeability and forchheimer_coef:
    print("- 'forchheimer_fit.png': Forchheimer equation fit visualization")
print("- 'actual_vs_predicted.png': Model prediction performance")
print("- 'feature_importance.png': Feature importance in XGBoost model")
if len(clogging_flow_rates) > 0:
    print("- 'clogging_prediction.png': Filter clogging time vs. flow rate")
print("================================================")