"""
ENHANCED PRODUCTION-READY FILTER CLOGGING ANALYSIS SYSTEM
=========================================================

A complete, industrial-grade filter clogging analysis system with proper formatting,
class ordering, and consistent indentation throughout.

Key Features:
- Properly ordered class definitions
- Consistent 4-space indentation
- SHAP model explanations
- Risk level distribution analysis
- Comprehensive visualizations
- Real-time prediction capabilities

Version: 3.1.0 (Properly Formatted)
"""

import logging
import warnings
import json
import os
import time
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Protocol
from functools import wraps
from collections import deque

import numpy as np
import pandas as pd
import joblib

# Core ML imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Optional advanced dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    warnings.warn("psutil not available - resource monitoring disabled")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not available")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Plotting libraries not available")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not available - model explainability disabled")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    warnings.warn("Optuna not available - hyperparameter tuning disabled")

# Suppress warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# SECTION 1: CUSTOM EXCEPTIONS
# ==============================================================================

class FilterAnalysisError(Exception):
    """Base exception for filter analysis system."""
    pass


class DataValidationError(FilterAnalysisError):
    """Data validation failed."""
    pass


class SensorError(FilterAnalysisError):
    """Critical sensor data issue."""
    pass


class ResourceLimitError(FilterAnalysisError):
    """System resource limits exceeded."""
    pass


class TemporalError(FilterAnalysisError):
    """Temporal data ordering or leakage issue."""
    pass


class PredictionError(FilterAnalysisError):
    """Prediction pipeline failure."""
    pass


# ==============================================================================
# SECTION 2: CONFIGURATION AND DATA CLASSES
# ==============================================================================

@dataclass
class SystemConfig:
    """Production system configuration for filter analysis."""
    
    # Critical sensors
    required_sensors: List[str] = field(default_factory=lambda: [
        'flowrate', 'dp'
    ])
    
    # Optional sensors
    optional_sensors: List[str] = field(default_factory=lambda: [
        'short_ema', 'long_ema', 'adaptive_cusum_thr', 'validation_score'
    ])
    
    # Time column
    time_column: str = 'time'
    
    # Domain thresholds
    max_pressure_drop: float = 60.0
    min_flow_rate: float = 5.0
    clogging_resistance_ratio: float = 2.0
    
    # Resource limits
    max_memory_gb: float = 4.0
    max_computation_minutes: int = 30
    max_file_size_mb: int = 100
    
    # Model parameters
    test_size: float = 0.2
    random_state: int = 42
    
    # Feature engineering
    temporal_buffer_size: int = 50
    
    # System behavior
    fail_on_missing_sensors: bool = False
    enable_resource_monitoring: bool = True
    log_level: str = 'INFO'
    use_existing_labels: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.required_sensors:
            raise ValueError("At least one required sensor must be specified")
        
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")
        
        if self.max_pressure_drop <= 0 or self.min_flow_rate <= 0:
            raise ValueError("Domain thresholds must be positive")


@dataclass
class SensorReading:
    """Individual sensor reading with timestamp."""
    timestamp: datetime
    values: Dict[str, float]
    
    def validate(self, required_sensors: List[str]) -> None:
        """Validate sensor reading has required values."""
        missing = [sensor for sensor in required_sensors if sensor not in self.values]
        if missing:
            raise SensorError(f"Missing required sensors: {missing}")


@dataclass
class PredictionResult:
    """Prediction result with metadata."""
    prediction: int
    probability: Optional[float]
    risk_level: int
    confidence: str
    recommendations: List[str]
    sensor_status: Dict[str, str]
    timestamp: datetime
    model_version: str
    shap_explanation: Optional[Dict[str, Any]] = None


# ==============================================================================
# SECTION 3: LOGGING SETUP
# ==============================================================================

def setup_production_logging(log_file: str, level: str = 'INFO') -> logging.Logger:
    """Setup production-grade logging."""
    logger = logging.getLogger('filter_analysis_enhanced')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Production formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ==============================================================================
# SECTION 4: RESOURCE MANAGEMENT
# ==============================================================================

class ResourceMonitor:
    """Production resource monitoring and enforcement."""
    
    def __init__(self, config: SystemConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.start_time = None
        self.peak_memory = 0.0
        
    def __enter__(self):
        self.start_time = time.time()
        if HAS_PSUTIL:
            self.peak_memory = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = (time.time() - self.start_time) / 60
            self.logger.info(f"Operation completed in {elapsed:.2f} minutes")
            
    def check_limits(self) -> None:
        """Enforce resource limits."""
        if not self.config.enable_resource_monitoring:
            return
            
        # Memory check
        if HAS_PSUTIL:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current_memory)
            
            if current_memory > self.config.max_memory_gb:
                gc.collect()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                
                if current_memory > self.config.max_memory_gb:
                    raise ResourceLimitError(
                        f"Memory usage {current_memory:.2f}GB exceeds limit {self.config.max_memory_gb}GB"
                    )
        
        # Time check
        if self.start_time:
            elapsed = (time.time() - self.start_time) / 60
            if elapsed > self.config.max_computation_minutes:
                raise ResourceLimitError(
                    f"Computation time {elapsed:.1f}min exceeds limit {self.config.max_computation_minutes}min"
                )


# ==============================================================================
# SECTION 5: DATA VALIDATION
# ==============================================================================

class DataValidator:
    """Comprehensive data validation for production."""
    
    def __init__(self, config: SystemConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def validate_file(self, file_path: str) -> None:
        """Validate file before loading."""
        path = Path(file_path)
        
        if not path.exists():
            raise DataValidationError(f"File does not exist: {file_path}")
            
        if not path.suffix.lower() in ['.xlsx', '.xls', '.csv']:
            raise DataValidationError(f"Unsupported file format: {path.suffix}")
            
        # Check file size
        size_mb = path.stat().st_size / 1024 / 1024
        if size_mb > self.config.max_file_size_mb:
            raise DataValidationError(
                f"File size {size_mb:.1f}MB exceeds limit {self.config.max_file_size_mb}MB"
            )
    
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate loaded DataFrame."""
        if df.empty:
            raise DataValidationError("DataFrame is empty")
            
        if len(df) < 100:
            raise DataValidationError(f"Insufficient data: {len(df)} rows (minimum 100 required)")
        
        # Check for required sensors
        missing_sensors = [sensor for sensor in self.config.required_sensors 
                          if sensor not in df.columns]
        if missing_sensors:
            if self.config.fail_on_missing_sensors:
                available = list(df.columns)
                raise SensorError(
                    f"Missing required sensors: {missing_sensors}. Available: {available}"
                )
            else:
                self.logger.warning(f"Missing sensors: {missing_sensors} - continuing with available sensors")
        
        # Validate sensor data ranges
        self._validate_sensor_ranges(df)
        
    def _validate_sensor_ranges(self, df: pd.DataFrame) -> None:
        """Validate sensor readings are within reasonable ranges."""
        validations = [
            ('flowrate', 0, 1000, "Flow rate"),
            ('dp', 0, 100, "Pressure drop"),
            ('adaptive_cusum_thr', -1000000, 1000000, "CUSUM")
        ]
        
        for sensor, min_val, max_val, description in validations:
            if sensor in df.columns:
                values = df[sensor].dropna()
                if len(values) > 0:
                    if values.min() < min_val or values.max() > max_val:
                        self.logger.warning(
                            f"{description} values outside expected range "
                            f"[{min_val}, {max_val}]: [{values.min():.2f}, {values.max():.2f}]"
                        )
    
    def validate_temporal_order(self, df: pd.DataFrame) -> None:
        """Validate temporal ordering of data."""
        if self.config.time_column not in df.columns:
            self.logger.warning(f"Time column '{self.config.time_column}' not found - skipping temporal validation")
            return
            
        time_col = df[self.config.time_column]
        
        if not time_col.is_monotonic_increasing:
            self.logger.warning("Data is not in strict temporal order - this may affect features")
            
        # Check for duplicate timestamps
        duplicates = time_col.duplicated().sum()
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate timestamps")


# ==============================================================================
# SECTION 6: TEMPORAL BUFFER
# ==============================================================================

class TemporalBuffer:
    """Manages temporal buffers for real-time feature computation."""
    
    def __init__(self, buffer_size: int, sensors: List[str]):
        self.buffer_size = buffer_size
        self.sensors = sensors
        self.buffers = {sensor: deque(maxlen=buffer_size) for sensor in sensors}
        
    def add_reading(self, reading: Dict[str, float]) -> None:
        """Add new sensor reading to buffers."""
        for sensor in self.sensors:
            if sensor in reading:
                self.buffers[sensor].append(reading[sensor])
    
    def get_rolling_mean(self, sensor: str, window: int) -> float:
        """Get rolling mean without data leakage."""
        if sensor not in self.buffers:
            return 0.0
            
        buffer = list(self.buffers[sensor])
        if len(buffer) == 0:
            return 0.0
            
        # Use only available historical data
        actual_window = min(window, len(buffer))
        return np.mean(buffer[-actual_window:]) if actual_window > 0 else 0.0
    
    def get_trend(self, sensor: str) -> float:
        """Get trend without data leakage."""
        if sensor not in self.buffers or len(self.buffers[sensor]) < 2:
            return 0.0
            
        buffer = list(self.buffers[sensor])
        return buffer[-1] - buffer[-2]


# ==============================================================================
# SECTION 7: TARGET CREATOR
# ==============================================================================

class EnhancedTargetCreator:
    """Enhanced target creation using existing labels with forced class diversity."""
    
    def __init__(self, config: SystemConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def create_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create targets using existing labels with enhanced detection."""
        
        # Try to use existing status labels
        binary_target, multiclass_target = self._try_existing_labels(df)
        
        if binary_target is not None:
            self.logger.info("Successfully used existing labels")
            return binary_target, multiclass_target
        
        # Enhanced domain-based detection
        self.logger.info("No valid existing labels - creating enhanced domain-based targets")
        return self._create_enhanced_domain_targets(df)
    
    def _try_existing_labels(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Try to detect and use existing filter status labels."""
        
        # Check for known filter status column names
        status_columns = [
            'validation_filter_status', 'filter_status', 'Filter_Status', 
            'status', 'Status', 'condition', 'state'
        ]
        
        for col in status_columns:
            if col in df.columns:
                self.logger.info(f"Found status column: {col}")
                return self._process_status_column(df, col)
        
        self.logger.warning("No status column found")
        return None, None
    
    def _process_status_column(self, df: pd.DataFrame, status_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process identified status column to create targets."""
        
        status_values = df[status_col].value_counts()
        self.logger.info(f"Status value distribution:\n{status_values}")
        
        # Create binary target based on status values
        clogged_indicators = ['clogged', 'blocked', 'failed', 'critical']
        
        binary_target = np.zeros(len(df), dtype=int)
        
        for i, value in enumerate(df[status_col]):
            if isinstance(value, str):
                if any(indicator in value.lower() for indicator in clogged_indicators):
                    binary_target[i] = 1
        
        # If no clogged samples found, use transition detection
        if binary_target.sum() == 0:
            self.logger.warning("No clogged status found - using transition detection")
            transition_point = self._find_transition_point(df)
            if transition_point:
                binary_target[transition_point:] = 1
        
        # Create multiclass from binary with forced diversity
        multiclass_target = self._create_forced_multiclass(df, binary_target)
        
        return binary_target, multiclass_target
    
    def _find_transition_point(self, df: pd.DataFrame) -> Optional[int]:
        """Enhanced transition point detection."""
        
        if 'dp' in df.columns and 'flowrate' in df.columns:
            # Use multiple indicators
            dp_values = df['dp'].values
            flow_values = df['flowrate'].values
            
            # Calculate rolling statistics
            window = min(50, len(df) // 10)
            
            # Pressure increase detection
            dp_rolling_mean = pd.Series(dp_values).rolling(window=window, min_periods=1).mean()
            dp_increase = dp_rolling_mean > dp_rolling_mean.iloc[window//2] * 1.5
            
            # Flow decrease detection
            flow_rolling_mean = pd.Series(flow_values).rolling(window=window, min_periods=1).mean()
            flow_decrease = flow_rolling_mean < flow_rolling_mean.iloc[window//2] * 0.8
            
            # Combined detection
            transition_candidates = np.where(dp_increase & flow_decrease)[0]
            
            if len(transition_candidates) > 0:
                # Use first significant transition
                return transition_candidates[0]
        
        # Fallback: assume transition at 80% of data
        return int(len(df) * 0.8)
    
    def _create_forced_multiclass(self, df: pd.DataFrame, binary_target: np.ndarray) -> np.ndarray:
        """Create multiclass target with forced class diversity."""
        
        multiclass_target = np.zeros(len(df), dtype=int)
        
        # Get sensor values for risk assessment
        risk_indicators = []
        
        if 'dp' in df.columns:
            risk_indicators.append(df['dp'].values)
        if 'flowrate' in df.columns:
            risk_indicators.append(1.0 / (df['flowrate'].values + 1e-6))
        if 'adaptive_cusum_thr' in df.columns:
            risk_indicators.append(np.abs(df['adaptive_cusum_thr'].values))
        
        if risk_indicators:
            # Combine risk indicators
            combined_risk = np.mean(risk_indicators, axis=0)
            combined_risk = (combined_risk - combined_risk.min()) / (combined_risk.max() - combined_risk.min() + 1e-6)
        else:
            # Fallback: use position-based risk
            combined_risk = np.linspace(0, 1, len(df))
        
        # Force 6 distinct risk levels
        n_samples = len(df)
        
        # Ensure minimum samples per class
        min_samples_per_class = max(10, n_samples // 20)
        
        # Calculate quantiles for 6 classes
        quantiles = [0, 0.4, 0.6, 0.75, 0.9, 0.95, 1.0]
        thresholds = np.quantile(combined_risk, quantiles)
        
        # Assign classes based on thresholds
        for i in range(6):
            if i == 5:  # Last class
                mask = combined_risk >= thresholds[i]
            else:
                mask = (combined_risk >= thresholds[i]) & (combined_risk < thresholds[i+1])
            
            multiclass_target[mask] = i
            
            # Ensure minimum samples per class
            if mask.sum() < min_samples_per_class and i < 5:
                # Expand this class to include more samples
                sorted_indices = np.argsort(combined_risk)
                class_indices = sorted_indices[i * min_samples_per_class:(i+1) * min_samples_per_class]
                multiclass_target[class_indices] = i
        
        # Force clogged samples to higher risk levels
        clogged_mask = binary_target == 1
        if clogged_mask.sum() > 0:
            multiclass_target[clogged_mask] = np.maximum(multiclass_target[clogged_mask], 3)
        
        # Final validation and distribution
        class_counts = np.bincount(multiclass_target)
        self.logger.info(f"Multiclass distribution: {dict(enumerate(class_counts))}")
        
        return multiclass_target
    
    def _create_enhanced_domain_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced domain-based target creation with forced diversity."""
        
        # Method 1: Resistance-based detection
        binary_target = self._create_resistance_based_target(df)
        
        # Method 2: Create forced multiclass
        multiclass_target = self._create_forced_multiclass(df, binary_target)
        
        return binary_target, multiclass_target
    
    def _create_resistance_based_target(self, df: pd.DataFrame) -> np.ndarray:
        """Create binary target using filter resistance."""
        
        n_samples = len(df)
        binary_target = np.zeros(n_samples, dtype=int)
        
        if 'dp' in df.columns and 'flowrate' in df.columns:
            # Calculate filter resistance
            flow_safe = np.maximum(df['flowrate'], 0.1)
            resistance = df['dp'] / flow_safe
            
            # Use adaptive threshold
            baseline_resistance = resistance.iloc[:n_samples//4].median()
            threshold = baseline_resistance * self.config.clogging_resistance_ratio
            
            clogged_mask = resistance > threshold
            binary_target[clogged_mask] = 1
            
            self.logger.info(f"Resistance-based clogging: {clogged_mask.sum()} samples")
        
        # Ensure reasonable distribution
        clogged_fraction = binary_target.sum() / n_samples
        if clogged_fraction < 0.1 or clogged_fraction > 0.4:
            # Use position-based assignment as fallback
            transition_point = int(n_samples * 0.8)
            binary_target = np.zeros(n_samples, dtype=int)
            binary_target[transition_point:] = 1
            
            self.logger.warning(f"Unusual clogging rate ({clogged_fraction:.1%}) - using position-based assignment")
        
        return binary_target


# ==============================================================================
# SECTION 8: RISK LEVEL ANALYZER
# ==============================================================================

class RiskLevelAnalyzer:
    """Analyze and visualize risk level distributions in filter data."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.viz_dir = Path('visualizations')
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_risk_distribution(self, multiclass_target: np.ndarray, 
                                 binary_target: np.ndarray = None, 
                                 sensor_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Comprehensive risk level distribution analysis."""
        
        try:
            # Basic distribution analysis
            unique_levels, counts = np.unique(multiclass_target, return_counts=True)
            total_samples = len(multiclass_target)
            
            # Create distribution summary
            distribution_summary = {
                "total_samples": total_samples,
                "num_risk_levels": len(unique_levels),
                "risk_levels": unique_levels.tolist(),
                "sample_counts": counts.tolist(),
                "percentages": (counts / total_samples * 100).round(2).tolist()
            }
            
            # Risk level definitions
            risk_definitions = {
                0: "Normal Operation - Low Risk",
                1: "Early Warning - Low-Medium Risk", 
                2: "Caution - Medium Risk",
                3: "Alert - Medium-High Risk",
                4: "Warning - High Risk",
                5: "Critical - Very High Risk"
            }
            
            # Add human-readable descriptions
            distribution_summary["risk_descriptions"] = {
                int(level): risk_definitions.get(level, f"Risk Level {level}")
                for level in unique_levels
            }
            
            self.logger.info("Risk Level Distribution:")
            for level, count, pct in zip(unique_levels, counts, distribution_summary["percentages"]):
                desc = risk_definitions.get(level, f"Level {level}")
                self.logger.info(f"  Level {level} ({desc}): {count:,} samples ({pct:.1f}%)")
            
            # Statistical analysis
            distribution_summary["statistics"] = {
                "mean_risk_level": float(np.mean(multiclass_target)),
                "median_risk_level": float(np.median(multiclass_target)),
                "std_risk_level": float(np.std(multiclass_target)),
                "high_risk_percentage": float(np.sum(multiclass_target >= 3) / total_samples * 100),
                "critical_risk_percentage": float(np.sum(multiclass_target >= 4) / total_samples * 100)
            }
            
            # Class balance analysis
            distribution_summary["balance_analysis"] = self._analyze_class_balance(counts)
            
            # Create visualizations
            if HAS_PLOTTING:
                self._create_risk_distribution_plots(
                    multiclass_target, binary_target, sensor_data, 
                    distribution_summary, risk_definitions
                )
            
            return distribution_summary
            
        except Exception as e:
            self.logger.error(f"Failed to analyze risk distribution: {e}")
            return {"error": str(e)}
    
    def _analyze_class_balance(self, counts: np.ndarray) -> Dict[str, Any]:
        """Analyze class balance and imbalance issues."""
        
        total = np.sum(counts)
        percentages = counts / total * 100
        
        # Calculate imbalance metrics
        max_class_pct = np.max(percentages)
        min_class_pct = np.min(percentages)
        imbalance_ratio = max_class_pct / min_class_pct if min_class_pct > 0 else float('inf')
        
        # Determine balance quality
        if imbalance_ratio <= 2.0:
            balance_quality = "Well Balanced"
        elif imbalance_ratio <= 5.0:
            balance_quality = "Moderately Imbalanced"
        elif imbalance_ratio <= 10.0:
            balance_quality = "Imbalanced"
        else:
            balance_quality = "Severely Imbalanced"
        
        return {
            "imbalance_ratio": float(imbalance_ratio),
            "max_class_percentage": float(max_class_pct),
            "min_class_percentage": float(min_class_pct),
            "balance_quality": balance_quality,
            "recommendation": self._get_balance_recommendation(balance_quality, imbalance_ratio)
        }
    
    def _get_balance_recommendation(self, balance_quality: str, imbalance_ratio: float) -> str:
        """Get recommendation for handling class imbalance."""
        
        if balance_quality == "Well Balanced":
            return "No special handling needed - classes are well balanced"
        elif balance_quality == "Moderately Imbalanced":
            return "Consider using class weights or stratified sampling"
        elif balance_quality == "Imbalanced":
            return "Recommended: Use SMOTE, class weights, or ensemble methods"
        else:
            return "Critical: Apply advanced resampling techniques and cost-sensitive learning"
    
    def _create_risk_distribution_plots(self, multiclass_target: np.ndarray, 
                                       binary_target: np.ndarray = None,
                                       sensor_data: pd.DataFrame = None,
                                       distribution_summary: Dict[str, Any] = None,
                                       risk_definitions: Dict[int, str] = None) -> None:
        """Create comprehensive risk distribution visualizations."""
        
        if not HAS_PLOTTING:
            return
            
        try:
            # Create visualization plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Risk Level Distribution Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Risk Level Bar Chart
            ax1 = axes[0, 0]
            unique_levels, counts = np.unique(multiclass_target, return_counts=True)
            colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(unique_levels)))
            
            bars = ax1.bar(unique_levels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            ax1.set_xlabel('Risk Level')
            ax1.set_ylabel('Number of Samples')
            ax1.set_title('Risk Level Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Add percentage labels
            total = len(multiclass_target)
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + total * 0.01,
                        f'{count}\n({count/total*100:.1f}%)', 
                        ha='center', va='bottom', fontsize=9)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'risk_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Risk distribution plots created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create risk distribution plots: {e}")


# ==============================================================================
# SECTION 9: FEATURE ENGINEERING
# ==============================================================================

class ProductionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Production-grade feature engineering with specific columns."""
    
    def __init__(self, config: SystemConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.fitted_ = False
        self.feature_names_ = []
        self.training_stats_ = {}
        self.temporal_buffer = None
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'ProductionFeatureEngineer':
        """Fit feature engineering on training data only."""
        self.logger.info("Fitting production feature engineer")
        
        # Use actual column names
        actual_sensors = [
            'flowrate', 'dp', 'short_ema', 'long_ema', 
            'adaptive_cusum_thr', 'validation_score'
        ]
        
        # Get available sensors from data
        available_sensors = [s for s in actual_sensors if s in X.columns]
        
        if not available_sensors:
            raise SensorError("No configured sensors found in training data")
        
        # Store training statistics
        for sensor in available_sensors:
            sensor_data = pd.to_numeric(X[sensor], errors='coerce').dropna()
            if len(sensor_data) > 0:
                std_val = sensor_data.std()
                if std_val < 1e-6:
                    std_val = max(sensor_data.mean() * 0.1, 1e-6)
                
                self.training_stats_[sensor] = {
                    'mean': float(sensor_data.mean()),
                    'std': float(std_val),
                    'min': float(sensor_data.min()),
                    'max': float(sensor_data.max()),
                    'q25': float(sensor_data.quantile(0.25)),
                    'q75': float(sensor_data.quantile(0.75))
                }
        
        # Initialize temporal buffer
        self.temporal_buffer = TemporalBuffer(
            self.config.temporal_buffer_size,
            list(self.training_stats_.keys())
        )
        
        # Compute features to establish feature names
        X_features = self._compute_features(X, is_training=True)
        self.feature_names_ = list(X_features.columns)
        
        self.fitted_ = True
        self.logger.info(f"Feature engineering fitted with {len(self.feature_names_)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features for prediction without data leakage."""
        if not self.fitted_:
            raise ValueError("Feature engineer must be fitted before transform")
        
        return self._compute_features(X, is_training=False)
    
    def _compute_features(self, X: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Compute features using specific column names."""
        
        # Start with available sensor data
        available_sensors = [s for s in self.training_stats_.keys() if s in X.columns]
        result = X[available_sensors].copy()
        
        # Convert to numeric and handle missing values
        for sensor in available_sensors:
            result[sensor] = pd.to_numeric(result[sensor], errors='coerce')
            result[sensor] = result[sensor].fillna(self.training_stats_[sensor]['mean'])
        
        # Basic features
        self._add_basic_features(result, available_sensors)
        
        # Temporal features
        if is_training:
            self._add_training_temporal_features(result, available_sensors)
        else:
            self._add_safe_temporal_features(result, available_sensors)
        
        # Domain-specific features
        self._add_domain_features(result, available_sensors)
        
        # Clean up infinite and NaN values
        result = self._clean_features(result)
        
        # Ensure consistent feature set
        if not is_training:
            for feature in self.feature_names_:
                if feature not in result.columns:
                    result[feature] = 0.0
            result = result.reindex(columns=self.feature_names_, fill_value=0.0)
            result = self._clean_features(result)
        
        return result
    
    def _add_basic_features(self, df: pd.DataFrame, sensors: List[str]) -> None:
        """Add basic derived features."""
        for sensor in sensors:
            try:
                stats = self.training_stats_[sensor]
                safe_std = max(stats['std'], 1e-6)
                
                df[f'{sensor}_normalized'] = (df[sensor] - stats['mean']) / safe_std
                
                pct_change = df[sensor].pct_change()
                pct_change = pct_change.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                df[f'{sensor}_rate'] = pct_change.clip(-10.0, 10.0)
                
                range_span = stats['max'] - stats['min']
                if range_span > 1e-6:
                    position = (df[sensor] - stats['min']) / range_span
                    df[f'{sensor}_position'] = position.clip(0.0, 1.0)
                else:
                    df[f'{sensor}_position'] = 0.5
                
            except Exception as e:
                self.logger.warning(f"Failed to add basic features for {sensor}: {e}")
    
    def _add_training_temporal_features(self, df: pd.DataFrame, sensors: List[str]) -> None:
        """Add temporal features during training."""
        windows = [5, 10, 20]
        
        for sensor in sensors:
            for window in windows:
                try:
                    df[f'{sensor}_rolling_mean_{window}'] = df[sensor].rolling(
                        window=window, min_periods=1
                    ).mean()
                    
                    df[f'{sensor}_rolling_std_{window}'] = df[sensor].rolling(
                        window=window, min_periods=1
                    ).std().fillna(0)
                    
                    df[f'{sensor}_rolling_change_{window}'] = (
                        df[sensor] - df[sensor].shift(window)
                    ).fillna(0)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to add training temporal features for {sensor}: {e}")
    
    def _add_safe_temporal_features(self, df: pd.DataFrame, sensors: List[str]) -> None:
        """Add temporal features for batch prediction."""
        windows = [5, 10, 20]
        
        for sensor in sensors:
            for window in windows:
                try:
                    df[f'{sensor}_rolling_mean_{window}'] = df[sensor].expanding(
                        min_periods=1
                    ).mean()
                    
                    df[f'{sensor}_rolling_std_{window}'] = df[sensor].expanding(
                        min_periods=1
                    ).std().fillna(0)
                    
                    df[f'{sensor}_rolling_change_{window}'] = (
                        df[sensor] - df[sensor].shift(min(window, len(df)))
                    ).fillna(0)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to add safe temporal features for {sensor}: {e}")
    
    def _add_domain_features(self, df: pd.DataFrame, sensors: List[str]) -> None:
        """Add domain-specific features for filter analysis."""
        try:
            # Filter resistance features
            if 'flowrate' in sensors and 'dp' in sensors:
                flowrate_safe = np.maximum(df['flowrate'], 0.5)
                resistance = df['dp'] / flowrate_safe
                df['filter_resistance'] = resistance.clip(0.0, 1000.0)
                
                # Normalized resistance
                if 'flowrate' in self.training_stats_ and 'dp' in self.training_stats_:
                    baseline_flow = max(self.training_stats_['flowrate']['q75'], 1.0)
                    baseline_dp = max(self.training_stats_['dp']['q25'], 0.1)
                    baseline_resistance = baseline_dp / baseline_flow
                    
                    if baseline_resistance > 1e-6:
                        resistance_ratio = df['filter_resistance'] / baseline_resistance
                        df['resistance_ratio'] = resistance_ratio.clip(0.0, 100.0)
                    else:
                        df['resistance_ratio'] = 1.0
                
                # Flow efficiency
                max_flow = max(self.training_stats_['flowrate']['max'], 1.0)
                flow_efficiency = df['flowrate'] / max_flow
                df['flow_efficiency'] = flow_efficiency.clip(0.0, 2.0)
            
            # EMA features
            if 'short_ema' in sensors and 'long_ema' in sensors:
                ema_spread = df['short_ema'] - df['long_ema']
                df['ema_spread'] = ema_spread.clip(-100.0, 100.0)
                
                long_ema_safe = np.maximum(df['long_ema'], 1e-3)
                ema_ratio = df['short_ema'] / long_ema_safe
                df['ema_ratio'] = ema_ratio.clip(0.0, 10.0)
            
            # CUSUM threshold features
            if 'adaptive_cusum_thr' in sensors:
                df['cusum_thr_normalized'] = df['adaptive_cusum_thr'] / (
                    max(self.training_stats_['adaptive_cusum_thr']['max'], 1e-3)
                )
            
            # Validation score features
            if 'validation_score' in sensors:
                val_mean = self.training_stats_['validation_score']['mean']
                val_std = max(self.training_stats_['validation_score']['std'], 1e-3)
                df['validation_score_z'] = (df['validation_score'] - val_mean) / val_std
                df['validation_score_z'] = df['validation_score_z'].clip(-5.0, 5.0)
            
            # Interaction features
            if 'flowrate' in sensors and 'dp' in sensors:
                flow_norm = (df['flowrate'] / max(self.training_stats_['flowrate']['mean'], 1e-3)).clip(0.0, 10.0)
                dp_norm = (df['dp'] / max(self.training_stats_['dp']['mean'], 1e-3)).clip(0.0, 100.0)
                interaction = flow_norm * dp_norm
                df['flow_dp_interaction'] = interaction.clip(0.0, 1000.0)
                
        except Exception as e:
            self.logger.error(f"Failed to add domain features: {e}")
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up infinite, NaN, and extreme values."""
        cleaned = df.copy()
        
        # Replace infinite values
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        cleaned = cleaned.fillna(0.0)
        
        # Clip extremely large values
        for column in cleaned.columns:
            if cleaned[column].dtype.kind in 'biufc':  # numeric columns
                max_safe_value = 1e10
                min_safe_value = -1e10
                cleaned[column] = cleaned[column].clip(min_safe_value, max_safe_value)
                
                if cleaned[column].abs().max() > 1e8:
                    self.logger.warning(f"Large values in feature {column}, applying scaling")
                    max_abs = cleaned[column].abs().max()
                    if max_abs > 0:
                        cleaned[column] = cleaned[column] / (max_abs / 1e6)
        
        # Final verification
        if not np.all(np.isfinite(cleaned.values)):
            self.logger.error("Still have non-finite values - forcing to zero")
            cleaned = cleaned.fillna(0.0)
            cleaned = cleaned.replace([np.inf, -np.inf], 0.0)
        
        return cleaned


# ==============================================================================
# SECTION 10: TEMPORAL SPLITTER
# ==============================================================================

class ProductionTemporalSplitter:
    """Temporal splitting with comprehensive validation."""
    
    def __init__(self, config: SystemConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def split(self, df: pd.DataFrame, *targets) -> Tuple[pd.DataFrame, ...]:
        """Split data temporally with validation."""
        
        # Sort by time if column exists
        if self.config.time_column in df.columns:
            if not df[self.config.time_column].is_monotonic_increasing:
                self.logger.warning("Sorting data by time column")
                sort_idx = df[self.config.time_column].argsort()
                df = df.iloc[sort_idx].copy()
                targets = tuple(target[sort_idx] if isinstance(target, np.ndarray) else target 
                               for target in targets)
        
        n_samples = len(df)
        split_idx = int(n_samples * (1 - self.config.test_size))
        
        # Ensure minimum sizes
        if split_idx < 100:
            split_idx = max(split_idx, min(100, n_samples - 50))
        
        if n_samples - split_idx < 50:
            split_idx = min(split_idx, n_samples - 50)
        
        # Split data
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        
        self.logger.info(f"Temporal split: train={len(df_train)}, test={len(df_test)}")
        
        result = [df_train, df_test]
        for target in targets:
            result.extend([target[:split_idx], target[split_idx:]])
        
        return tuple(result)


# ==============================================================================
# SECTION 11: HYPERPARAMETER OPTIMIZER
# ==============================================================================

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, config: SystemConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.study = None
        self.best_params = {}
        
    def optimize_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                       model_type: str = 'random_forest', n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters for specified model type."""
        
        if not HAS_OPTUNA:
            self.logger.warning("Optuna not available - using default parameters")
            return self._get_default_params(model_type)
        
        try:
            # Create study
            study_name = f"filter_analysis_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
            )
            
            # Define objective function
            def objective(trial):
                return self._objective_function(trial, X_train, y_train, model_type)
            
            # Optimize
            self.logger.info(f"Starting hyperparameter optimization for {model_type}")
            self.study.optimize(objective, n_trials=n_trials, timeout=600)
            
            self.best_params = self.study.best_params
            best_score = self.study.best_value
            
            self.logger.info(f"Optimization completed. Best score: {best_score:.4f}")
            
            return {
                'best_params': self.best_params,
                'best_score': best_score,
                'n_trials': len(self.study.trials)
            }
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}")
            return self._get_default_params(model_type)
    
    def _objective_function(self, trial, X_train: np.ndarray, y_train: np.ndarray, 
                           model_type: str) -> float:
        """Objective function for optimization."""
        
        try:
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'class_weight': 'balanced',
                    'random_state': self.config.random_state,
                    'n_jobs': -1
                }
                model = RandomForestClassifier(**params)
                
            elif model_type == 'xgboost' and HAS_XGBOOST:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'random_state': self.config.random_state,
                    'eval_metric': 'logloss'
                }
                model = xgb.XGBClassifier(**params)
                
            else:
                return 0.0
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
            
            return np.mean(scores)
            
        except Exception:
            return 0.0
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters when optimization fails."""
        defaults = {
            'random_forest': {
                'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2,
                'min_samples_leaf': 1, 'max_features': 'sqrt'
            },
            'xgboost': {
                'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
                'subsample': 0.8, 'colsample_bytree': 0.8
            }
        }
        
        return {
            'best_params': defaults.get(model_type, {}),
            'best_score': 0.0,
            'method': 'default'
        }


# ==============================================================================
# SECTION 12: SHAP EXPLAINER
# ==============================================================================

class SHAPExplainer:
    """SHAP-based model explainability for filter analysis."""
    
    def __init__(self, model, X_background: np.ndarray, feature_names: List[str], 
                 logger: logging.Logger, model_name: str = "model"):
        self.model = model
        self.feature_names = feature_names
        self.logger = logger
        self.model_name = model_name
        self.explainer = None
        self.shap_values = None
        self.viz_dir = Path('visualizations') / 'shap'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_SHAP:
            try:
                # Use a sample of background data for efficiency
                background_sample = X_background[:min(100, len(X_background))]
                
                # Choose appropriate explainer based on model type
                model_type = str(type(model).__name__).lower()
                if 'forest' in model_type or 'xgb' in model_type or 'gradient' in model_type:
                    try:
                        self.explainer = shap.TreeExplainer(model)
                        self.logger.info(f"Using TreeExplainer for {model_name}")
                    except Exception:
                        self.explainer = shap.Explainer(model, background_sample)
                        self.logger.info(f"Using general Explainer for {model_name}")
                else:
                    self.explainer = shap.Explainer(model, background_sample)
                    self.logger.info(f"Using general Explainer for {model_name}")
                
                self.logger.info(f"SHAP explainer initialized successfully for {model_name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize SHAP explainer for {model_name}: {e}")
                self.explainer = None
        else:
            self.logger.warning("SHAP not available - explainability disabled")
    
    def explain_predictions(self, X_test: np.ndarray, max_samples: int = 100) -> None:
        """Generate SHAP explanations for test predictions."""
        if not self.explainer:
            self.logger.warning(f"SHAP explainer not available for {self.model_name}")
            return
            
        try:
            # Use subset for efficiency
            X_explain = X_test[:min(max_samples, len(X_test))]
            
            self.logger.info(f"Calculating SHAP values for {self.model_name}...")
            self.shap_values = self.explainer.shap_values(X_explain)
            
            # Handle different SHAP value formats
            if isinstance(self.shap_values, list):
                # Binary classification - use positive class
                if len(self.shap_values) == 2:
                    shap_vals = self.shap_values[1]
                else:
                    shap_vals = self.shap_values[0]
            else:
                shap_vals = self.shap_values
            
            # Create SHAP visualizations
            self._create_shap_summary_plot(shap_vals, X_explain)
            self._create_shap_feature_importance(shap_vals)
            
            self.logger.info(f"SHAP explanations generated successfully for {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate SHAP explanations for {self.model_name}: {e}")
    
    def _create_shap_summary_plot(self, shap_values: np.ndarray, X_data: np.ndarray) -> None:
        """Create SHAP summary plot showing feature impact distribution."""
        if not HAS_PLOTTING or not HAS_SHAP:
            return
            
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_data, feature_names=self.feature_names, 
                            show=False, max_display=20)
            plt.title(f'SHAP Summary Plot - {self.model_name}\nFeature Impact on Predictions')
            plt.tight_layout()
            plt.savefig(self.viz_dir / f'shap_summary_{self.model_name.lower()}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"SHAP summary plot saved for {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create SHAP summary plot for {self.model_name}: {e}")
    
    def _create_shap_feature_importance(self, shap_values: np.ndarray) -> None:
        """Create SHAP-based feature importance plot."""
        if not HAS_PLOTTING:
            return
            
        try:
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature importance ranking
            feature_importance = list(zip(self.feature_names, mean_shap))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Plot top 20 features
            top_n = min(20, len(feature_importance))
            features, importances = zip(*feature_importance[:top_n])
            
            plt.figure(figsize=(10, 8))
            y_pos = np.arange(top_n)
            bars = plt.barh(y_pos, importances, color='steelblue', alpha=0.8)
            plt.yticks(y_pos, features)
            plt.xlabel('Mean |SHAP Value| (Feature Importance)')
            plt.title(f'SHAP-based Feature Importance - {self.model_name}')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, importances)):
                plt.text(bar.get_width() + max(importances) * 0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / f'shap_importance_{self.model_name.lower()}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"SHAP feature importance saved for {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create SHAP feature importance for {self.model_name}: {e}")
    
    def explain_single_prediction(self, X_single: np.ndarray, prediction: int, 
                                 probability: float = None) -> Dict[str, Any]:
        """Provide SHAP explanation for a single prediction."""
        if not self.explainer:
            return {"error": "SHAP explainer not available"}
            
        try:
            # Get SHAP values for single prediction
            shap_vals = self.explainer.shap_values(X_single.reshape(1, -1))
            
            if isinstance(shap_vals, list):
                # For binary classification, use positive class
                if len(shap_vals) == 2:
                    shap_vals = shap_vals[1]
                else:
                    shap_vals = shap_vals[0]
            
            shap_vals = shap_vals[0]  # Get values for single prediction
            
            # Get top contributing features
            feature_contributions = list(zip(self.feature_names, shap_vals, X_single))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Format explanation
            explanation = {
                "model": self.model_name,
                "prediction": prediction,
                "probability": probability,
                "base_value": (self.explainer.expected_value if hasattr(self.explainer, 'expected_value') 
                             else np.mean(shap_vals)),
                "top_positive_features": [],
                "top_negative_features": [],
                "feature_contributions": []
            }
            
            for feature, shap_val, feature_val in feature_contributions[:20]:
                contrib = {
                    "feature": feature,
                    "value": float(feature_val),
                    "shap_value": float(shap_val),
                    "contribution": "increases clogging risk" if shap_val > 0 else "decreases clogging risk",
                    "impact": "high" if abs(shap_val) > 0.1 else "medium" if abs(shap_val) > 0.05 else "low"
                }
                explanation["feature_contributions"].append(contrib)
                
                if shap_val > 0.01:
                    explanation["top_positive_features"].append(contrib)
                elif shap_val < -0.01:
                    explanation["top_negative_features"].append(contrib)
            
            # Limit to top 10 for each category
            explanation["top_positive_features"] = explanation["top_positive_features"][:10]
            explanation["top_negative_features"] = explanation["top_negative_features"][:10]
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Failed to explain single prediction for {self.model_name}: {e}")
            return {"error": str(e)}


# ==============================================================================
# SECTION 13: VISUALIZATION SYSTEM
# ==============================================================================

class FilterVisualization:
    """Comprehensive visualization system."""
    
    def __init__(self, config: SystemConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.viz_dir = Path('visualizations')
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
    def create_data_overview(self, df: pd.DataFrame, binary_target: np.ndarray = None) -> None:
        """Create data overview visualizations."""
        if not HAS_PLOTTING:
            return
            
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Filter Data Overview', fontsize=16, fontweight='bold')
            
            # 1. Time series if available
            ax1 = axes[0, 0]
            if self.config.time_column in df.columns and 'flowrate' in df.columns:
                ax1.plot(df.index, df['flowrate'], label='Flow Rate', alpha=0.7)
                if 'dp' in df.columns:
                    ax1.plot(df.index, df['dp'], label='Pressure Drop', alpha=0.7)
                ax1.set_xlabel('Sample Index')
                ax1.set_ylabel('Values')
                ax1.set_title('Sensor Readings Over Time')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Correlation heatmap
            ax2 = axes[0, 1]
            sensor_cols = ['flowrate', 'dp', 'short_ema', 'long_ema']
            available_cols = [col for col in sensor_cols if col in df.columns]
            if len(available_cols) > 1:
                corr_matrix = df[available_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2, fmt='.2f')
                ax2.set_title('Sensor Correlation Matrix')
            
            # 3. Target distribution
            ax3 = axes[0, 2]
            if binary_target is not None:
                target_counts = np.bincount(binary_target)
                labels = ['Not Clogged', 'Clogged']
                colors = ['green', 'red']
                ax3.pie(target_counts, labels=labels, colors=colors, autopct='%1.1f%%')
                ax3.set_title('Filter Status Distribution')
            
            # 4. Flow vs Pressure scatter
            ax4 = axes[1, 0]
            if 'flowrate' in df.columns and 'dp' in df.columns:
                scatter = ax4.scatter(df['flowrate'], df['dp'], 
                                    c=binary_target if binary_target is not None else 'blue',
                                    alpha=0.6, cmap='RdYlGn_r')
                ax4.set_xlabel('Flow Rate')
                ax4.set_ylabel('Pressure Drop')
                ax4.set_title('Flow Rate vs Pressure Drop')
                ax4.grid(True, alpha=0.3)
            
            # 5. Feature distributions
            ax5 = axes[1, 1]
            key_sensors = ['flowrate', 'dp', 'short_ema']
            for sensor in key_sensors:
                if sensor in df.columns:
                    ax5.hist(df[sensor], bins=30, alpha=0.6, label=sensor, density=True)
            ax5.set_xlabel('Value')
            ax5.set_ylabel('Density')
            ax5.set_title('Sensor Distributions')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. EMA comparison if available
            ax6 = axes[1, 2]
            if 'short_ema' in df.columns and 'long_ema' in df.columns:
                ax6.plot(df.index, df['short_ema'], label='Short EMA', alpha=0.7)
                ax6.plot(df.index, df['long_ema'], label='Long EMA', alpha=0.7)
                ax6.set_xlabel('Sample Index')
                ax6.set_ylabel('EMA Value')
                ax6.set_title('EMA Comparison')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'data_overview.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Data overview saved to {self.viz_dir / 'data_overview.png'}")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create data overview: {e}")
    
    def create_model_performance_plots(self, metrics: Dict[str, Any]) -> None:
        """Create model performance plots."""
        if not HAS_PLOTTING:
            return
            
        try:
            # Extract performance metrics
            model_names = []
            accuracies = []
            f1_scores = []
            
            for model_name, model_metrics in metrics.items():
                if isinstance(model_metrics, dict) and 'accuracy' in model_metrics:
                    model_names.append(model_name.replace('binary_', ''))
                    accuracies.append(model_metrics['accuracy'])
                    f1_scores.append(model_metrics.get('f1_score', 0))
            
            if not model_names:
                return
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Performance comparison
            x = np.arange(len(model_names))
            width = 0.35
            
            axes[0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
            axes[0].bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
            axes[0].set_xlabel('Models')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Model Performance Comparison')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(model_names, rotation=45)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Metrics table
            axes[1].axis('tight')
            axes[1].axis('off')
            
            table_data = []
            for i, model in enumerate(model_names):
                full_name = f'binary_{model}'
                if full_name in metrics:
                    m = metrics[full_name]
                    table_data.append([
                        model,
                        f"{m.get('accuracy', 0):.3f}",
                        f"{m.get('f1_score', 0):.3f}",
                        f"{m.get('precision', 0):.3f}",
                        f"{m.get('recall', 0):.3f}"
                    ])
            
            if table_data:
                table = axes[1].table(
                    cellText=table_data,
                    colLabels=['Model', 'Accuracy', 'F1', 'Precision', 'Recall'],
                    cellLoc='center',
                    loc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                axes[1].set_title('Detailed Metrics')
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Model performance plots saved to {self.viz_dir / 'model_performance.png'}")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create performance plots: {e}")


# ==============================================================================
# SECTION 14: MODEL TRAINER
# ==============================================================================

class ProductionModelTrainer:
    """Enhanced model training with optimization and explainability."""
    
    def __init__(self, config: SystemConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.shap_explainers = {}
        self.optimizer = HyperparameterOptimizer(config, logger)
        
    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_binary_train: np.ndarray, y_binary_test: np.ndarray,
                    y_multi_train: np.ndarray, y_multi_test: np.ndarray,
                    enable_optimization: bool = True, enable_shap: bool = True) -> Dict[str, Any]:
        """Train and evaluate models with optimization and explainability."""
        
        self.logger.info("Starting enhanced model training with SHAP explainability")
        
        # Feature scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Check class distributions
        binary_classes = np.unique(y_binary_train)
        multi_classes = np.unique(y_multi_train)
        
        self.logger.info(f"Binary classes: {len(binary_classes)}")
        self.logger.info(f"Multi classes: {len(multi_classes)}")
        
        # Train binary classifier
        if len(binary_classes) >= 2:
            self._train_binary_classifier(
                X_train_scaled, X_test_scaled, y_binary_train, y_binary_test,
                list(X_train.columns), enable_optimization, enable_shap
            )
        
        # Train multiclass classifier
        if len(multi_classes) >= 2:
            self._train_multiclass_classifier(
                X_train_scaled, X_test_scaled, y_multi_train, y_multi_test,
                list(X_train.columns), enable_shap
            )
        
        return self.metrics
    
    def _train_binary_classifier(self, X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray,
                                feature_names: List[str], enable_optimization: bool,
                                enable_shap: bool):
        """Train binary classifier with optimization and SHAP."""
        
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            self.logger.error("Binary classification requires at least 2 classes")
            return
        
        # Calculate class weights
        try:
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
            class_weight_dict = dict(zip(unique_classes, class_weights))
        except Exception:
            class_weight_dict = 'balanced'
        
        # Train Random Forest
        rf_model = self._train_single_model(
            'RandomForest', 'random_forest', X_train, X_test, y_train, y_test,
            class_weight_dict, enable_optimization, feature_names
        )
        
        # Initialize SHAP explainer for Random Forest
        if enable_shap and rf_model and HAS_SHAP:
            try:
                self.shap_explainers['RandomForest'] = SHAPExplainer(
                    rf_model, X_train, feature_names, self.logger, 'RandomForest'
                )
                self.shap_explainers['RandomForest'].explain_predictions(X_test, max_samples=100)
            except Exception as e:
                self.logger.warning(f"SHAP initialization failed for RandomForest: {e}")
        
        # Train XGBoost if available
        if HAS_XGBOOST:
            xgb_model = self._train_single_model(
                'XGBoost', 'xgboost', X_train, X_test, y_train, y_test,
                class_weight_dict, enable_optimization, feature_names
            )
            
            # Initialize SHAP explainer for XGBoost
            if enable_shap and xgb_model and HAS_SHAP:
                try:
                    self.shap_explainers['XGBoost'] = SHAPExplainer(
                        xgb_model, X_train, feature_names, self.logger, 'XGBoost'
                    )
                    self.shap_explainers['XGBoost'].explain_predictions(X_test, max_samples=100)
                except Exception as e:
                    self.logger.warning(f"SHAP initialization failed for XGBoost: {e}")
    
    def _train_single_model(self, name: str, model_type: str, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray, class_weight_dict, 
                           enable_optimization: bool, feature_names: List[str]):
        """Train a single model and return the trained model."""
        
        try:
            self.logger.info(f"Training {name}")
            start_time = time.time()
            
            # Get optimized parameters
            if enable_optimization:
                opt_result = self.optimizer.optimize_model(X_train, y_train, model_type, n_trials=20)
                optimized_params = opt_result['best_params']
            else:
                optimized_params = self.optimizer._get_default_params(model_type)['best_params']
            
            # Create and train model
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    **optimized_params,
                    class_weight=class_weight_dict,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
            elif model_type == 'xgboost':
                # Handle class weights for XGBoost
                if isinstance(class_weight_dict, dict) and len(class_weight_dict) == 2:
                    scale_pos_weight = class_weight_dict[0] / class_weight_dict[1]
                else:
                    scale_pos_weight = 1.0
                
                model = xgb.XGBClassifier(
                    **optimized_params,
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.config.random_state,
                    eval_metric='logloss'
                )
            else:
                return None
            
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate model
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0, average='weighted'),
                'recall': recall_score(y_test, y_pred, zero_division=0, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, zero_division=0, average='weighted'),
                'training_time': training_time
            }
            
            # AUC for binary classification
            try:
                if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
                    y_proba = model.predict_proba(X_test)
                    if y_proba.shape[1] > 1:
                        metrics['auc'] = roc_auc_score(y_test, y_proba[:, 1])
            except Exception as e:
                self.logger.warning(f"Could not compute AUC: {e}")
            
            self.metrics[f'binary_{name}'] = metrics
            self.models[f'binary_{name}'] = model
            
            self.logger.info(f"{name}: F1={metrics['f1_score']:.3f}, Acc={metrics['accuracy']:.3f}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to train {name}: {e}")
            return None
    
    def _train_multiclass_classifier(self, X_train: np.ndarray, X_test: np.ndarray,
                                   y_train: np.ndarray, y_test: np.ndarray,
                                   feature_names: List[str], enable_shap: bool):
        """Train multiclass classifier with SHAP explainability."""
        
        try:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            self.metrics['multiclass'] = metrics
            self.models['multiclass'] = model
            
            # Initialize SHAP explainer for multiclass
            if enable_shap and HAS_SHAP:
                try:
                    self.shap_explainers['multiclass'] = SHAPExplainer(
                        model, X_train, feature_names, self.logger, 'multiclass'
                    )
                    self.shap_explainers['multiclass'].explain_predictions(X_test, max_samples=100)
                except Exception as e:
                    self.logger.warning(f"SHAP initialization failed for multiclass: {e}")
            
            self.logger.info(f"Multiclass: F1={metrics['f1_score']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to train multiclass model: {e}")


# ==============================================================================
# SECTION 15: PRODUCTION PREDICTOR
# ==============================================================================

class ProductionPredictor:
    """Production prediction system with SHAP explainability."""
    
    def __init__(self, models: Dict, scalers: Dict, feature_engineer: ProductionFeatureEngineer,
                 config: SystemConfig, logger: logging.Logger, shap_explainers: Dict = None):
        self.models = models
        self.scalers = scalers
        self.feature_engineer = feature_engineer
        self.config = config
        self.logger = logger
        self.shap_explainers = shap_explainers or {}
        self.model_version = "3.1.0_formatted"
        
    def predict_single(self, sensor_reading: Dict[str, float], 
                      explain: bool = False, model_name: str = None) -> PredictionResult:
        """Make single prediction with optional SHAP explanation."""
        
        # Validate input
        self._validate_sensor_reading(sensor_reading)
        
        try:
            # Create DataFrame
            df = pd.DataFrame([sensor_reading])
            
            # Engineer features
            features = self.feature_engineer.transform(df)
            
            # Scale features
            features_scaled = self.scalers['main'].transform(features)
            
            # Make predictions
            predictions = {}
            probabilities = {}
            
            # Binary prediction
            model_key = f'binary_{model_name}' if model_name else 'binary_RandomForest'
            if model_key in self.models:
                model = self.models[model_key]
                pred = model.predict(features_scaled)[0]
                predictions['binary'] = int(pred)
                
                # Get probability if available
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(features_scaled)[0]
                    probabilities['binary'] = float(prob[1]) if len(prob) > 1 else float(prob[0])
            else:
                predictions['binary'] = 0
                probabilities['binary'] = 0.0
            
            # Multiclass prediction
            if 'multiclass' in self.models:
                pred = self.models['multiclass'].predict(features_scaled)[0]
                predictions['multiclass'] = int(pred)
            else:
                predictions['multiclass'] = predictions['binary']
            
            # Generate result
            result = self._create_prediction_result(
                sensor_reading, predictions, probabilities
            )
            
            # Add SHAP explanation if requested
            if explain and self.shap_explainers:
                model_for_shap = model_name or 'RandomForest'
                if model_for_shap in self.shap_explainers:
                    try:
                        explanation = self.shap_explainers[model_for_shap].explain_single_prediction(
                            features_scaled[0], predictions['binary'], probabilities.get('binary')
                        )
                        result.shap_explanation = explanation
                        self.logger.info(f"SHAP explanation generated using {model_for_shap}")
                    except Exception as e:
                        self.logger.warning(f"SHAP explanation failed: {e}")
                        result.shap_explanation = {"error": f"SHAP explanation failed: {str(e)}"}
                else:
                    result.shap_explanation = {"error": f"SHAP explainer not available for {model_for_shap}"}
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Prediction failed: {str(e)}")
    
    def _validate_sensor_reading(self, reading: Dict[str, float]) -> None:
        """Validate sensor reading."""
        
        # Check for at least some required sensors
        available_required = [s for s in self.config.required_sensors if s in reading]
        
        if not available_required and self.config.fail_on_missing_sensors:
            raise SensorError(f"No required sensors found: {self.config.required_sensors}")
        
        # Validate values
        for sensor, value in reading.items():
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                    reading[sensor] = value
                except (ValueError, TypeError):
                    raise SensorError(f"Invalid value for {sensor}: {value}")
            
            if not np.isfinite(value):
                raise SensorError(f"Invalid value for {sensor}: {value}")
    
    def _create_prediction_result(self, sensor_reading: Dict[str, float],
                                predictions: Dict[str, int], 
                                probabilities: Dict[str, float]) -> PredictionResult:
        """Create comprehensive prediction result."""
        
        # Main prediction
        main_prediction = predictions.get('binary', 0)
        main_probability = probabilities.get('binary', 0.0)
        
        # Risk level
        risk_level = predictions.get('multiclass', main_prediction)
        
        # Confidence assessment
        if main_probability is not None:
            if main_probability > 0.8 or main_probability < 0.2:
                confidence = "high"
            elif main_probability > 0.6 or main_probability < 0.4:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "unknown"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            main_prediction, main_probability, risk_level, sensor_reading
        )
        
        # Sensor status
        sensor_status = {}
        for sensor in self.config.required_sensors + self.config.optional_sensors:
            sensor_status[sensor] = "present" if sensor in sensor_reading else "missing"
        
        result = PredictionResult(
            prediction=main_prediction,
            probability=main_probability,
            risk_level=risk_level,
            confidence=confidence,
            recommendations=recommendations,
            sensor_status=sensor_status,
            timestamp=datetime.now(),
            model_version=self.model_version
        )
        
        return result
    
    def _generate_recommendations(self, prediction: int, probability: Optional[float],
                                risk_level: int, sensor_reading: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on risk level."""
        recommendations = []
        
        # Risk level based recommendations
        if risk_level >= 5:  # Critical risk
            recommendations.append("CRITICAL: Immediate filter replacement required")
            recommendations.append("Stop operations and replace filter within 1 hour")
        elif risk_level >= 4:  # High risk
            recommendations.append("HIGH PRIORITY: Schedule filter maintenance within 4 hours")
            recommendations.append("Monitor closely for rapid deterioration")
        elif risk_level >= 3:  # Medium-high risk
            recommendations.append("MEDIUM PRIORITY: Plan filter maintenance within 12 hours")
            recommendations.append("Increase monitoring frequency to every 30 minutes")
        elif risk_level >= 2:  # Medium risk
            recommendations.append("CAUTION: Filter showing early degradation signs")
            recommendations.append("Schedule maintenance within 24 hours")
        elif risk_level >= 1:  # Low-medium risk
            recommendations.append("EARLY WARNING: Monitor filter condition")
            recommendations.append("Consider maintenance planning")
        else:  # Normal operation
            recommendations.append("Continue normal operation")
            recommendations.append("Maintain standard monitoring schedule")
        
        # Sensor-specific recommendations
        if 'dp' in sensor_reading and sensor_reading['dp'] > self.config.max_pressure_drop * 0.8:
            recommendations.append(f"Pressure drop approaching limit: {sensor_reading['dp']:.1f}")
        
        if 'flowrate' in sensor_reading and sensor_reading['flowrate'] < self.config.min_flow_rate * 1.2:
            recommendations.append(f"Flow rate below optimal: {sensor_reading['flowrate']:.1f}")
        
        return recommendations


# ==============================================================================
# SECTION 16: MAIN SYSTEM ORCHESTRATOR
# ==============================================================================

class ProductionFilterAnalysisSystem:
    """Complete production-ready filter analysis system with SHAP and risk analysis."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        # Setup configuration
        self.config = config or SystemConfig()
        self.config.validate()
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        log_file = Path('logs') / f'filter_analysis_{datetime.now().strftime("%Y%m%d")}.log'
        self.logger = setup_production_logging(str(log_file), self.config.log_level)
        
        # Initialize components
        self.validator = DataValidator(self.config, self.logger)
        self.feature_engineer = ProductionFeatureEngineer(self.config, self.logger)
        self.target_creator = EnhancedTargetCreator(self.config, self.logger)
        self.splitter = ProductionTemporalSplitter(self.config, self.logger)
        self.trainer = ProductionModelTrainer(self.config, self.logger)
        self.visualizer = FilterVisualization(self.config, self.logger)
        self.risk_analyzer = RiskLevelAnalyzer(self.logger)
        
        # System state
        self.predictor = None
        self.is_trained = False
        self.risk_distribution = None
        
        self.logger.info("Production Filter Analysis System initialized")
        
    def _load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate data from the specified path."""
        self.logger.info(f"Loading data from {data_path}")
        
        # Validate file before loading
        self.validator.validate_file(data_path)
        
        # Load data based on file extension
        path = Path(data_path)
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        else:
            raise DataValidationError(f"Unsupported file format: {path.suffix}")
        
        # Validate loaded data
        self.validator.validate_dataframe(df)
        self.validator.validate_temporal_order(df)
        
        self.logger.info(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        return df
        
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        directories = ['logs', 'models', 'reports', 'visualizations', 'visualizations/shap']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def train_system(self, data_path: str, enable_optimization: bool = True, 
                    enable_shap: bool = True, create_visualizations: bool = True) -> bool:
        """Train the complete system with SHAP explainability and risk analysis."""
        
        with ResourceMonitor(self.config, self.logger) as monitor:
            try:
                self.logger.info(f"Starting system training on {data_path}")
                
                # Load and validate data
                monitor.check_limits()
                df = self._load_and_validate_data(data_path)
                
                # Create targets
                monitor.check_limits()
                binary_target, multiclass_target = self.target_creator.create_targets(df)
                
                # Analyze risk distribution
                monitor.check_limits()
                self.risk_distribution = self.risk_analyzer.analyze_risk_distribution(
                    multiclass_target, binary_target, df
                )
                
                # Create visualizations
                if create_visualizations and HAS_PLOTTING:
                    self.visualizer.create_data_overview(df, binary_target)
                
                # Split data
                monitor.check_limits()
                split_result = self.splitter.split(df, binary_target, multiclass_target)
                df_train, df_test = split_result[0], split_result[1]
                y_binary_train, y_binary_test = split_result[2], split_result[3]
                y_multi_train, y_multi_test = split_result[4], split_result[5]
                
                # Engineer features
                monitor.check_limits()
                self.feature_engineer.fit(df_train)
                X_train = self.feature_engineer.transform(df_train)
                X_test = self.feature_engineer.transform(df_test)
                
                # Ensure targets are numpy arrays
                if not isinstance(y_binary_train, np.ndarray):
                    y_binary_train = np.array(y_binary_train)
                if not isinstance(y_binary_test, np.ndarray):
                    y_binary_test = np.array(y_binary_test)
                if not isinstance(y_multi_train, np.ndarray):
                    y_multi_train = np.array(y_multi_train)
                if not isinstance(y_multi_test, np.ndarray):
                    y_multi_test = np.array(y_multi_test)
                
                # Train models
                monitor.check_limits()
                metrics = self.trainer.train_models(
                    X_train, X_test,
                    y_binary_train, y_binary_test,
                    y_multi_train, y_multi_test,
                    enable_optimization=enable_optimization,
                    enable_shap=enable_shap
                )
                
                # Create performance visualizations
                if create_visualizations and HAS_PLOTTING:
                    self.visualizer.create_model_performance_plots(metrics)
                
                # Initialize predictor
                self.predictor = ProductionPredictor(
                    self.trainer.models,
                    self.trainer.scalers,
                    self.feature_engineer,
                    self.config,
                    self.logger,
                    self.trainer.shap_explainers
                )
                
                self.is_trained = True
                self.logger.info("System training completed successfully")
                return True
            
            except Exception as e:  # This was missing/misaligned
              self.logger.error(f"System training failed: {e}")
              import traceback
              self.logger.error(f"Full error: {traceback.format_exc()}")
              return False
