"""
FILTER ANALYSIS SYSTEM WITH GENETIC ALGORITHM OPTIMIZER
========================================================

A complete filter clogging analysis system with Genetic Algorithm
hyperparameter optimization for superior model performance.

Features:
- Genetic Algorithm for hyperparameter optimization
- Automatic feature engineering
- Class imbalance handling
- SHAP-like feature importance
- Comprehensive reporting

Version: 2.0.0
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('FilterAnalysis')


# ==============================================================================
# GENETIC ALGORITHM OPTIMIZER
# ==============================================================================

class GeneticAlgorithmOptimizer:
    """
    Advanced Genetic Algorithm for hyperparameter optimization.
    
    This optimizer uses evolutionary principles to find optimal hyperparameters:
    - Population-based search
    - Crossover and mutation operations
    - Elite preservation
    - Tournament selection
    """
    
    def __init__(self, 
                 population_size: int = 20,
                 n_generations: int = 15,
                 mutation_rate: float = 0.2,
                 elite_size: int = 4,
                 tournament_size: int = 3,
                 verbose: bool = True):
        """
        Initialize the Genetic Algorithm optimizer.
        
        Args:
            population_size: Number of individuals in each generation
            n_generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each gene
            elite_size: Number of best individuals to preserve
            tournament_size: Number of individuals in tournament selection
            verbose: Whether to print progress
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.verbose = verbose
        
        self.best_params = None
        self.best_score = 0
        self.optimization_history = []
        
        # Define comprehensive hyperparameter search space
        self.param_space = {
            'n_estimators': (50, 300),
            'max_depth': (3, 20),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'criterion': ['gini', 'entropy'],
            'bootstrap': [True, False],
            'min_impurity_decrease': (0.0, 0.01),
            'max_samples': [None, 0.5, 0.7, 0.9]
        }
        
        logger.info(f"GA Optimizer initialized: pop_size={population_size}, generations={n_generations}")
    
    def create_individual(self) -> Dict[str, Any]:
        """Create a random individual (set of hyperparameters)."""
        individual = {}
        for param, values in self.param_space.items():
            if isinstance(values, tuple):
                # Continuous parameter - use random value in range
                if isinstance(values[0], float):
                    individual[param] = random.uniform(values[0], values[1])
                else:
                    individual[param] = random.randint(values[0], values[1])
            elif isinstance(values, list):
                # Categorical parameter - choose randomly
                individual[param] = random.choice(values)
        return individual
    
    def create_population(self) -> List[Dict[str, Any]]:
        """Create initial random population."""
        population = []
        for _ in range(self.population_size):
            population.append(self.create_individual())
        return population
    
    def evaluate_fitness(self, individual: Dict[str, Any], 
                        X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Evaluate fitness of an individual using cross-validation.
        
        Returns F1 score as fitness metric (good for imbalanced data).
        """
        try:
            # Create model with individual's parameters
            params = individual.copy()
            
            # Fix the bootstrap/max_samples conflict
            # If bootstrap is False, we cannot use max_samples
            if 'bootstrap' in params and not params['bootstrap']:
                params.pop('max_samples', None)  # Remove max_samples if bootstrap is False
            elif 'max_samples' in params and params['max_samples'] is None:
                params.pop('max_samples', None)  # Remove if None
            
            # Remove min_impurity_decrease for now (not critical)
            params.pop('min_impurity_decrease', None)
            
            # Handle None values for max_features
            if params.get('max_features') is None:
                params.pop('max_features', None)
                
            # Ensure integer parameters
            for int_param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                if int_param in params and params[int_param] is not None:
                    params[int_param] = int(params[int_param])
            
            model = RandomForestClassifier(
                **params,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            # Use stratified k-fold for robust evaluation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, 
                                   cv=cv, scoring='f1_weighted', n_jobs=-1)
            
            fitness = np.mean(scores)
            return fitness
            
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return 0.0
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform uniform crossover between two parents.
        
        Each gene has 50% chance to come from either parent.
        """
        child = {}
        for param in self.param_space.keys():
            if random.random() < 0.5:
                child[param] = parent1.get(param, self.create_individual()[param])
            else:
                child[param] = parent2.get(param, self.create_individual()[param])
        return child
    
    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply mutation to an individual.
        
        Each gene has mutation_rate probability of being mutated.
        """
        mutated = individual.copy()
        
        for param, values in self.param_space.items():
            if random.random() < self.mutation_rate:
                if isinstance(values, tuple):
                    # Mutate continuous parameter
                    current_val = mutated[param]
                    if isinstance(current_val, (int, float)):
                        # Apply Gaussian noise
                        range_size = values[1] - values[0]
                        noise = random.gauss(0, 0.2 * range_size)
                        
                        if isinstance(values[0], int):
                            new_val = int(current_val + noise)
                            mutated[param] = max(values[0], min(values[1], new_val))
                        else:
                            new_val = current_val + noise
                            mutated[param] = max(values[0], min(values[1], new_val))
                    else:
                        # If type mismatch, create new random value
                        if isinstance(values[0], float):
                            mutated[param] = random.uniform(values[0], values[1])
                        else:
                            mutated[param] = random.randint(values[0], values[1])
                            
                elif isinstance(values, list):
                    # Mutate categorical parameter
                    mutated[param] = random.choice(values)
        
        return mutated
    
    def tournament_selection(self, population: List[Dict[str, Any]], 
                           fitness_scores: List[float]) -> Dict[str, Any]:
        """
        Select parent using tournament selection.
        
        Randomly select tournament_size individuals and return the best.
        """
        tournament_indices = random.sample(range(len(population)), 
                                         min(self.tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index].copy()
    
    def evolve_population(self, population: List[Dict[str, Any]], 
                         fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """
        Evolve population to next generation.
        
        Applies elitism, crossover, and mutation.
        """
        new_population = []
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        
        # Keep elite individuals
        for i in range(min(self.elite_size, len(sorted_population))):
            new_population.append(sorted_population[i].copy())
        
        # Generate rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.tournament_selection(population, fitness_scores)
            parent2 = self.tournament_selection(population, fitness_scores)
            
            # Crossover
            child = self.crossover(parent1, parent2)
            
            # Mutation
            child = self.mutate(child)
            
            new_population.append(child)
        
        return new_population
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """
        Run genetic algorithm optimization.
        
        Returns:
            Tuple of (best_parameters, best_score)
        """
        logger.info("="*60)
        logger.info("STARTING GENETIC ALGORITHM OPTIMIZATION")
        logger.info("="*60)
        logger.info(f"Configuration: {self.population_size} individuals, {self.n_generations} generations")
        
        # Create initial population
        population = self.create_population()
        
        # Evolution loop
        for generation in range(self.n_generations):
            generation_start = time.time()
            
            # Evaluate fitness for all individuals
            fitness_scores = []
            for i, individual in enumerate(population):
                if self.verbose and i % 5 == 0:
                    logger.debug(f"Generation {generation+1}: Evaluating individual {i+1}/{self.population_size}")
                
                fitness = self.evaluate_fitness(individual, X_train, y_train)
                fitness_scores.append(fitness)
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            generation_best_score = fitness_scores[best_idx]
            generation_best_params = population[best_idx].copy()
            
            # Update global best
            if generation_best_score > self.best_score:
                self.best_score = generation_best_score
                self.best_params = generation_best_params.copy()
                logger.info(f"🎯 New best score: {self.best_score:.4f}")
            
            # Calculate statistics
            avg_fitness = np.mean(fitness_scores)
            std_fitness = np.std(fitness_scores)
            min_fitness = np.min(fitness_scores)
            
            # Log generation results
            generation_time = time.time() - generation_start
            logger.info(
                f"Generation {generation+1:2d}/{self.n_generations} | "
                f"Best: {generation_best_score:.4f} | "
                f"Avg: {avg_fitness:.4f} | "
                f"Std: {std_fitness:.4f} | "
                f"Min: {min_fitness:.4f} | "
                f"Time: {generation_time:.1f}s"
            )
            
            # Store history
            self.optimization_history.append({
                'generation': generation + 1,
                'best_fitness': generation_best_score,
                'avg_fitness': avg_fitness,
                'std_fitness': std_fitness,
                'min_fitness': min_fitness,
                'best_params': generation_best_params.copy(),
                'time': generation_time
            })
            
            # Check for convergence
            if generation >= 5:
                recent_improvements = [
                    self.optimization_history[i]['best_fitness'] 
                    for i in range(-5, 0)
                ]
                if np.std(recent_improvements) < 0.001:
                    logger.info("Converged early - stopping optimization")
                    break
            
            # Evolve to next generation
            if generation < self.n_generations - 1:
                population = self.evolve_population(population, fitness_scores)
        
        logger.info("="*60)
        logger.info(f"✅ OPTIMIZATION COMPLETED")
        logger.info(f"Best F1 Score: {self.best_score:.4f}")
        logger.info("Best parameters found:")
        for param, value in self.best_params.items():
            logger.info(f"  - {param}: {value}")
        logger.info("="*60)
        
        return self.best_params, self.best_score
    
    def get_optimization_report(self) -> str:
        """Generate detailed optimization report."""
        if not self.optimization_history:
            return "No optimization history available"
        
        report_lines = []
        report_lines.append("\n" + "="*70)
        report_lines.append("GENETIC ALGORITHM OPTIMIZATION REPORT")
        report_lines.append("="*70)
        
        # Configuration
        report_lines.append(f"\nConfiguration:")
        report_lines.append(f"  Population Size: {self.population_size}")
        report_lines.append(f"  Generations: {self.n_generations}")
        report_lines.append(f"  Mutation Rate: {self.mutation_rate:.2%}")
        report_lines.append(f"  Elite Size: {self.elite_size}")
        report_lines.append(f"  Tournament Size: {self.tournament_size}")
        
        # Results
        report_lines.append(f"\nResults:")
        report_lines.append(f"  Final Best Score: {self.best_score:.4f}")
        report_lines.append(f"  Total Evaluations: {len(self.optimization_history) * self.population_size}")
        
        total_time = sum(gen['time'] for gen in self.optimization_history)
        report_lines.append(f"  Total Time: {total_time:.1f} seconds")
        
        # Best Parameters
        report_lines.append(f"\nBest Hyperparameters:")
        for param, value in sorted(self.best_params.items()):
            report_lines.append(f"  {param:20s}: {value}")
        
        # Evolution Progress
        report_lines.append(f"\nEvolution Progress:")
        report_lines.append(f"{'Gen':>4s} | {'Best':>7s} | {'Average':>7s} | {'Std Dev':>7s} | {'Time(s)':>7s}")
        report_lines.append("-" * 45)
        
        for gen_data in self.optimization_history[::max(1, len(self.optimization_history)//10)]:
            report_lines.append(
                f"{gen_data['generation']:4d} | "
                f"{gen_data['best_fitness']:7.4f} | "
                f"{gen_data['avg_fitness']:7.4f} | "
                f"{gen_data['std_fitness']:7.4f} | "
                f"{gen_data['time']:7.1f}"
            )
        
        # Convergence Analysis
        if len(self.optimization_history) > 1:
            improvements = []
            for i in range(1, len(self.optimization_history)):
                prev_best = self.optimization_history[i-1]['best_fitness']
                curr_best = self.optimization_history[i]['best_fitness']
                improvements.append(max(0, curr_best - prev_best))
            
            report_lines.append(f"\nConvergence Analysis:")
            report_lines.append(f"  Average Improvement per Generation: {np.mean(improvements):.4f}")
            report_lines.append(f"  Generations with Improvement: {sum(1 for x in improvements if x > 0)}")
            report_lines.append(f"  Final Plateau Length: {len(improvements) - max(i for i, x in enumerate(improvements) if x > 0.001) if any(x > 0.001 for x in improvements) else 0}")
        
        report_lines.append("="*70)
        
        return "\n".join(report_lines)


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

class FeatureEngineer:
    """
    Comprehensive feature engineering for filter analysis.
    
    Creates domain-specific features optimized for filter clogging detection.
    """
    
    def __init__(self):
        self.feature_names = []
        self.fitted = False
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features."""
        features = self.transform(df)
        self.fitted = True
        return features
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set."""
        features = pd.DataFrame()
        
        # Basic sensor features
        sensor_cols = ['flowrate', 'dp', 'short_ema', 'long_ema', 'adaptive_cusum_thr']
        available_sensors = [col for col in sensor_cols if col in df.columns]
        
        # Add raw sensor values
        for col in available_sensors:
            features[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Domain-specific features
        if 'flowrate' in features.columns and 'dp' in features.columns:
            # Filter resistance (key indicator)
            features['resistance'] = features['dp'] / (features['flowrate'] + 1e-6)
            features['resistance_log'] = np.log1p(features['resistance'])
            
            # Power consumption estimate
            features['power'] = features['flowrate'] * features['dp']
            
            # Efficiency metrics
            features['efficiency'] = features['flowrate'] / (features['dp'] + 1)
            
            # Rolling statistics (temporal patterns)
            for window in [5, 10, 20, 50]:
                # Rolling means
                features[f'flowrate_roll_mean_{window}'] = (
                    features['flowrate'].rolling(window, min_periods=1).mean()
                )
                features[f'dp_roll_mean_{window}'] = (
                    features['dp'].rolling(window, min_periods=1).mean()
                )
                
                # Rolling standard deviations (variability)
                features[f'flowrate_roll_std_{window}'] = (
                    features['flowrate'].rolling(window, min_periods=1).std().fillna(0)
                )
                features[f'dp_roll_std_{window}'] = (
                    features['dp'].rolling(window, min_periods=1).std().fillna(0)
                )
                
                # Rolling resistance
                features[f'resistance_roll_mean_{window}'] = (
                    features['resistance'].rolling(window, min_periods=1).mean()
                )
            
            # Rate of change features
            features['flowrate_change'] = features['flowrate'].diff().fillna(0)
            features['dp_change'] = features['dp'].diff().fillna(0)
            features['resistance_change'] = features['resistance'].diff().fillna(0)
            
            # Acceleration (second derivative)
            features['flowrate_accel'] = features['flowrate_change'].diff().fillna(0)
            features['dp_accel'] = features['dp_change'].diff().fillna(0)
            
            # Interaction features
            features['flow_dp_product'] = features['flowrate'] * features['dp']
            features['flow_dp_ratio'] = features['flowrate'] / (features['dp'] + 1e-6)
            
            # Polynomial features
            features['flowrate_squared'] = features['flowrate'] ** 2
            features['dp_squared'] = features['dp'] ** 2
            features['resistance_squared'] = features['resistance'] ** 2
        
        # EMA features (if available)
        if 'short_ema' in features.columns and 'long_ema' in features.columns:
            features['ema_diff'] = features['short_ema'] - features['long_ema']
            features['ema_ratio'] = features['short_ema'] / (features['long_ema'] + 1e-6)
            features['ema_cross'] = (features['ema_diff'] > 0).astype(int)
            
            # EMA momentum
            features['ema_momentum'] = features['ema_diff'].diff().fillna(0)
        
        # CUSUM features
        if 'adaptive_cusum_thr' in features.columns:
            features['cusum_normalized'] = (
                features['adaptive_cusum_thr'] / 
                (features['adaptive_cusum_thr'].abs().max() + 1e-6)
            )
            features['cusum_change'] = features['adaptive_cusum_thr'].diff().fillna(0)
        
        # Statistical features for all sensors
        for col in available_sensors:
            if col in features.columns:
                # Z-score normalization
                mean_val = features[col].mean()
                std_val = features[col].std() + 1e-6
                features[f'{col}_zscore'] = (features[col] - mean_val) / std_val
                
                # Percentile ranks
                features[f'{col}_rank'] = features[col].rank(pct=True)
                
                # Lag features
                for lag in [1, 5, 10]:
                    features[f'{col}_lag_{lag}'] = features[col].shift(lag).fillna(0)
        
        # Remove any infinite values and fill NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Store feature names
        self.feature_names = list(features.columns)
        logger.info(f"Created {len(self.feature_names)} features")
        
        return features


# ==============================================================================
# MAIN FILTER ANALYSIS SYSTEM
# ==============================================================================

class FilterAnalysisSystem:
    """
    Complete filter analysis system with GA optimization.
    
    Integrates feature engineering, GA optimization, model training,
    and prediction into a unified system.
    """
    
    def __init__(self, use_optimization: bool = True, ga_config: Dict[str, Any] = None):
        """
        Initialize the filter analysis system.
        
        Args:
            use_optimization: Whether to use GA optimization
            ga_config: Custom GA configuration (optional)
        """
        self.use_optimization = use_optimization
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.optimizer = None
        self.best_params = None
        self.is_trained = False
        
        # GA configuration
        if use_optimization:
            ga_params = ga_config or {
                'population_size': 20,
                'n_generations': 15,
                'mutation_rate': 0.2,
                'elite_size': 4
            }
            self.optimizer = GeneticAlgorithmOptimizer(**ga_params)
        
        # Create necessary directories
        for dir_name in ['models', 'reports', 'visualizations', 'logs']:
            Path(dir_name).mkdir(exist_ok=True)
        
        logger.info(f"Filter Analysis System initialized (GA Optimization: {use_optimization})")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file."""
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None
    
    def create_target(self, df: pd.DataFrame) -> np.ndarray:
        """Create target variable from labels or sensor patterns."""
        # Try to use existing labels
        label_cols = ['validation_filter_status', 'filter_status', 'status', 'label', 'target']
        
        for col in label_cols:
            if col in df.columns:
                logger.info(f"Using existing label column: {col}")
                
                # Convert to binary
                target = pd.Series(0, index=df.index)
                
                # Check for clogged indicators
                clogged_keywords = ['clogged', 'blocked', 'failed', 'critical', '1', 1, True]
                
                for keyword in clogged_keywords:
                    if isinstance(keyword, str):
                        mask = df[col].astype(str).str.lower().str.contains(keyword, na=False)
                    else:
                        mask = df[col] == keyword
                    target[mask] = 1
                
                # Log distribution
                unique, counts = np.unique(target, return_counts=True)
                distribution = dict(zip(unique, counts))
                logger.info(f"Target distribution: {distribution}")
                
                if len(unique) >= 2:
                    return target.values
                else:
                    logger.warning(f"Only one class found in {col}, trying synthetic target")
        
        # Fallback: Create synthetic target based on sensor patterns
        logger.info("Creating synthetic target from sensor patterns")
        target = np.zeros(len(df))
        
        if 'dp' in df.columns and 'flowrate' in df.columns:
            dp_values = pd.to_numeric(df['dp'], errors='coerce').fillna(0)
            flow_values = pd.to_numeric(df['flowrate'], errors='coerce').fillna(100)
            
            # High pressure and low flow indicates clogging
            dp_threshold = dp_values.quantile(0.7)
            flow_threshold = flow_values.quantile(0.3)
            
            clogged_mask = (dp_values > dp_threshold) & (flow_values < flow_threshold)
            target[clogged_mask] = 1
            
            logger.info(f"Synthetic target - Clogged: {clogged_mask.sum()}, Normal: {(~clogged_mask).sum()}")
        else:
            # Simple split for demo
            target[len(target)//2:] = 1
            logger.warning("Using simple 50/50 split for target")
        
        return target
    
    def train(self, file_path: str) -> bool:
        """
        Train the filter analysis model.
        
        Args:
            file_path: Path to training data
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("="*70)
            logger.info("STARTING MODEL TRAINING")
            logger.info("="*70)
            
            # Load data
            df = self.load_data(file_path)
            if df is None:
                return False
            
            # Create features
            logger.info("Engineering features...")
            X = self.feature_engineer.fit_transform(df)
            
            # Create target
            y = self.create_target(df)
            
            # Check class balance
            unique, counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip(unique, counts))
            logger.info(f"Class distribution: {class_distribution}")
            
            if len(unique) < 2:
                logger.error("Need at least 2 classes for classification")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Optimize hyperparameters or use defaults
            if self.use_optimization:
                logger.info("Starting Genetic Algorithm optimization...")
                best_params, best_cv_score = self.optimizer.optimize(X_train_scaled, y_train)
                self.best_params = best_params
                
                # Clean parameters for RandomForest
                model_params = best_params.copy()
                for param in ['max_samples', 'min_impurity_decrease']:
                    model_params.pop(param, None)
                
                # Ensure correct types
                for int_param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                    if int_param in model_params:
                        model_params[int_param] = int(model_params[int_param])
                
                logger.info(f"GA Optimization complete. Best CV F1: {best_cv_score:.4f}")
                
            else:
                # Use default parameters
                logger.info("Using default hyperparameters")
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'criterion': 'gini',
                    'bootstrap': True
                }
                self.best_params = model_params
            
            # Train final model
            logger.info("Training final model with optimized parameters...")
            self.model = RandomForestClassifier(
                **model_params,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_proba = self.model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC for binary classification
            if len(unique) == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = 0.0
            
            # Log results
            logger.info("="*70)
            logger.info("MODEL EVALUATION RESULTS")
            logger.info("="*70)
            logger.info(f"Accuracy:  {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall:    {recall:.4f}")
            logger.info(f"F1 Score:  {f1:.4f}")
            if roc_auc > 0:
                logger.info(f"ROC AUC:   {roc_auc:.4f}")
            
            # Feature importance analysis
            feature_importance = sorted(
                zip(self.feature_engineer.feature_names, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )
            
            logger.info("\nTop 15 Most Important Features:")
            for i, (feat, imp) in enumerate(feature_importance[:15], 1):
                logger.info(f"  {i:2d}. {feat:30s}: {imp:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
            logger.info(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
            
            # Classification Report
            report = classification_report(y_test, y_pred, target_names=['Normal', 'Clogged'])
            logger.info(f"\nDetailed Classification Report:")
            logger.info(report)
            
            # Save model and results
            self.save_model()
            self.save_report(accuracy, precision, recall, f1, roc_auc, feature_importance)
            
            self.is_trained = True
            logger.info("="*70)
            logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, sensor_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction for new sensor data.
        
        Args:
            sensor_data: Dictionary with sensor readings
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Create DataFrame
            df = pd.DataFrame([sensor_data])
            
            # Create features
            features = self.feature_engineer.transform(df)
            
            # Handle missing features
            for feat in self.feature_engineer.feature_names:
                if feat not in features.columns:
                    features[feat] = 0
            
            # Ensure correct order
            features = features[self.feature_engineer.feature_names]
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Calculate feature contributions (simplified SHAP-like explanation)
            feature_values = features.values[0]
            feature_importances = self.model.feature_importances_
            contributions = feature_values * feature_importances
            
            # Get top contributing features
            top_contributors = sorted(
                zip(self.feature_engineer.feature_names, contributions),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]
            
            # Determine confidence level
            prob_clogged = probability[1] if len(probability) > 1 else probability[0]
            if prob_clogged > 0.8 or prob_clogged < 0.2:
                confidence = 'High'
            elif prob_clogged > 0.6 or prob_clogged < 0.4:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            # Generate recommendations
            recommendations = self._generate_recommendations(prediction, prob_clogged, sensor_data)
            
            result = {
                'prediction': 'Clogged' if prediction == 1 else 'Normal',
                'probability': float(prob_clogged),
                'confidence': confidence,
                'top_factors': top_contributors,
                'recommendations': recommendations,
                'sensor_status': self._check_sensor_status(sensor_data)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_recommendations(self, prediction: int, probability: float, 
                                 sensor_data: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on prediction."""
        recommendations = []
        
        if prediction == 1:  # Clogged
            if probability > 0.9:
                recommendations.append("🚨 CRITICAL: Immediate filter replacement required")
                recommendations.append("⚠️ Stop operation and replace filter within 1 hour")
            elif probability > 0.7:
                recommendations.append("⚠️ HIGH RISK: Schedule filter replacement within 4 hours")
                recommendations.append("📊 Increase monitoring frequency to every 15 minutes")
            else:
                recommendations.append("⚡ MODERATE RISK: Plan filter maintenance within 24 hours")
                recommendations.append("👁️ Monitor closely for further degradation")
        else:  # Normal
            if probability < 0.2:
                recommendations.append("✅ Filter operating normally")
                recommendations.append("📅 Maintain regular maintenance schedule")
            else:
                recommendations.append("👀 Early warning signs detected")
                recommendations.append("📈 Continue monitoring sensor trends")
        
        # Sensor-specific recommendations
        if 'dp' in sensor_data and sensor_data['dp'] > 30:
            recommendations.append(f"📍 Pressure drop elevated ({sensor_data['dp']:.1f})")
        
        if 'flowrate' in sensor_data and sensor_data['flowrate'] < 70:
            recommendations.append(f"📉 Flow rate below optimal ({sensor_data['flowrate']:.1f})")
        
        return recommendations
    
    def _check_sensor_status(self, sensor_data: Dict[str, float]) -> Dict[str, str]:
        """Check status of each sensor."""
        expected_sensors = ['flowrate', 'dp', 'short_ema', 'long_ema', 'adaptive_cusum_thr']
        status = {}
        
        for sensor in expected_sensors:
            if sensor in sensor_data:
                value = sensor_data[sensor]
                if pd.isna(value) or np.isinf(value):
                    status[sensor] = 'Error'
                else:
                    status[sensor] = 'OK'
            else:
                status[sensor] = 'Missing'
        
        return status
    
    def save_model(self):
        """Save trained model and components."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path('models') / f'filter_model_GA_{timestamp}'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        joblib.dump(self.model, save_dir / 'model.pkl')
        joblib.dump(self.scaler, save_dir / 'scaler.pkl')
        joblib.dump(self.feature_engineer, save_dir / 'feature_engineer.pkl')
        joblib.dump(self.best_params, save_dir / 'best_params.pkl')
        
        # Save GA optimization history if available
        if self.optimizer and self.optimizer.optimization_history:
            joblib.dump(self.optimizer.optimization_history, save_dir / 'ga_history.pkl')
            
            # Save optimization report
            with open(save_dir / 'ga_report.txt', 'w') as f:
                f.write(self.optimizer.get_optimization_report())
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'version': '2.0.0',
            'use_optimization': self.use_optimization,
            'best_params': self.best_params,
            'feature_count': len(self.feature_engineer.feature_names),
            'features': self.feature_engineer.feature_names
        }
        
        with open(save_dir / 'metadata.json', 'w') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {save_dir}")
    
    def save_report(self, accuracy: float, precision: float, recall: float, 
                   f1: float, roc_auc: float, feature_importance: List[Tuple[str, float]]):
        """Save comprehensive training report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path('reports') / f'training_report_GA_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FILTER ANALYSIS SYSTEM - TRAINING REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"System Version: 2.0.0\n")
            f.write(f"Optimization: {'Genetic Algorithm' if self.use_optimization else 'None'}\n")
            f.write("\n")
            
            # Model Performance
            f.write("MODEL PERFORMANCE\n")
            f.write("-"*40 + "\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")
            if roc_auc > 0:
                f.write(f"ROC AUC:   {roc_auc:.4f}\n")
            f.write("\n")
            
            # Hyperparameters
            if self.best_params:
                f.write("OPTIMIZED HYPERPARAMETERS\n")
                f.write("-"*40 + "\n")
                for param, value in sorted(self.best_params.items()):
                    f.write(f"{param:25s}: {value}\n")
                f.write("\n")
            
            # GA Optimization Results
            if self.optimizer and self.optimizer.optimization_history:
                f.write(self.optimizer.get_optimization_report())
                f.write("\n")
            
            # Feature Importance
            f.write("TOP 20 FEATURE IMPORTANCES\n")
            f.write("-"*40 + "\n")
            for i, (feat, imp) in enumerate(feature_importance[:20], 1):
                f.write(f"{i:3d}. {feat:35s}: {imp:.6f}\n")
            f.write("\n")
            
            # System Configuration
            f.write("SYSTEM CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Features: {len(self.feature_engineer.feature_names)}\n")
            f.write(f"GA Population Size: {self.optimizer.population_size if self.optimizer else 'N/A'}\n")
            f.write(f"GA Generations: {self.optimizer.n_generations if self.optimizer else 'N/A'}\n")
            f.write(f"GA Mutation Rate: {self.optimizer.mutation_rate if self.optimizer else 'N/A'}\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Training report saved to {report_path}")
    
    def load_model(self, model_dir: str) -> bool:
        """Load a saved model."""
        try:
            model_path = Path(model_dir)
            
            # Load components
            self.model = joblib.load(model_path / 'model.pkl')
            self.scaler = joblib.load(model_path / 'scaler.pkl')
            self.feature_engineer = joblib.load(model_path / 'feature_engineer.pkl')
            self.best_params = joblib.load(model_path / 'best_params.pkl')
            
            self.is_trained = True
            logger.info(f"Model loaded from {model_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print(" FILTER ANALYSIS SYSTEM WITH GENETIC ALGORITHM OPTIMIZER")
    print("="*70)
    print("Version: 2.0.0")
    print("Features: GA Optimization, Advanced Feature Engineering, Auto-labeling")
    print("="*70)
    
    # Get user preferences
    use_optimization = input("\n📊 Enable Genetic Algorithm optimization? (y/n): ").lower() == 'y'
    
    if use_optimization:
        print("\n🧬 GA Configuration (press Enter for defaults):")
        pop_size = input("  Population size [20]: ").strip()
        n_gen = input("  Number of generations [15]: ").strip()
        mut_rate = input("  Mutation rate [0.2]: ").strip()
        
        ga_config = {
            'population_size': int(pop_size) if pop_size else 20,
            'n_generations': int(n_gen) if n_gen else 15,
            'mutation_rate': float(mut_rate) if mut_rate else 0.2,
            'elite_size': 4
        }
        
        system = FilterAnalysisSystem(use_optimization=True, ga_config=ga_config)
    else:
        system = FilterAnalysisSystem(use_optimization=False)
    
    # Check for data file
    data_file = 'Comprehensive_Filter_Analysis.xlsx'
    
    if not Path(data_file).exists():
        print(f"\n⚠️  {data_file} not found")
        
        # Create sample data
        create_sample = input("Create sample data for demonstration? (y/n): ").lower() == 'y'
        
        if create_sample:
            print("\n📝 Creating sample data...")
            np.random.seed(42)
            n_samples = 1000
            
            # Create realistic sensor data
            time_points = pd.date_range('2024-01-01', periods=n_samples, freq='H')
            
            # Simulate filter degradation over time
            degradation_curve = np.linspace(0, 1, n_samples) + np.random.normal(0, 0.1, n_samples)
            degradation_curve = np.clip(degradation_curve, 0, 1)
            
            sample_data = {
                'time': time_points,
                'flowrate': 100 - 40 * degradation_curve + np.random.normal(0, 5, n_samples),
                'dp': 10 + 30 * degradation_curve + np.random.normal(0, 3, n_samples),
                'short_ema': 50 + 10 * degradation_curve + np.random.normal(0, 2, n_samples),
                'long_ema': 45 + 8 * degradation_curve + np.random.normal(0, 2, n_samples),
                'adaptive_cusum_thr': np.cumsum(np.random.normal(0.1 * degradation_curve, 0.5, n_samples)),
                'validation_filter_status': ['Not Clogged'] * (n_samples//2) + ['Clogged'] * (n_samples//2)
            }
            
            df = pd.DataFrame(sample_data)
            data_file = 'sample_filter_data.csv'
            df.to_csv(data_file, index=False)
            print(f"✅ Sample data saved to {data_file}")
        else:
            print("Exiting. Please provide data file.")
            return
    
    # Train model
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING PROCESS")
    print("="*70)
    
    if use_optimization:
        print("🧬 Using Genetic Algorithm optimization")
        print("⏳ This may take several minutes...")
    else:
        print("⚡ Using default hyperparameters (fast mode)")
    
    success = system.train(data_file)
    
    if success:
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Interactive prediction demo
        print("\n" + "="*70)
        print("🔮 PREDICTION DEMO")
        print("="*70)
        
        while True:
            print("\nEnter sensor values (or 'quit' to exit):")
            
            try:
                flowrate = input("  Flowrate [75]: ").strip()
                if flowrate.lower() == 'quit':
                    break
                flowrate = float(flowrate) if flowrate else 75.0
                
                dp = input("  Pressure Drop [25]: ").strip()
                dp = float(dp) if dp else 25.0
                
                short_ema = input("  Short EMA [30]: ").strip()
                short_ema = float(short_ema) if short_ema else 30.0
                
                long_ema = input("  Long EMA [28]: ").strip()
                long_ema = float(long_ema) if long_ema else 28.0
                
                cusum = input("  CUSUM Threshold [0.5]: ").strip()
                cusum = float(cusum) if cusum else 0.5
                
                test_data = {
                    'flowrate': flowrate,
                    'dp': dp,
                    'short_ema': short_ema,
                    'long_ema': long_ema,
                    'adaptive_cusum_thr': cusum
                }
                
                # Make prediction
                result = system.predict(test_data)
                
                if result:
                    print("\n" + "-"*50)
                    print("📊 PREDICTION RESULTS")
                    print("-"*50)
                    print(f"🎯 Prediction: {result['prediction']}")
                    print(f"📈 Probability of Clogging: {result['probability']:.1%}")
                    print(f"💪 Confidence Level: {result['confidence']}")
                    
                    print(f"\n🔍 Sensor Status:")
                    for sensor, status in result['sensor_status'].items():
                        icon = "✅" if status == "OK" else "❌"
                        print(f"  {icon} {sensor}: {status}")
                    
                    print(f"\n📋 Recommendations:")
                    for rec in result['recommendations']:
                        print(f"  {rec}")
                    
                    if result['top_factors']:
                        print(f"\n🎲 Top Contributing Factors:")
                        for i, (factor, contrib) in enumerate(result['top_factors'][:5], 1):
                            print(f"  {i}. {factor}: {contrib:.4f}")
                    
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
            except KeyboardInterrupt:
                break
        
        print("\n" + "="*70)
        print("📁 SYSTEM OUTPUTS")
        print("="*70)
        print("✅ Models saved in:        models/")
        print("✅ Reports saved in:       reports/")
        print("✅ GA optimization in:     models/*/ga_report.txt")
        print("="*70)
        print("Thank you for using the Filter Analysis System!")
        
    else:
        print("\n❌ Training failed - check logs for details")


if __name__ == "__main__":
    main()