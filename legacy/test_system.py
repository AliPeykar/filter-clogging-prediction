"""
FILTER ANALYSIS SYSTEM - SIMPLIFIED WORKING VERSION
===================================================
A streamlined version that will run without errors.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import random
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeneticAlgorithmOptimizer:
    """Genetic Algorithm for hyperparameter optimization."""
    
    def __init__(self, population_size=20, n_generations=15, mutation_rate=0.2, elite_size=4):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.best_params = None
        self.best_score = 0
        self.optimization_history = []
        
        # Define hyperparameter search space
        self.param_space = {
            'n_estimators': (50, 300),
            'max_depth': (3, 20),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'criterion': ['gini', 'entropy'],
            'bootstrap': [True, False]
        }
        
        logger.info("Genetic Algorithm Optimizer initialized")
    
    def create_individual(self):
        """Create a random individual (set of hyperparameters)."""
        individual = {}
        for param, values in self.param_space.items():
            if isinstance(values, tuple):
                # Continuous parameter
                individual[param] = random.randint(values[0], values[1])
            elif isinstance(values, list):
                # Categorical parameter
                individual[param] = random.choice(values)
        return individual
    
    def create_population(self):
        """Create initial population."""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def evaluate_fitness(self, individual, X_train, y_train):
        """Evaluate fitness of an individual using cross-validation."""
        try:
            # Create model with individual's parameters
            model = RandomForestClassifier(
                n_estimators=individual['n_estimators'],
                max_depth=individual['max_depth'],
                min_samples_split=individual['min_samples_split'],
                min_samples_leaf=individual['min_samples_leaf'],
                max_features=individual['max_features'],
                criterion=individual['criterion'],
                bootstrap=individual['bootstrap'],
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            # Use cross-validation for robust evaluation
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted')
            fitness = np.mean(scores)
            
            return fitness
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return 0
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        child = {}
        for param in self.param_space.keys():
            # Randomly inherit from either parent
            if random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child
    
    def mutate(self, individual):
        """Mutate an individual."""
        mutated = individual.copy()
        
        for param, values in self.param_space.items():
            if random.random() < self.mutation_rate:
                if isinstance(values, tuple):
                    # Mutate continuous parameter
                    current_val = mutated[param]
                    # Small mutation: change by ±20% of range
                    range_size = values[1] - values[0]
                    mutation = random.uniform(-0.2, 0.2) * range_size
                    new_val = int(current_val + mutation)
                    mutated[param] = max(values[0], min(values[1], new_val))
                elif isinstance(values, list):
                    # Mutate categorical parameter
                    mutated[param] = random.choice(values)
        
        return mutated
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index]
    
    def evolve_population(self, population, fitness_scores):
        """Evolve population to next generation."""
        new_population = []
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite individuals
        for i in range(self.elite_size):
            new_population.append(sorted_population[i])
        
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
    
    def optimize(self, X_train, y_train):
        """Run genetic algorithm optimization."""
        logger.info("Starting Genetic Algorithm optimization...")
        logger.info(f"Population size: {self.population_size}, Generations: {self.n_generations}")
        
        # Create initial population
        population = self.create_population()
        
        # Evolution loop
        for generation in range(self.n_generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_fitness(individual, X_train, y_train)
                fitness_scores.append(fitness)
            
            # Track best individual
            best_idx = np.argmax(fitness_scores)
            generation_best_score = fitness_scores[best_idx]
            generation_best_params = population[best_idx]
            
            # Update global best
            if generation_best_score > self.best_score:
                self.best_score = generation_best_score
                self.best_params = generation_best_params.copy()
            
            # Log progress
            avg_fitness = np.mean(fitness_scores)
            logger.info(f"Generation {generation + 1}/{self.n_generations} - "
                       f"Best: {generation_best_score:.4f}, Avg: {avg_fitness:.4f}")
            
            self.optimization_history.append({
                'generation': generation + 1,
                'best_fitness': generation_best_score,
                'avg_fitness': avg_fitness,
                'best_params': generation_best_params.copy()
            })
            
            # Evolve to next generation
            if generation < self.n_generations - 1:
                population = self.evolve_population(population, fitness_scores)
        
        logger.info(f"Optimization completed! Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_score
    
    def get_optimization_report(self):
        """Generate optimization report."""
        report = []
        report.append("\n" + "="*60)
        report.append("GENETIC ALGORITHM OPTIMIZATION REPORT")
        report.append("="*60)
        report.append(f"Final Best Score: {self.best_score:.4f}")
        report.append(f"Total Generations: {self.n_generations}")
        report.append(f"Population Size: {self.population_size}")
        report.append(f"Mutation Rate: {self.mutation_rate}")
        report.append(f"Elite Size: {self.elite_size}")
        report.append("\nBest Hyperparameters Found:")
        for param, value in self.best_params.items():
            report.append(f"  - {param}: {value}")
        
        report.append("\nEvolution Progress:")
        for i in range(0, len(self.optimization_history), max(1, len(self.optimization_history)//5)):
            gen_data = self.optimization_history[i]
            report.append(f"  Gen {gen_data['generation']:3d}: "
                         f"Best={gen_data['best_fitness']:.4f}, "
                         f"Avg={gen_data['avg_fitness']:.4f}")
        
        return "\n".join(report)


class FilterAnalysisSystem:
    """Simplified filter analysis system with Genetic Algorithm optimization."""
    
    def __init__(self, use_optimization=True):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.use_optimization = use_optimization
        self.optimizer = None
        self.best_params = None
        
        # Create directories
        for dir_name in ['models', 'reports', 'visualizations']:
            Path(dir_name).mkdir(exist_ok=True)
        
        logger.info(f"Filter Analysis System initialized (Optimization: {use_optimization})")
    
    def load_data(self, file_path):
        """Load data from Excel or CSV file."""
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            logger.info(f"Loaded {len(df)} samples with columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None
    
    def create_features(self, df):
        """Create features from sensor data."""
        features = pd.DataFrame()
        
        # Basic sensor features
        sensor_cols = ['flowrate', 'dp', 'short_ema', 'long_ema', 'adaptive_cusum_thr']
        available_sensors = [col for col in sensor_cols if col in df.columns]
        
        for col in available_sensors:
            if col in df.columns:
                features[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Derived features
        if 'flowrate' in features.columns and 'dp' in features.columns:
            # Filter resistance
            features['resistance'] = features['dp'] / (features['flowrate'] + 1e-6)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                features[f'flowrate_roll_{window}'] = features['flowrate'].rolling(window, min_periods=1).mean()
                features[f'dp_roll_{window}'] = features['dp'].rolling(window, min_periods=1).mean()
                
            # Rate of change
            features['flowrate_change'] = features['flowrate'].diff().fillna(0)
            features['dp_change'] = features['dp'].diff().fillna(0)
            
            # Interaction features
            features['flow_dp_product'] = features['flowrate'] * features['dp']
            features['flow_dp_ratio'] = features['flowrate'] / (features['dp'] + 1e-6)
        
        # EMA features
        if 'short_ema' in features.columns and 'long_ema' in features.columns:
            features['ema_diff'] = features['short_ema'] - features['long_ema']
            features['ema_ratio'] = features['short_ema'] / (features['long_ema'] + 1e-6)
        
        # Statistical features
        for col in available_sensors:
            if col in features.columns:
                features[f'{col}_zscore'] = (features[col] - features[col].mean()) / (features[col].std() + 1e-6)
        
        # Fill any remaining NaN values
        features = features.fillna(0)
        
        # Store feature names
        self.feature_names = list(features.columns)
        logger.info(f"Created {len(self.feature_names)} features")
        
        return features
    
    def create_target(self, df):
        """Create target variable from existing labels or sensor data."""
        # Try to use existing labels
        label_cols = ['validation_filter_status', 'filter_status', 'status']
        
        for col in label_cols:
            if col in df.columns:
                logger.info(f"Using existing label column: {col}")
                # Convert to binary
                target = pd.Series(0, index=df.index)
                clogged_keywords = ['clogged', 'blocked', 'failed', '1']
                
                for keyword in clogged_keywords:
                    mask = df[col].astype(str).str.lower().str.contains(keyword, na=False)
                    target[mask] = 1
                
                logger.info(f"Target distribution: {target.value_counts().to_dict()}")
                return target.values
        
        # Fallback: Create synthetic target based on sensor patterns
        logger.info("No label column found - creating synthetic target")
        target = np.zeros(len(df))
        
        if 'dp' in df.columns and 'flowrate' in df.columns:
            # High pressure drop and low flow indicates clogging
            dp_values = pd.to_numeric(df['dp'], errors='coerce').fillna(0)
            flow_values = pd.to_numeric(df['flowrate'], errors='coerce').fillna(0)
            
            dp_high = dp_values > dp_values.quantile(0.7)
            flow_low = flow_values < flow_values.quantile(0.3)
            
            target[dp_high & flow_low] = 1
        else:
            # Simple split for demo
            target[len(target)//2:] = 1
        
        logger.info(f"Synthetic target distribution: {pd.Series(target).value_counts().to_dict()}")
        return target
    
    def train(self, file_path):
        """Train the filter analysis model with optional GA optimization."""
        try:
            # Load data
            df = self.load_data(file_path)
            if df is None:
                return False
            
            # Create features and target
            X = self.create_features(df)
            y = self.create_target(df)
            
            # Check for class imbalance
            unique, counts = np.unique(y, return_counts=True)
            logger.info(f"Class distribution: {dict(zip(unique, counts))}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Optimize hyperparameters using Genetic Algorithm
            if self.use_optimization:
                logger.info("\n" + "="*60)
                logger.info("HYPERPARAMETER OPTIMIZATION USING GENETIC ALGORITHM")
                logger.info("="*60)
                
                self.optimizer = GeneticAlgorithmOptimizer(
                    population_size=20,
                    n_generations=15,
                    mutation_rate=0.2,
                    elite_size=4
                )
                
                best_params, best_score = self.optimizer.optimize(X_train_scaled, y_train)
                self.best_params = best_params
                
                # Print optimization report
                print(self.optimizer.get_optimization_report())
                
                # Train final model with best parameters
                logger.info("\nTraining final model with optimized parameters...")
                self.model = RandomForestClassifier(
                    **best_params,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                )
            else:
                # Use default parameters
                logger.info("Training Random Forest with default parameters...")
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
                self.best_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'criterion': 'gini',
                    'bootstrap': True
                }
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"\nFinal Model Performance - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            
            # Get feature importances
            feature_importance = sorted(
                zip(self.feature_names, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )
            
            print("\nTop 10 Most Important Features:")
            for i, (feat, imp) in enumerate(feature_importance[:10], 1):
                print(f"  {i:2d}. {feat:20s}: {imp:.4f}")
            
            # Generate report
            report = classification_report(y_test, y_pred)
            print("\nClassification Report:")
            print(report)
            
            # Save model
            self.save_model()
            self.is_trained = True
            
            # Save detailed report
            self.save_detailed_report(accuracy, f1, report, feature_importance)
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, sensor_data):
        """Make prediction for new sensor data."""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Create dataframe
            df = pd.DataFrame([sensor_data])
            
            # Create features
            features = pd.DataFrame()
            for col in self.feature_names:
                if col in df.columns:
                    features[col] = df[col]
                elif col == 'resistance' and 'dp' in df.columns and 'flowrate' in df.columns:
                    features[col] = df['dp'] / (df['flowrate'] + 1e-6)
                elif 'ema_diff' in col and 'short_ema' in df.columns and 'long_ema' in df.columns:
                    features[col] = df['short_ema'] - df['long_ema']
                else:
                    features[col] = 0
            
            # Scale
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Get feature contributions (simplified SHAP-like explanation)
            feature_values = features.values[0]
            feature_importances = self.model.feature_importances_
            contributions = feature_values * feature_importances
            
            top_contributors = sorted(
                zip(self.feature_names, contributions),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            result = {
                'prediction': 'Clogged' if prediction == 1 else 'Normal',
                'probability': float(probability[1]),
                'confidence': 'High' if probability[1] > 0.8 or probability[1] < 0.2 else 'Medium',
                'top_factors': top_contributors
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def save_model(self):
        """Save trained model and optimizer results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path('models') / f'filter_model_{timestamp}'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, save_dir / 'model.pkl')
        joblib.dump(self.scaler, save_dir / 'scaler.pkl')
        joblib.dump(self.feature_names, save_dir / 'features.pkl')
        joblib.dump(self.best_params, save_dir / 'best_params.pkl')
        
        if self.optimizer:
            joblib.dump(self.optimizer.optimization_history, save_dir / 'ga_history.pkl')
        
        logger.info(f"Model saved to {save_dir}")
    
    def load_model(self, model_dir):
        """Load a saved model."""
        try:
            model_path = Path(model_dir)
            self.model = joblib.load(model_path / 'model.pkl')
            self.scaler = joblib.load(model_path / 'scaler.pkl')
            self.feature_names = joblib.load(model_path / 'features.pkl')
            self.best_params = joblib.load(model_path / 'best_params.pkl')
            self.is_trained = True
            logger.info(f"Model loaded from {model_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def save_detailed_report(self, accuracy, f1, report, feature_importance):
        """Save detailed training report including GA results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path('reports') / f'report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("FILTER ANALYSIS TRAINING REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Optimization Used: {self.use_optimization}\n")
            f.write(f"\nModel Performance:\n")
            f.write(f"  Accuracy: {accuracy:.3f}\n")
            f.write(f"  F1 Score: {f1:.3f}\n")
            
            if self.best_params:
                f.write(f"\nOptimized Hyperparameters:\n")
                for param, value in self.best_params.items():
                    f.write(f"  {param}: {value}\n")
            
            if self.optimizer:
                f.write(self.optimizer.get_optimization_report())
            
            f.write("\nTop 15 Feature Importances:\n")
            for i, (feat, imp) in enumerate(feature_importance[:15], 1):
                f.write(f"  {i:2d}. {feat:25s}: {imp:.6f}\n")
            
            f.write("\nDetailed Classification Report:\n")
            f.write(report)
        
        logger.info(f"Detailed report saved to {report_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("FILTER ANALYSIS SYSTEM WITH GENETIC ALGORITHM OPTIMIZER")
    print("=" * 60)
    
    # Initialize system
    use_optimization = input("\nEnable Genetic Algorithm optimization? (y/n): ").lower() == 'y'
    system = FilterAnalysisSystem(use_optimization=use_optimization)
    
    # Check for data file
    data_file = 'Comprehensive_Filter_Analysis.xlsx'
    
    if not Path(data_file).exists():
        print(f"\n⚠ Warning: {data_file} not found")
        print("\nCreating sample data for demonstration...")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'time': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'flowrate': np.concatenate([
                np.random.normal(100, 5, n_samples//2),
                np.random.normal(60, 10, n_samples//2)
            ]),
            'dp': np.concatenate([
                np.random.normal(10, 2, n_samples//2),
                np.random.normal(40, 5, n_samples//2)
            ]),
            'short_ema': np.random.normal(50, 10, n_samples),
            'long_ema': np.random.normal(45, 8, n_samples),
            'adaptive_cusum_thr': np.random.normal(0, 1, n_samples),
            'validation_filter_status': ['Not Clogged'] * (n_samples//2) + ['Clogged'] * (n_samples//2)
        }
        
        df = pd.DataFrame(sample_data)
        data_file = 'sample_filter_data.csv'
        df.to_csv(data_file, index=False)
        print(f"Sample data saved to {data_file}")
    
    # Train model
    print("\nStarting training...")
    if use_optimization:
        print("Using Genetic Algorithm for hyperparameter optimization")
        print("This may take several minutes...")
    
    success = system.train(data_file)
    
    if success:
        print("\n✅ Training completed successfully!")
        
        # Demo prediction
        print("\n" + "=" * 60)
        print("DEMO PREDICTION")
        print("=" * 60)
        
        test_data = {
            'flowrate': 75.0,
            'dp': 25.0,
            'short_ema': 30.0,
            'long_ema': 28.0,
            'adaptive_cusum_thr': 0.5
        }
        
        result = system.predict(test_data)
        
        if result:
            print(f"\nTest Input: {test_data}")
            print(f"\nPrediction: {result['prediction']}")
            print(f"Probability of Clogging: {result['probability']:.3f}")
            print(f"Confidence Level: {result['confidence']}")
            
            if 'top_factors' in result:
                print("\nTop Contributing Factors:")
                for i, (factor, contrib) in enumerate(result['top_factors'], 1):
                    print(f"  {i}. {factor}: {contrib:.4f}")
        
        print("\n✅ System is ready for use!")
        print("\nOutputs saved in:")
        print("  - models/    (trained models with GA-optimized parameters)")
        print("  - reports/   (training reports with GA optimization history)")
        
    else:
        print("\n❌ Training failed - check logs for details")


if __name__ == "__main__":
    main()
