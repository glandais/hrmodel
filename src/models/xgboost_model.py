import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from .base_model import BaseModel

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")


class XGBoostModel(BaseModel):
    """XGBoost regression model for HR prediction."""
    
    def __init__(self, feature_columns=None, n_estimators=100, max_depth=6, 
                 learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, 
                 normalize=False, random_state=42):
        super().__init__("XGBoost", feature_columns)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.normalize = normalize
        
        self.scaler = StandardScaler() if normalize else None
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0  # Suppress XGBoost output
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the XGBoost model."""
        X_prepared = self.prepare_features(X)
        
        # Remove rows with NaN in target
        mask = ~y.isna()
        X_clean = X_prepared[mask]
        y_clean = y[mask]
        
        if self.normalize and self.scaler:
            X_clean = self.scaler.fit_transform(X_clean)
        
        # Train XGBoost model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress XGBoost warnings
            self.model.fit(X_clean, y_clean)
        
        self.is_trained = True
        
        # Store metrics
        self.metrics['training_samples'] = len(X_clean)
        self.metrics['n_features'] = X_clean.shape[1]
        self.metrics['n_estimators'] = self.n_estimators
        self.metrics['max_depth'] = self.max_depth
        self.metrics['learning_rate'] = self.learning_rate
        self.metrics['subsample'] = self.subsample
        self.metrics['colsample_bytree'] = self.colsample_bytree
        
        # Evaluate on training data
        train_metrics = self.evaluate(X[mask], y[mask])
        self.metrics.update({f'train_{k}': v for k, v in train_metrics.items()})
        
        print(f"XGBoost model trained on {len(X_clean)} samples with {X_clean.shape[1]} features")
        print(f"Trees: {self.n_estimators}, Max depth: {self.max_depth}, LR: {self.learning_rate}")
        print(f"Training R²: {self.metrics['train_r2']:.4f}, MAE: {self.metrics['train_mae']:.2f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the XGBoost model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_prepared = self.prepare_features(X)
        
        if self.normalize and self.scaler:
            X_prepared = self.scaler.transform(X_prepared)
        
        return self.model.predict(X_prepared)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the XGBoost model."""
        if not self.is_trained:
            return None
        
        # Get feature importance (gain-based by default)
        importance_gain = self.model.feature_importances_
        
        # Also get other importance types
        if hasattr(self.model, 'get_booster'):
            booster = self.model.get_booster()
            importance_weight = booster.get_score(importance_type='weight')
            importance_cover = booster.get_score(importance_type='cover')
        else:
            importance_weight = {}
            importance_cover = {}
        
        # Create importance dataframe
        feature_names = self.feature_columns[:len(importance_gain)]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_gain': importance_gain,
            'importance_weight': [importance_weight.get(f'f{i}', 0) for i in range(len(feature_names))],
            'importance_cover': [importance_cover.get(f'f{i}', 0) for i in range(len(feature_names))]
        })
        
        # Use gain as primary importance and add derived columns
        importance_df['importance'] = importance_df['importance_gain']  # For compatibility
        importance_df = importance_df.sort_values('importance_gain', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        importance_df['cumulative_importance'] = importance_df['importance_gain'].cumsum()
        
        return importance_df
    
    def get_model_info(self) -> dict:
        """Get detailed model information."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "model_type": "XGBoost",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "n_features": len(self.feature_columns),
            "training_samples": self.metrics.get('training_samples', 0)
        }
        
        if hasattr(self.model, 'get_booster'):
            booster = self.model.get_booster()
            info['actual_trees'] = booster.num_boosted_rounds()
        
        return info
    
    def tune_hyperparameters(self, X, y, n_trials=50, cv_folds=5):
        """Tune hyperparameters using Optuna (Bayesian optimization)."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna")
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        print(f"🔧 Starting XGBoost hyperparameter tuning with {n_trials} trials...")
        print(f"🎯 Target: Beat current 3.12 bpm MAE")
        print(f"📊 Training data: {len(X)} samples, {X.shape[1]} features")
        
        # Prepare features for tuning
        X_prepared = self.prepare_features(X)
        mask = ~y.isna()
        X_clean = X_prepared[mask]
        y_clean = y[mask]
        
        def objective(trial):
            """Objective function for Optuna optimization."""
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),  # L1 regularization
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),  # L2 regularization
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),  # Minimum loss reduction
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            # Create XGBoost model with trial parameters
            model = xgb.XGBRegressor(**params)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_clean, y_clean,
                cv=cv_folds,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            # Return negative MAE (Optuna minimizes)
            mae = -cv_scores.mean()
            return mae
        
        # Create study and optimize
        study = optuna.create_study(
            direction='minimize',
            study_name='xgboost_hr_prediction',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        print("🚀 Starting hyperparameter search...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        best_mae = study.best_value
        
        print(f"\n🏆 BEST HYPERPARAMETERS FOUND:")
        print(f"   📉 Best MAE: {best_mae:.3f} bpm")
        print(f"   🌳 Trees: {best_params['n_estimators']}")
        print(f"   📏 Max depth: {best_params['max_depth']}")
        print(f"   📚 Learning rate: {best_params['learning_rate']:.4f}")
        print(f"   🎲 Subsample: {best_params['subsample']:.3f}")
        print(f"   📊 Col sample (tree): {best_params['colsample_bytree']:.3f}")
        print(f"   📊 Col sample (level): {best_params['colsample_bylevel']:.3f}")
        print(f"   📊 Col sample (node): {best_params['colsample_bynode']:.3f}")
        print(f"   ⚖️  L1 reg (alpha): {best_params['reg_alpha']:.4f}")
        print(f"   ⚖️  L2 reg (lambda): {best_params['reg_lambda']:.4f}")
        print(f"   👶 Min child weight: {best_params['min_child_weight']}")
        print(f"   🔥 Gamma: {best_params['gamma']:.4f}")
        
        improvement = ((3.12 - best_mae) / 3.12) * 100 if best_mae < 3.12 else 0
        if improvement > 0:
            print(f"📊 Improvement: {improvement:.1f}% better than current 3.12 bpm")
        else:
            print(f"📊 Current best: {best_mae:.3f} vs previous 3.12 bpm")
        
        # Update model with best parameters
        self.n_estimators = best_params['n_estimators']
        self.max_depth = best_params['max_depth']
        self.learning_rate = best_params['learning_rate']
        self.subsample = best_params['subsample']
        self.colsample_bytree = best_params['colsample_bytree']
        
        # Store additional parameters
        self.colsample_bylevel = best_params['colsample_bylevel']
        self.colsample_bynode = best_params['colsample_bynode']
        self.reg_alpha = best_params['reg_alpha']
        self.reg_lambda = best_params['reg_lambda']
        self.min_child_weight = best_params['min_child_weight']
        self.gamma = best_params['gamma']
        
        # Recreate model with optimized parameters
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        # Store optimization results
        self.tuning_results = {
            'best_params': best_params,
            'best_mae': best_mae,
            'n_trials': n_trials,
            'study': study
        }
        
        return best_params, best_mae