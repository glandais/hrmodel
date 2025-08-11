import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel

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