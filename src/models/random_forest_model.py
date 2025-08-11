import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest regression model for HR prediction."""
    
    def __init__(self, feature_columns=None, n_estimators=100, max_depth=None, 
                 min_samples_split=5, min_samples_leaf=2, normalize=False):
        super().__init__("RandomForest", feature_columns)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.normalize = normalize
        
        self.scaler = StandardScaler() if normalize else None
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Random Forest model."""
        X_prepared = self.prepare_features(X)
        
        # Remove rows with NaN in target
        mask = ~y.isna()
        X_clean = X_prepared[mask]
        y_clean = y[mask]
        
        if self.normalize and self.scaler:
            X_clean = self.scaler.fit_transform(X_clean)
        
        self.model.fit(X_clean, y_clean)
        self.is_trained = True
        
        # Store metrics
        self.metrics['training_samples'] = len(X_clean)
        self.metrics['n_features'] = X_clean.shape[1]
        self.metrics['n_estimators'] = self.n_estimators
        self.metrics['max_depth'] = self.max_depth
        self.metrics['min_samples_split'] = self.min_samples_split
        self.metrics['min_samples_leaf'] = self.min_samples_leaf
        
        # Evaluate on training data
        train_metrics = self.evaluate(X[mask], y[mask])
        self.metrics.update({f'train_{k}': v for k, v in train_metrics.items()})
        
        print(f"Random Forest model trained on {len(X_clean)} samples with {X_clean.shape[1]} features")
        print(f"Trees: {self.n_estimators}, Max depth: {self.max_depth}, Min samples split: {self.min_samples_split}")
        print(f"Training R²: {self.metrics['train_r2']:.4f}, MAE: {self.metrics['train_mae']:.2f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the Random Forest model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_prepared = self.prepare_features(X)
        
        if self.normalize and self.scaler:
            X_prepared = self.scaler.transform(X_prepared)
        
        return self.model.predict(X_prepared)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the Random Forest model."""
        if not self.is_trained:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns[:len(self.model.feature_importances_)],
            'importance': self.model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        return importance_df