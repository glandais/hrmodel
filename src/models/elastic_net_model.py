import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel


class ElasticNetModel(BaseModel):
    """Elastic Net regression model with L1 and L2 regularization."""
    
    def __init__(self, feature_columns=None, alpha=1.0, l1_ratio=0.5, normalize=True):
        super().__init__("ElasticNet", feature_columns)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Elastic Net model."""
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
        self.metrics['alpha'] = self.alpha
        self.metrics['l1_ratio'] = self.l1_ratio
        self.metrics['n_iter'] = self.model.n_iter_
        
        # Count selected features (non-zero coefficients)
        n_selected = np.sum(np.abs(self.model.coef_) > 1e-5)
        self.metrics['n_selected_features'] = n_selected
        
        # Evaluate on training data
        train_metrics = self.evaluate(X[mask], y[mask])
        self.metrics.update({f'train_{k}': v for k, v in train_metrics.items()})
        
        print(f"Elastic Net model trained on {len(X_clean)} samples with {X_clean.shape[1]} features")
        print(f"Alpha: {self.alpha}, L1_ratio: {self.l1_ratio}, Selected features: {n_selected}/{X_clean.shape[1]}")
        print(f"Training R²: {self.metrics['train_r2']:.4f}, MAE: {self.metrics['train_mae']:.2f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the Elastic Net model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_prepared = self.prepare_features(X)
        
        if self.normalize and self.scaler:
            X_prepared = self.scaler.transform(X_prepared)
        
        return self.model.predict(X_prepared)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature coefficients as importance."""
        if not self.is_trained:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns[:len(self.model.coef_)],
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        })
        
        # Mark selected features (non-zero coefficients)
        importance_df['selected'] = importance_df['abs_coefficient'] > 1e-5
        
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df