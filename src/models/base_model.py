from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import joblib


class BaseModel(ABC):
    """Abstract base class for all regression models."""
    
    def __init__(self, name: str, feature_columns: Optional[List[str]] = None):
        self.name = name
        self.model = None
        self.feature_columns = feature_columns or self._default_features()
        self.metrics = {}
        self.is_trained = False
    
    def _default_features(self) -> List[str]:
        """Default feature columns."""
        return [
            'cad', 'power', 'ele',
            'cad5', 'cad10', 'cad30', 'cad60',
            'power5', 'power10', 'power30', 'power60'
        ]
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the model."""
        # Select only available features
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        if len(available_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            print(f"Warning: Missing features {missing}")
        
        return df[available_features].fillna(0)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        
        # Remove NaN values
        mask = ~(np.isnan(y) | np.isnan(y_pred))
        y_clean = y[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_clean) == 0:
            return {}
        
        # Calculate metrics
        mae = np.mean(np.abs(y_clean - y_pred_clean))
        rmse = np.sqrt(np.mean((y_clean - y_pred_clean) ** 2))
        mape = np.mean(np.abs((y_clean - y_pred_clean) / y_clean)) * 100
        
        # R-squared
        ss_res = np.sum((y_clean - y_pred_clean) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'n_samples': len(y_clean)
        }
    
    def save(self, filepath: Path) -> None:
        """Save model to file."""
        model_data = {
            'name': self.name,
            'model': self.model,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: Path) -> None:
        """Load model from file."""
        model_data = joblib.load(filepath)
        self.name = model_data['name']
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        return None