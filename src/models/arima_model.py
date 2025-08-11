import pandas as pd
import numpy as np
import warnings
from .base_model import BaseModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available. Install with: pip install statsmodels")


class ARIMAModel(BaseModel):
    """ARIMA model with exogenous variables for HR prediction.
    
    ARIMA (AutoRegressive Integrated Moving Average) is a time-series forecasting method
    that combines autoregression, differencing, and moving averages. When combined with
    exogenous variables (ARIMAX), it can use external features like power and cadence
    to improve predictions.
    """
    
    def __init__(self, feature_columns=None, 
                 order=(2, 1, 2), seasonal_order=(0, 0, 0, 0),
                 use_exog=True, normalize=False):
        """Initialize ARIMA model.
        
        Args:
            feature_columns: List of feature column names
            order: (p, d, q) where:
                p = order of autoregression
                d = degree of differencing
                q = order of moving average
            seasonal_order: (P, D, Q, s) for seasonal components
            use_exog: Whether to use exogenous variables
            normalize: Whether to normalize features
        """
        super().__init__("ARIMA", feature_columns)
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels is not available. Install with: pip install statsmodels")
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_exog = use_exog
        self.normalize = normalize
        self.model = None
        self.fitted_model = None
        
        # Store time series data for forecasting
        self.train_endog = None
        self.train_exog = None
        self.train_index = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the ARIMA model."""
        # Store original data
        self.train_endog = y.copy()
        self.train_index = y.index if hasattr(y, 'index') else range(len(y))
        
        if self.use_exog and self.feature_columns:
            X_prepared = self.prepare_features(X)
            self.train_exog = X_prepared
            
            # Remove rows with NaN
            mask = ~(y.isna() | X_prepared.isna().any(axis=1))
            y_clean = y[mask]
            X_clean = X_prepared[mask]
        else:
            mask = ~y.isna()
            y_clean = y[mask]
            X_clean = None
        
        # Subsample for faster training on large datasets
        max_samples = 10000  # Limit to 10k samples for ARIMA
        if len(y_clean) > max_samples:
            print(f"Subsampling ARIMA training data from {len(y_clean)} to {max_samples} samples")
            sample_indices = np.linspace(0, len(y_clean)-1, max_samples, dtype=int)
            y_clean = y_clean.iloc[sample_indices] if hasattr(y_clean, 'iloc') else y_clean[sample_indices]
            if X_clean is not None:
                X_clean = X_clean.iloc[sample_indices] if hasattr(X_clean, 'iloc') else X_clean[sample_indices]
        
        try:
            # Use SARIMAX for more flexibility
            self.model = SARIMAX(
                endog=y_clean,
                exog=X_clean if self.use_exog else None,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            # Fit the model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fitted_model = self.model.fit(disp=False, maxiter=100)
            
            self.is_trained = True
            
            # Store metrics
            self.metrics['training_samples'] = len(y_clean)
            self.metrics['n_features'] = X_clean.shape[1] if X_clean is not None else 0
            self.metrics['aic'] = self.fitted_model.aic
            self.metrics['bic'] = self.fitted_model.bic
            
            # Evaluate on training data (in-sample)
            train_pred = self.fitted_model.fittedvalues
            train_mae = mean_absolute_error(y_clean, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_clean, train_pred))
            train_r2 = r2_score(y_clean, train_pred)
            
            self.metrics['train_mae'] = train_mae
            self.metrics['train_rmse'] = train_rmse
            self.metrics['train_r2'] = train_r2
            
            print(f"ARIMA model trained on {len(y_clean)} samples")
            print(f"Order: {self.order}, Seasonal: {self.seasonal_order}")
            print(f"Training R²: {train_r2:.4f}, MAE: {train_mae:.2f}")
            print(f"AIC: {self.metrics['aic']:.2f}, BIC: {self.metrics['bic']:.2f}")
            
        except Exception as e:
            print(f"ARIMA training failed: {e}")
            print("Falling back to simpler configuration...")
            
            # Try simpler model without seasonality
            self.model = SARIMAX(
                endog=y_clean,
                exog=X_clean if self.use_exog else None,
                order=(1, 0, 1),  # Simpler ARMA(1,1)
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fitted_model = self.model.fit(disp=False)
            
            self.is_trained = True
            print("Trained simpler ARMA(1,1) model instead")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ARIMA model."""
        if not self.is_trained or self.fitted_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        n_samples = len(X)
        
        if self.use_exog and self.feature_columns:
            X_prepared = self.prepare_features(X)
            
            # For ARIMA, we need to use forecast for out-of-sample predictions
            # This is a simplified approach - proper time series forecasting would
            # require maintaining temporal order
            try:
                # Get in-sample predictions for the training period
                in_sample_pred = self.fitted_model.fittedvalues
                
                # For new data, use the model's forecast capability
                # Note: This is a simplification - proper forecasting would need
                # to maintain temporal continuity
                forecast = self.fitted_model.forecast(
                    steps=n_samples,
                    exog=X_prepared if self.use_exog else None
                )
                
                return np.array(forecast)
                
            except:
                # Fallback: use mean prediction if forecasting fails
                mean_hr = self.train_endog.mean() if self.train_endog is not None else 130
                return np.full(n_samples, mean_hr)
        else:
            # Without exogenous variables, forecast based on history
            try:
                forecast = self.fitted_model.forecast(steps=n_samples)
                return np.array(forecast)
            except:
                mean_hr = self.train_endog.mean() if self.train_endog is not None else 130
                return np.full(n_samples, mean_hr)
    
    def get_model_info(self) -> dict:
        """Get detailed model information."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "model_type": "ARIMA",
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "use_exog": self.use_exog,
            "aic": self.metrics.get('aic', None),
            "bic": self.metrics.get('bic', None),
            "training_samples": self.metrics.get('training_samples', 0)
        }
        
        if self.fitted_model is not None:
            try:
                info['parameters'] = self.fitted_model.params.to_dict()
            except:
                pass
        
        return info
    
    def save(self, filepath: str) -> None:
        """Save the model to a file."""
        model_data = {
            'model': self.fitted_model,
            'original_model': self.model,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'use_exog': self.use_exog,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'train_endog': self.train_endog,
            'train_exog': self.train_exog,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"ARIMA model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the model from a file."""
        model_data = joblib.load(filepath)
        self.fitted_model = model_data['model']
        self.model = model_data.get('original_model')
        self.order = model_data['order']
        self.seasonal_order = model_data['seasonal_order']
        self.use_exog = model_data['use_exog']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']
        self.train_endog = model_data.get('train_endog')
        self.train_exog = model_data.get('train_exog')
        self.is_trained = model_data['is_trained']
        print(f"ARIMA model loaded from {filepath}")