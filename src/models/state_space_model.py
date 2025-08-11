import pandas as pd
import numpy as np
import warnings
from .base_model import BaseModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

try:
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
    from statsmodels.tsa.statespace.mlemodel import MLEModel
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available. Install with: pip install statsmodels")


class StateSpaceModel(BaseModel):
    """State Space Model using Kalman filtering for HR prediction.
    
    State Space Models represent time series as a system with:
    - Hidden states that evolve over time
    - Observations that depend on hidden states
    - Kalman filter for optimal state estimation
    
    This is particularly suitable for HR prediction as heart rate has
    internal states (fitness, fatigue) that aren't directly observable.
    """
    
    def __init__(self, feature_columns=None,
                 level=True, trend=True, seasonal=None,
                 use_exog=True, normalize=False):
        """Initialize State Space Model.
        
        Args:
            feature_columns: List of feature column names
            level: Include level component (local level model)
            trend: Include trend component
            seasonal: Seasonal period (None for no seasonality)
            use_exog: Whether to use exogenous variables
            normalize: Whether to normalize features
        """
        super().__init__("StateSpace", feature_columns)
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels is not available. Install with: pip install statsmodels")
        
        self.level = level
        self.trend = trend
        self.seasonal = seasonal
        self.use_exog = use_exog
        self.normalize = normalize
        
        self.model = None
        self.fitted_model = None
        
        # Store training data for forecasting
        self.train_endog = None
        self.train_exog = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the State Space Model."""
        # Store original data
        self.train_endog = y.copy()
        
        # Prepare exogenous variables if requested
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
        
        try:
            # Create Unobserved Components model (structural time series)
            # This provides a flexible framework for state space modeling
            
            # Determine model specification
            if self.level and not self.trend:
                model_type = 'local level'  # Random walk
            elif self.level and self.trend:
                model_type = 'local linear trend'  # Local linear trend
            else:
                model_type = 'irregular'  # Just noise
            
            # Create the model
            self.model = UnobservedComponents(
                endog=y_clean,
                level=model_type,
                seasonal=self.seasonal,
                exog=X_clean if self.use_exog else None
            )
            
            # Fit the model using maximum likelihood
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fitted_model = self.model.fit(
                    disp=False,
                    maxiter=100,
                    method='powell'  # More robust than default
                )
            
            self.is_trained = True
            
            # Get filtered states (Kalman filter output)
            filtered_states = self.fitted_model.filtered_state
            
            # Get in-sample predictions
            train_pred = self.fitted_model.fittedvalues
            
            # Calculate metrics
            self.metrics['training_samples'] = len(y_clean)
            self.metrics['n_features'] = X_clean.shape[1] if X_clean is not None else 0
            self.metrics['loglikelihood'] = self.fitted_model.llf
            self.metrics['aic'] = self.fitted_model.aic
            self.metrics['bic'] = self.fitted_model.bic
            
            # Training performance metrics
            self.metrics['train_mae'] = mean_absolute_error(y_clean, train_pred)
            self.metrics['train_rmse'] = np.sqrt(mean_squared_error(y_clean, train_pred))
            self.metrics['train_r2'] = r2_score(y_clean, train_pred)
            
            print(f"State Space model trained on {len(y_clean)} samples")
            print(f"Model type: {model_type}")
            print(f"Training R²: {self.metrics['train_r2']:.4f}, MAE: {self.metrics['train_mae']:.2f}")
            print(f"Log-likelihood: {self.metrics['loglikelihood']:.2f}")
            
        except Exception as e:
            print(f"State Space model training failed: {e}")
            print("Falling back to simpler local level model...")
            
            # Try simpler model without trend or seasonality
            try:
                self.model = UnobservedComponents(
                    endog=y_clean,
                    level='local level',
                    exog=X_clean if self.use_exog else None
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.fitted_model = self.model.fit(disp=False)
                
                self.is_trained = True
                print("Trained simpler local level model instead")
                
                # Update metrics for simpler model
                train_pred = self.fitted_model.fittedvalues
                self.metrics['train_mae'] = mean_absolute_error(y_clean, train_pred)
                self.metrics['train_rmse'] = np.sqrt(mean_squared_error(y_clean, train_pred))
                self.metrics['train_r2'] = r2_score(y_clean, train_pred)
                
            except Exception as e2:
                print(f"Even simple model failed: {e2}")
                self.is_trained = False
                raise ValueError("Could not train State Space model")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the State Space Model."""
        if not self.is_trained or self.fitted_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        n_samples = len(X)
        
        try:
            if self.use_exog and self.feature_columns:
                X_prepared = self.prepare_features(X)
                
                # Forecast with exogenous variables
                forecast = self.fitted_model.forecast(
                    steps=n_samples,
                    exog=X_prepared if self.use_exog else None
                )
            else:
                # Forecast without exogenous variables
                forecast = self.fitted_model.forecast(steps=n_samples)
            
            # Convert to numpy array and clip to reasonable HR range
            predictions = np.array(forecast)
            predictions = np.clip(predictions, 40, 220)
            
            return predictions
            
        except Exception as e:
            print(f"State Space prediction failed: {e}")
            # Return mean HR as fallback
            mean_hr = self.train_endog.mean() if self.train_endog is not None else 130
            return np.full(n_samples, mean_hr)
    
    def get_smoothed_states(self) -> pd.DataFrame:
        """Get smoothed states from Kalman smoother."""
        if not self.is_trained or self.fitted_model is None:
            return None
        
        try:
            # Get smoothed states (Kalman smoother output)
            smoothed_states = self.fitted_model.smoothed_state
            
            # Convert to DataFrame for easier interpretation
            state_names = self.fitted_model.model.state_names
            states_df = pd.DataFrame(
                smoothed_states.T,
                columns=state_names
            )
            
            return states_df
            
        except:
            return None
    
    def get_model_info(self) -> dict:
        """Get detailed model information."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "model_type": "StateSpace",
            "level": self.level,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "use_exog": self.use_exog,
            "loglikelihood": self.metrics.get('loglikelihood', None),
            "aic": self.metrics.get('aic', None),
            "bic": self.metrics.get('bic', None),
            "training_samples": self.metrics.get('training_samples', 0)
        }
        
        if self.fitted_model is not None:
            try:
                info['n_states'] = self.fitted_model.model.k_states
                info['state_names'] = self.fitted_model.model.state_names
            except:
                pass
        
        return info
    
    def plot_diagnostics(self):
        """Plot model diagnostics."""
        if not self.is_trained or self.fitted_model is None:
            raise ValueError("Model must be trained before plotting")
        
        try:
            import matplotlib.pyplot as plt
            
            # Create diagnostic plots
            fig = self.fitted_model.plot_diagnostics(figsize=(12, 8))
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not create diagnostic plots: {e}")
    
    def save(self, filepath: str) -> None:
        """Save the model to a file."""
        model_data = {
            'model': self.fitted_model,
            'original_model': self.model,
            'level': self.level,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'use_exog': self.use_exog,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'train_endog': self.train_endog,
            'train_exog': self.train_exog,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"State Space model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the model from a file."""
        model_data = joblib.load(filepath)
        self.fitted_model = model_data['model']
        self.model = model_data.get('original_model')
        self.level = model_data['level']
        self.trend = model_data['trend']
        self.seasonal = model_data['seasonal']
        self.use_exog = model_data['use_exog']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']
        self.train_endog = model_data.get('train_endog')
        self.train_exog = model_data.get('train_exog')
        self.is_trained = model_data['is_trained']
        print(f"State Space model loaded from {filepath}")