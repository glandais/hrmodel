import pandas as pd
import numpy as np
import warnings
from .base_model import BaseModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False
        warnings.warn("Prophet not available. Install with: pip install prophet")
        # Create dummy class when prophet is not available
        class Prophet:
            pass


class ProphetModel(BaseModel):
    """Facebook Prophet model for HR prediction.
    
    Prophet is a time series forecasting model developed by Facebook that works well with:
    - Strong seasonal patterns
    - Missing data and outliers
    - Trend changes
    - Additional regressors (like power and cadence)
    """
    
    def __init__(self, feature_columns=None,
                 yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True,
                 changepoint_prior_scale=0.05, seasonality_prior_scale=10.0,
                 normalize=False):
        """Initialize Prophet model.
        
        Args:
            feature_columns: List of feature column names to use as regressors
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality  
            daily_seasonality: Include daily seasonality (for within-day patterns)
            changepoint_prior_scale: Flexibility of trend changes (higher = more flexible)
            seasonality_prior_scale: Flexibility of seasonality (higher = more flexible)
            normalize: Whether to normalize features
        """
        super().__init__("Prophet", feature_columns)
        
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install with: pip install prophet")
        
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.normalize = normalize
        
        self.model = None
        self.regressor_columns = []
        
        # Store training data characteristics
        self.training_start_time = None
        self.training_end_time = None
        self.time_step = None
    
    def _prepare_prophet_data(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Prepare data in Prophet's expected format."""
        # Prophet expects a DataFrame with 'ds' (datetime) and 'y' (target) columns
        df = pd.DataFrame()
        
        # Create synthetic timestamps if not available
        # Assume 1-second intervals for cycling data
        if 'time' in X.columns:
            df['ds'] = pd.to_datetime(X['time'])
        else:
            # Create synthetic timestamps
            start_time = datetime(2024, 1, 1, 12, 0, 0)
            df['ds'] = [start_time + timedelta(seconds=i) for i in range(len(y))]
        
        df['y'] = y.values
        
        # Add regressors (external features)
        if self.feature_columns:
            X_prepared = self.prepare_features(X)
            for col in X_prepared.columns:
                if col in self.feature_columns:
                    df[col] = X_prepared[col].values
                    if col not in self.regressor_columns:
                        self.regressor_columns.append(col)
        
        # Remove rows with NaN in target
        df = df.dropna(subset=['y'])
        
        return df
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Prophet model."""
        # Prepare data
        train_df = self._prepare_prophet_data(X, y)
        
        if len(train_df) < 2:
            raise ValueError("Not enough valid data points for Prophet training")
        
        # Store training time characteristics
        self.training_start_time = train_df['ds'].min()
        self.training_end_time = train_df['ds'].max()
        if len(train_df) > 1:
            self.time_step = (train_df['ds'].iloc[1] - train_df['ds'].iloc[0]).total_seconds()
        else:
            self.time_step = 1.0
        
        # Initialize Prophet model
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            interval_width=0.95,
            uncertainty_samples=100
        )
        
        # Add custom seasonalities for cycling patterns
        # Add hourly seasonality for within-ride patterns
        self.model.add_seasonality(
            name='hourly',
            period=1/24,  # 1 hour in days
            fourier_order=3
        )
        
        # Add regressors
        for col in self.regressor_columns:
            self.model.add_regressor(col)
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(train_df)
        
        self.is_trained = True
        
        # Get in-sample predictions for metrics
        train_forecast = self.model.predict(train_df)
        train_pred = train_forecast['yhat'].values
        train_y = train_df['y'].values
        
        # Calculate metrics
        self.metrics['training_samples'] = len(train_df)
        self.metrics['n_regressors'] = len(self.regressor_columns)
        self.metrics['train_mae'] = mean_absolute_error(train_y, train_pred)
        self.metrics['train_rmse'] = np.sqrt(mean_squared_error(train_y, train_pred))
        self.metrics['train_r2'] = r2_score(train_y, train_pred)
        
        print(f"Prophet model trained on {len(train_df)} samples")
        print(f"Regressors: {len(self.regressor_columns)}")
        print(f"Training R²: {self.metrics['train_r2']:.4f}, MAE: {self.metrics['train_mae']:.2f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the Prophet model."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare future dataframe
        future_df = pd.DataFrame()
        
        # Create timestamps for prediction
        if 'time' in X.columns:
            future_df['ds'] = pd.to_datetime(X['time'])
        else:
            # Create synthetic timestamps continuing from training
            start_time = self.training_end_time + timedelta(seconds=self.time_step)
            future_df['ds'] = [start_time + timedelta(seconds=i * self.time_step) 
                              for i in range(len(X))]
        
        # Add regressors
        if self.feature_columns:
            X_prepared = self.prepare_features(X)
            for col in self.regressor_columns:
                if col in X_prepared.columns:
                    future_df[col] = X_prepared[col].values
                else:
                    # Use mean value if column missing
                    future_df[col] = 0
        
        # Make predictions
        try:
            forecast = self.model.predict(future_df)
            predictions = forecast['yhat'].values
            
            # Clip predictions to reasonable HR range
            predictions = np.clip(predictions, 40, 220)
            
            return predictions
            
        except Exception as e:
            print(f"Prophet prediction failed: {e}")
            # Return mean HR as fallback
            return np.full(len(X), 130)
    
    def get_model_info(self) -> dict:
        """Get detailed model information."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "model_type": "Prophet",
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "n_regressors": len(self.regressor_columns),
            "regressor_columns": self.regressor_columns,
            "training_samples": self.metrics.get('training_samples', 0)
        }
        
        return info
    
    def plot_components(self):
        """Plot the forecast components (trend, seasonality, regressors)."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before plotting")
        
        try:
            from prophet.plot import plot_components_plotly
            import plotly.io as pio
            
            # Create a sample forecast for visualization
            future = self.model.make_future_dataframe(periods=0)
            forecast = self.model.predict(future)
            
            # Plot components
            fig = plot_components_plotly(self.model, forecast)
            pio.show(fig)
            
        except ImportError:
            print("Plotly required for component plotting. Install with: pip install plotly")
    
    def save(self, filepath: str) -> None:
        """Save the model to a file."""
        model_data = {
            'model': self.model,
            'regressor_columns': self.regressor_columns,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'training_start_time': self.training_start_time,
            'training_end_time': self.training_end_time,
            'time_step': self.time_step,
            'is_trained': self.is_trained,
            'config': {
                'yearly_seasonality': self.yearly_seasonality,
                'weekly_seasonality': self.weekly_seasonality,
                'daily_seasonality': self.daily_seasonality,
                'changepoint_prior_scale': self.changepoint_prior_scale,
                'seasonality_prior_scale': self.seasonality_prior_scale
            }
        }
        joblib.dump(model_data, filepath)
        print(f"Prophet model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the model from a file."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.regressor_columns = model_data['regressor_columns']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']
        self.training_start_time = model_data.get('training_start_time')
        self.training_end_time = model_data.get('training_end_time')
        self.time_step = model_data.get('time_step', 1.0)
        self.is_trained = model_data['is_trained']
        
        config = model_data.get('config', {})
        self.yearly_seasonality = config.get('yearly_seasonality', False)
        self.weekly_seasonality = config.get('weekly_seasonality', False)
        self.daily_seasonality = config.get('daily_seasonality', True)
        self.changepoint_prior_scale = config.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = config.get('seasonality_prior_scale', 10.0)
        
        print(f"Prophet model loaded from {filepath}")