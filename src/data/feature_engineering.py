import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """Feature engineering for cycling data."""
    
    def __init__(self, windows: List[int] = [5, 10, 30, 60]):
        self.windows = windows
    
    def add_moving_averages(self, df: pd.DataFrame, columns: List[str] = ['hr', 'cad', 'power']) -> pd.DataFrame:
        """Add moving averages for specified columns."""
        result_df = df.copy()
        
        # Ensure time index for rolling windows
        if 'time' in df.columns:
            df_indexed = df.set_index('time').sort_index()
        else:
            print("Warning: No 'time' column found")
            return result_df
        
        for column in columns:
            if column not in df.columns:
                continue
                
            for window in self.windows:
                column_name = f"{column}{window}"
                window_str = f"{window}s"
                
                try:
                    result_df[column_name] = df_indexed[column].rolling(
                        window=window_str,
                        min_periods=1
                    ).mean().values
                except Exception as e:
                    print(f"Error creating {column_name}: {e}")
                    result_df[column_name] = np.nan
        
        return result_df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features like gradients, deltas, etc."""
        result_df = df.copy()
        
        # Add time-based features
        if 'time' in df.columns:
            result_df['hour'] = df['time'].dt.hour
            result_df['minute'] = df['time'].dt.minute
            
            # Calculate time differences for rate calculations
            time_diff = df['time'].diff().dt.total_seconds().fillna(1)
        
        # Gradient features (elevation change rate)
        if 'ele' in df.columns:
            ele_diff = df['ele'].diff().fillna(0)
            result_df['gradient'] = ele_diff / time_diff if 'time' in df.columns else ele_diff
            result_df['gradient_abs'] = np.abs(result_df['gradient'])
            
            # Gradient moving averages
            for window in [10, 30]:
                result_df[f'gradient_ma{window}'] = result_df['gradient'].rolling(window, min_periods=1).mean()
        
        # Power intensity features
        if 'power' in df.columns:
            power_series = df['power'].fillna(0)
            
            # Power variability (coefficient of variation in rolling windows)
            for window in [10, 30, 60]:
                power_window = power_series.rolling(window, min_periods=1)
                mean_power = power_window.mean()
                std_power = power_window.std()
                result_df[f'power_cv{window}'] = (std_power / (mean_power + 1e-6)).fillna(0)
            
            # Power zones (approximate - could be personalized)
            # Assumes FTP around 250W for average cyclist
            ftp_estimate = power_series.quantile(0.85)  # Rough FTP estimate
            result_df['power_zone'] = pd.cut(power_series, 
                bins=[0, 0.55*ftp_estimate, 0.75*ftp_estimate, 0.9*ftp_estimate, 1.05*ftp_estimate, float('inf')],
                labels=[1, 2, 3, 4, 5], include_lowest=True).cat.codes
            
            # Normalized power (relative to recent average)
            result_df['power_normalized'] = power_series / (power_series.rolling(300, min_periods=30).mean() + 1e-6)
            
            # Power above threshold indicators
            threshold_75 = power_series.quantile(0.75)
            threshold_90 = power_series.quantile(0.90)
            result_df['power_above_75pct'] = (power_series > threshold_75).astype(int)
            result_df['power_above_90pct'] = (power_series > threshold_90).astype(int)
        
        # Cadence variability features
        if 'cad' in df.columns:
            cad_series = df['cad'].fillna(0)
            
            # Cadence variability
            for window in [10, 30, 60]:
                cad_window = cad_series.rolling(window, min_periods=1)
                result_df[f'cad_std{window}'] = cad_window.std().fillna(0)
                
            # Cadence efficiency indicators
            result_df['cad_optimal'] = ((cad_series >= 80) & (cad_series <= 100)).astype(int)
            result_df['cad_too_low'] = (cad_series < 70).astype(int)
            result_df['cad_too_high'] = (cad_series > 110).astype(int)
        
        # Interaction features
        if 'power' in df.columns and 'cad' in df.columns:
            # Power per RPM (efficiency measure)
            result_df['power_per_cad'] = df['power'] / (df['cad'] + 1e-6)
            result_df['power_per_cad'] = result_df['power_per_cad'].fillna(0)
        
        # Cumulative work features (fatigue indicators) - using integer windows
        if 'power' in df.columns:
            power_series = df['power'].fillna(0)
            
            # Convert time windows to approximate sample windows (assuming ~1Hz sampling)
            # 5min = 300 samples, 10min = 600 samples, 30min = 1800 samples
            for window_samples, window_name in [(300, 5), (600, 10), (1800, 30)]:
                work_window = power_series.rolling(window_samples, min_periods=1).sum()
                result_df[f'cumulative_work_{window_name}min'] = work_window
                
            # Work above threshold (intensity factor)  
            if 'power_above_75pct' in result_df.columns:
                for window_samples, window_name in [(300, 5), (600, 10)]:
                    power_above = power_series * result_df['power_above_75pct']
                    intensity_work = power_above.rolling(window_samples, min_periods=1).sum()
                    result_df[f'intense_work_{window_name}min'] = intensity_work
            
            # Time since start (in seconds)
            result_df['time_elapsed'] = (df['time'] - df['time'].min()).dt.total_seconds()
        
        # Add elevation gradient
        if 'ele' in df.columns and 'time_elapsed' in result_df.columns:
            result_df['ele_gradient'] = result_df['ele'].diff() / result_df['time_elapsed'].diff()
            result_df['ele_gradient'] = result_df['ele_gradient'].fillna(0)
        
        # Add power/cadence ratio
        if 'power' in df.columns and 'cad' in df.columns:
            result_df['power_per_cad'] = result_df['power'] / (result_df['cad'] + 1)  # Add 1 to avoid division by zero
        
        # Add cumulative features
        if 'power' in df.columns and 'time_elapsed' in result_df.columns:
            # Cumulative work (kilojoules)
            result_df['cumulative_work'] = (result_df['power'] * result_df['time_elapsed'].diff()).cumsum() / 1000
            result_df['cumulative_work'] = result_df['cumulative_work'].fillna(0)
        
        # Add lag features for power and cadence (effort history - valid predictors)
        # These represent recent effort that should influence current HR
        if 'power' in df.columns:
            for lag in [1, 5, 10, 15, 30]:  # 1-30 seconds ago
                result_df[f'power_lag_{lag}'] = df['power'].shift(lag)
        
        if 'cad' in df.columns:
            for lag in [1, 5, 10, 15, 30]:  # 1-30 seconds ago  
                result_df[f'cad_lag_{lag}'] = df['cad'].shift(lag)
        
        # NOTE: HR lag features removed to prevent data leakage
        # In real prediction, we don't have current/future HR values
        
        return result_df
    
    def add_hr_zones(self, df: pd.DataFrame, max_hr: int = 180) -> pd.DataFrame:
        """Add HR zone information."""
        result_df = df.copy()
        
        if 'hr' not in df.columns:
            return result_df
        
        # Define HR zones (as percentage of max HR)
        zones = {
            'zone1': (0.5, 0.6),   # Recovery
            'zone2': (0.6, 0.7),   # Aerobic
            'zone3': (0.7, 0.8),   # Threshold
            'zone4': (0.8, 0.9),   # VO2 Max
            'zone5': (0.9, 1.0),   # Anaerobic
        }
        
        for zone_name, (low, high) in zones.items():
            result_df[f'hr_{zone_name}'] = (
                (result_df['hr'] >= low * max_hr) & 
                (result_df['hr'] < high * max_hr)
            ).astype(int)
        
        # Time in each zone (cumulative)
        for zone_name in zones.keys():
            result_df[f'time_in_{zone_name}'] = result_df[f'hr_{zone_name}'].cumsum()
        
        return result_df
    
    def process_dataframe(self, df: pd.DataFrame, 
                         add_ma: bool = True,
                         add_derived: bool = True,
                         add_zones: bool = False) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        result = df.copy()
        
        if add_ma:
            result = self.add_moving_averages(result)
        
        if add_derived:
            result = self.add_derived_features(result)
        
        if add_zones:
            result = self.add_hr_zones(result)
        
        return result