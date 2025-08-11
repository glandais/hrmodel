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
        
        # Add lag features for HR (previous values)
        if 'hr' in df.columns:
            for lag in [1, 5, 10]:
                result_df[f'hr_lag_{lag}'] = result_df['hr'].shift(lag)
        
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