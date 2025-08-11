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
        
        # Advanced physiological features (NO HR USED)
        if 'power' in df.columns:
            power_series = df['power'].fillna(0)
            
            # Physiological response modeling - effort patterns that predict HR response
            # Weighted effort history (mimics HR physiological delay response)
            result_df['effort_response_15s'] = (0.6 * power_series.shift(15) + 
                                              0.3 * power_series.shift(30) + 
                                              0.1 * power_series.shift(45)).fillna(0)
            
            # Effort ramp rate (sudden changes in power demand)
            power_diff = power_series.diff().fillna(0)
            result_df['effort_ramp_rate_30s'] = power_diff.rolling(30, min_periods=1).mean()
            result_df['effort_ramp_rate_60s'] = power_diff.rolling(60, min_periods=1).mean()
            
            # Power variability (neuromuscular fatigue indicators)
            for window in [30, 60, 120]:
                power_window = power_series.rolling(window, min_periods=1)
                result_df[f'power_variability_{window}s'] = power_window.std().fillna(0)
                result_df[f'power_cv_{window}s'] = (power_window.std() / (power_window.mean() + 1e-6)).fillna(0)
            
            # Critical Power estimates (3min, 5min, 20min efforts)
            for window_min, window_samples in [(3, 180), (5, 300), (20, 1200)]:
                cp_window = power_series.rolling(window_samples, min_periods=30)
                result_df[f'critical_power_{window_min}min'] = cp_window.mean().fillna(0)
            
            # Functional Threshold Power proxy (estimated from data)
            ftp_proxy = power_series.quantile(0.85)  # Rough FTP estimate from current ride
            result_df['ftp_proxy'] = ftp_proxy
            
            # Power zones based on FTP proxy (NO HR zones used)
            result_df['power_zone_1'] = (power_series <= 0.55 * ftp_proxy).astype(int)  # Recovery
            result_df['power_zone_2'] = ((power_series > 0.55 * ftp_proxy) & (power_series <= 0.75 * ftp_proxy)).astype(int)  # Endurance
            result_df['power_zone_3'] = ((power_series > 0.75 * ftp_proxy) & (power_series <= 0.9 * ftp_proxy)).astype(int)  # Tempo
            result_df['power_zone_4'] = ((power_series > 0.9 * ftp_proxy) & (power_series <= 1.05 * ftp_proxy)).astype(int)  # Threshold
            result_df['power_zone_5'] = (power_series > 1.05 * ftp_proxy).astype(int)  # VO2max+
            
            # Anaerobic contribution (power above threshold)
            result_df['anaerobic_reserve'] = (power_series - ftp_proxy).clip(0, None)
            
            # Fatigue modeling features
            if 'time_elapsed' in result_df.columns:
                time_elapsed = result_df['time_elapsed']
                
                # Glycogen depletion proxy (cumulative work over time)
                cumulative_work_kj = (power_series * time_elapsed.diff()).cumsum() / 1000
                result_df['fatigue_index'] = (cumulative_work_kj / (time_elapsed + 1)).fillna(0)
                
                # Intensity factor over time
                normalized_power_30min = power_series.rolling(1800, min_periods=30).mean()
                result_df['intensity_factor'] = (power_series / (normalized_power_30min + 1e-6)).fillna(1)
                
                # Time in power zones (as percentages)
                for zone in range(1, 6):
                    zone_time = result_df[f'power_zone_{zone}'].rolling(600, min_periods=1).sum()  # 10min windows
                    result_df[f'time_in_zone_{zone}_pct'] = (zone_time / 600 * 100).fillna(0)
        
        # Advanced cadence features
        if 'cad' in df.columns:
            cad_series = df['cad'].fillna(0)
            
            # Cadence efficiency patterns
            result_df['cad_optimal_range'] = ((cad_series >= 80) & (cad_series <= 100)).astype(int)
            result_df['cad_suboptimal_low'] = (cad_series < 70).astype(int)
            result_df['cad_suboptimal_high'] = (cad_series > 110).astype(int)
            
            # Cadence variability (pedaling smoothness)
            for window in [30, 60, 120]:
                cad_window = cad_series.rolling(window, min_periods=1)
                result_df[f'cad_smoothness_{window}s'] = 1 / (cad_window.std() + 1e-6)  # Inverse of std
                
            # Cadence change rate
            cad_diff = cad_series.diff().fillna(0)
            result_df['cad_change_rate'] = cad_diff.rolling(30, min_periods=1).mean()
        
        # Power-Cadence interaction features (pedaling efficiency)
        if 'power' in df.columns and 'cad' in df.columns:
            power_series = df['power'].fillna(0)
            cad_series = df['cad'].fillna(0)
            
            # Torque proxy (Power/Cadence relationship)
            result_df['torque_proxy'] = power_series / (cad_series + 1e-6)
            result_df['torque_proxy'] = result_df['torque_proxy'].fillna(0)
            
            # Pedaling efficiency at different power levels
            for power_threshold in [100, 200, 300]:
                mask = power_series >= power_threshold
                if mask.sum() > 0:
                    avg_cad_at_power = cad_series[mask].mean()
                    result_df[f'cad_efficiency_{power_threshold}w'] = (cad_series / max(avg_cad_at_power, 1)).fillna(1)
                else:
                    result_df[f'cad_efficiency_{power_threshold}w'] = 1.0
        
        # Environmental/terrain features
        if 'ele' in df.columns:
            ele_series = df['ele'].fillna(0)
            
            # Gradient features (already implemented, enhance them)
            ele_diff = ele_series.diff().fillna(0)
            
            # Gradient categories
            result_df['flat_terrain'] = (abs(ele_diff) <= 2).astype(int)  # <2m elevation change
            result_df['uphill_terrain'] = (ele_diff > 2).astype(int)
            result_df['downhill_terrain'] = (ele_diff < -2).astype(int)
            
            # Climbing metrics
            result_df['cumulative_elevation_gain'] = ele_diff.clip(0, None).cumsum()
            result_df['climbing_rate'] = ele_diff.rolling(60, min_periods=1).sum()  # m/min climbing
            
            # VAM proxy (Vertical Ascent Meters per hour)
            if 'time_elapsed' in result_df.columns:
                time_diff_hours = result_df['time_elapsed'].diff() / 3600
                result_df['vam_proxy'] = (ele_diff / (time_diff_hours + 1e-6)).fillna(0)
        
        # Contextual features (workout structure)
        if 'time_elapsed' in result_df.columns:
            time_elapsed = result_df['time_elapsed']
            
            # Time-based features
            if 'time' in df.columns:
                result_df['hour_of_day'] = df['time'].dt.hour
                result_df['minute_of_hour'] = df['time'].dt.minute
                
                # Circadian rhythm factors (simplified)
                hour = df['time'].dt.hour
                result_df['circadian_factor'] = np.sin(2 * np.pi * hour / 24)  # Peak around noon
            
            # Workout progression
            total_duration = time_elapsed.max()
            result_df['workout_progress'] = (time_elapsed / max(total_duration, 1)).fillna(0)
            
            # Early/late workout indicators
            result_df['workout_start_phase'] = (time_elapsed <= total_duration * 0.2).astype(int)
            result_df['workout_end_phase'] = (time_elapsed >= total_duration * 0.8).astype(int)
        
        return result_df

    def process_dataframe(self, df: pd.DataFrame, 
                         add_ma: bool = True,
                         add_derived: bool = True) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        result = df.copy()
        
        if add_ma:
            result = self.add_moving_averages(result)
        
        if add_derived:
            result = self.add_derived_features(result)

        return result