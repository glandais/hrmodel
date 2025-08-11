import numpy as np
import pandas as pd
from typing import Dict, Tuple


class MetricsCalculator:
    """Calculate regression metrics."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all regression metrics."""
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {
                'mae': np.nan,
                'rmse': np.nan,
                'mape': np.nan,
                'r2': np.nan,
                'correlation': np.nan,
                'bias': np.nan,
                'n_samples': 0
            }
        
        # MAE
        mae = np.mean(np.abs(y_true_clean - y_pred_clean))
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
        
        # MAPE
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
        
        # R-squared
        ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
        ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Correlation
        if len(y_true_clean) > 1:
            correlation = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
        else:
            correlation = 0
        
        # Bias
        bias = np.mean(y_pred_clean - y_true_clean)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'correlation': correlation,
            'bias': bias,
            'n_samples': len(y_true_clean)
        }
    
    @staticmethod
    def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare multiple model results."""
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.round(4)
        df = df.sort_values('mae')
        return df
    
    @staticmethod
    def calculate_confidence_intervals(y_true: np.ndarray, y_pred: np.ndarray, 
                                      confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence intervals for predictions."""
        errors = y_pred - y_true
        errors_clean = errors[~np.isnan(errors)]
        
        if len(errors_clean) == 0:
            return (np.nan, np.nan)
        
        mean_error = np.mean(errors_clean)
        std_error = np.std(errors_clean)
        
        # Calculate confidence interval
        z_score = 1.96 if confidence == 0.95 else 2.58  # 95% or 99% CI
        margin = z_score * std_error
        
        return (mean_error - margin, mean_error + margin)