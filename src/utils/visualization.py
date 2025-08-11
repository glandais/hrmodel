import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class Visualizer:
    """Visualization utilities for model results."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Predictions vs Actual", 
                        save_path: Optional[Path] = None) -> None:
        """Plot predictions vs actual values."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', lw=2)
        axes[0].set_xlabel('Actual HR (bpm)')
        axes[0].set_ylabel('Predicted HR (bpm)')
        axes[0].set_title(f'{title} - Scatter Plot')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_pred - y_true
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residual (Predicted - Actual)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_time_series(self, df: pd.DataFrame, 
                        columns: List[str] = ['hr', 'predicted_hr'],
                        title: str = "Heart Rate Time Series",
                        save_path: Optional[Path] = None) -> None:
        """Plot time series data."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Use time column if available, otherwise use index
        x_axis = df['time'] if 'time' in df.columns else df.index
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, col in enumerate(columns):
            if col in df.columns:
                color = colors[i % len(colors)]
                alpha = 0.8 if col == 'hr' else 0.7
                linewidth = 1.5 if col == 'hr' else 1.2
                
                ax.plot(x_axis, df[col], 
                       label=col.replace('_', ' ').title(), 
                       alpha=alpha, 
                       color=color,
                       linewidth=linewidth)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Heart Rate (bpm)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for datetime
        if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_time_series_with_error(self, df: pd.DataFrame,
                                   title: str = "Heart Rate with Prediction Error",
                                   save_path: Optional[Path] = None) -> None:
        """Plot HR time series with error bands."""
        if 'hr' not in df.columns or 'predicted_hr' not in df.columns:
            print("Warning: Missing required columns for error plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        x_axis = df['time'] if 'time' in df.columns else df.index
        
        # Main plot - HR vs Predicted
        ax1.plot(x_axis, df['hr'], label='Actual HR', color='#1f77b4', linewidth=1.5, alpha=0.8)
        ax1.plot(x_axis, df['predicted_hr'], label='Predicted HR', color='#ff7f0e', linewidth=1.2, alpha=0.7)
        
        # Fill between for visual comparison
        ax1.fill_between(x_axis, df['hr'], df['predicted_hr'], 
                        alpha=0.3, color='lightgray', label='Error')
        
        ax1.set_ylabel('Heart Rate (bpm)')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error subplot
        if 'hr_error' in df.columns:
            error = df['hr_error']
        else:
            error = df['hr'] - df['predicted_hr']
            
        ax2.plot(x_axis, error, color='red', alpha=0.7, linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(x_axis, error, 0, alpha=0.3, color='red')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Error (bpm)')
        ax2.set_title('Prediction Error')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis for datetime
        if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_model_comparison(self, metrics_df: pd.DataFrame,
                            metric: str = 'mae',
                            title: str = "Model Comparison",
                            save_path: Optional[Path] = None) -> None:
        """Plot comparison of different models."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = metrics_df.index
        values = metrics_df[metric]
        
        bars = ax.bar(models, values)
        
        # Color best model differently
        best_idx = values.argmin() if metric in ['mae', 'rmse', 'mape'] else values.argmax()
        bars[best_idx].set_color('green')
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{title} - {metric.upper()}')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               top_n: int = 15,
                               title: str = "Feature Importance",
                               save_path: Optional[Path] = None) -> None:
        """Plot feature importance."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select top N features
        plot_df = importance_df.head(top_n)
        
        ax.barh(plot_df['feature'], plot_df['abs_coefficient'])
        ax.set_xlabel('Importance (Absolute Coefficient)')
        ax.set_ylabel('Feature')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis to have most important at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        else:
            plt.show()
    
    def create_report_plots(self, results: Dict, output_dir: Path) -> None:
        """Create all standard plots for a model report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots based on available data
        if 'predictions' in results:
            self.plot_predictions(
                results['predictions']['y_true'],
                results['predictions']['y_pred'],
                title=results.get('model_name', 'Model'),
                save_path=output_dir / 'predictions.png'
            )
        
        if 'metrics_comparison' in results:
            self.plot_model_comparison(
                results['metrics_comparison'],
                save_path=output_dir / 'model_comparison.png'
            )
        
        if 'feature_importance' in results:
            self.plot_feature_importance(
                results['feature_importance'],
                save_path=output_dir / 'feature_importance.png'
            )
        
        print(f"Plots saved to {output_dir}")