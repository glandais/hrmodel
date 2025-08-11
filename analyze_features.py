#!/usr/bin/env python3
"""
Feature Analysis Script - Analyze correlations and feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.gpx_parser import GPXParser
from src.data.feature_engineering import FeatureEngineer

def analyze_features():
    """Analyze feature correlations and distributions."""
    
    # Load one representative file for analysis
    parser = GPXParser()
    engineer = FeatureEngineer(windows=[5, 10, 30, 60])
    
    # Get processed data directory
    processed_dir = Path("output/processed_data")
    if not processed_dir.exists():
        print("No processed data found. Please run main.py first.")
        return
    
    # Load first processed file
    files = list(processed_dir.glob("*.parquet"))
    if not files:
        print("No parquet files found in processed_data.")
        return
    
    print(f"Analyzing {len(files)} processed files...")
    
    # Load all files and combine
    dfs = []
    for file in files:
        df = pd.read_parquet(file)
        df['source'] = file.stem
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} rows, {combined_df.shape[1]} columns")
    
    # Focus on numeric columns for analysis
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove time-related columns
    feature_cols = [col for col in numeric_cols if not col.startswith('time')]
    
    print(f"Analyzing {len(feature_cols)} numeric features")
    
    # 1. Basic statistics
    print("\n=== BASIC STATISTICS ===")
    stats = combined_df[feature_cols].describe()
    print(stats.round(2))
    
    # 2. Correlation with target (HR)
    print("\n=== CORRELATION WITH HR ===")
    if 'hr' in combined_df.columns:
        hr_corr = combined_df[feature_cols].corrwith(combined_df['hr']).sort_values(key=abs, ascending=False)
        print("Top 15 features correlated with HR:")
        print(hr_corr.head(15).round(4))
        
        # Save correlation plot
        plt.figure(figsize=(10, 8))
        hr_corr.head(15).plot(kind='barh')
        plt.title('Feature Correlation with Heart Rate')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.savefig('output/feature_hr_correlation.png', dpi=150, bbox_inches='tight')
        print("Saved correlation plot: output/feature_hr_correlation.png")
        plt.close()
    
    # 3. Feature correlation matrix (top features only)
    print("\n=== FEATURE CORRELATION MATRIX ===")
    if 'hr' in combined_df.columns:
        top_features = hr_corr.head(10).index.tolist()
        if 'hr' not in top_features:
            top_features.append('hr')
        
        corr_matrix = combined_df[top_features].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('Top Features Correlation Matrix')
        plt.tight_layout()
        plt.savefig('output/feature_correlation_matrix.png', dpi=150, bbox_inches='tight')
        print("Saved correlation matrix: output/feature_correlation_matrix.png")
        plt.close()
        
        # Identify highly correlated features (multicollinearity)
        print("\nHighly correlated feature pairs (|r| > 0.8):")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        for feat1, feat2, corr_val in high_corr_pairs:
            print(f"  {feat1} <-> {feat2}: {corr_val:.4f}")
    
    # 4. Missing values analysis
    print("\n=== MISSING VALUES ===")
    missing = combined_df[feature_cols].isnull().sum()
    missing_pct = (missing / len(combined_df) * 100).round(2)
    missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
    missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)
    
    if len(missing_df) > 0:
        print("Features with missing values:")
        print(missing_df)
    else:
        print("No missing values found.")
    
    # 5. Feature importance analysis from current models
    print("\n=== MODEL FEATURE IMPORTANCE ===")
    model_dirs = ['output/ols', 'output/ridge', 'output/elastic_net']
    
    for model_dir in model_dirs:
        model_path = Path(model_dir)
        if not model_path.exists():
            continue
            
        model_name = model_path.name
        print(f"\n{model_name.upper()} Model:")
        
        try:
            # Try to load and get feature importance
            import joblib
            model_file = model_path / f'{model_name}_model.joblib'
            if model_file.exists():
                model_data = joblib.load(model_file)
                if hasattr(model_data, 'get_feature_importance'):
                    importance = model_data.get_feature_importance()
                    if importance is not None:
                        print("Top 10 important features:")
                        print(importance.head(10)[['feature', 'abs_coefficient' if 'abs_coefficient' in importance.columns else 'importance']].to_string(index=False))
        except Exception as e:
            print(f"Could not load {model_name} model: {e}")
    
    # 6. Recommendations
    print("\n=== RECOMMENDATIONS ===")
    recommendations = []
    
    if 'hr' in combined_df.columns:
        # Check if we have HR lag features
        hr_lag_features = [col for col in feature_cols if 'hr' in col.lower() and ('lag' in col.lower() or 'shift' in col.lower())]
        if len(hr_lag_features) == 0:
            recommendations.append("ADD: HR lag features (hr_lag_15s, hr_lag_30s) - physiological HR response delay")
        
        # Check for power intensity features
        power_intensity_features = [col for col in feature_cols if 'power' in col.lower() and ('zone' in col.lower() or 'relative' in col.lower() or 'threshold' in col.lower())]
        if len(power_intensity_features) == 0:
            recommendations.append("ADD: Power intensity features (power zones, relative to threshold)")
        
        # Check for gradient features
        gradient_features = [col for col in feature_cols if 'gradient' in col.lower() or ('ele' in col.lower() and 'rate' in col.lower())]
        if len(gradient_features) == 0:
            recommendations.append("ADD: Gradient features (elevation change rate)")
        
        # Check for variability features
        variability_features = [col for col in feature_cols if 'std' in col.lower() or 'var' in col.lower()]
        if len(variability_features) == 0:
            recommendations.append("ADD: Variability features (rolling std of power, cadence)")
    
    # Model recommendations based on current performance
    recommendations.extend([
        "TRY: Random Forest - handles feature interactions well",
        "TRY: XGBoost - superior gradient boosting performance",
        "TRY: Polynomial features - capture non-linear power/HR relationship",
        "TRY: LSTM - capture temporal dependencies in HR response"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\nAnalysis complete. Results saved to output/")

if __name__ == "__main__":
    analyze_features()