#!/usr/bin/env python3
"""
MLP Hyperparameter Tuning Script

This script runs isolated hyperparameter tuning for the MLP model
to find optimal parameters for heart rate prediction.

Target: Beat Random Forest's 1.17 bpm MAE
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.gpx_parser import GPXParser
from src.data.feature_engineering import FeatureEngineer  
from src.models.mlp_model import MLPModel

def load_and_prepare_data():
    """Load and prepare data for MLP tuning."""
    print("📂 Loading GPX data...")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize parsers
    gpx_parser = GPXParser()
    feature_engineer = FeatureEngineer(windows=config['features']['moving_averages'])
    
    # Load GPX files
    gpx_files = list(Path(config['data']['gpx_dir']).glob('*.gpx'))
    processed_data = {}
    
    total_points = 0
    for gpx_file in gpx_files:
        print(f"Processing {gpx_file.name}...")
        df = gpx_parser.parse_file(gpx_file)
        if df is not None and len(df) > 0:
            df_engineered = feature_engineer.process_dataframe(df)
            processed_data[gpx_file.stem] = df_engineered
            total_points += len(df_engineered)
    
    print(f"✅ Loaded {len(processed_data)} files with {total_points:,} total points")
    
    # Combine all datasets
    all_dataframes = []
    for filename, df in processed_data.items():
        df_copy = df.copy()
        df_copy['source_file'] = filename
        all_dataframes.append(df_copy)
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Define features (no HR features to prevent data leakage)
    feature_cols = ['cad', 'power', 'ele']
    for base_feature in ['cad', 'power']:
        for window in [5, 10, 30, 60]:
            feature_cols.append(f"{base_feature}{window}")
    
    # Add lag features
    for base_feature in ['power', 'cad']:
        for lag in [1, 5, 10, 15, 30]:
            feature_cols.append(f"{base_feature}_lag_{lag}")
    
    # Filter available features
    available_features = [col for col in feature_cols if col in combined_df.columns]
    available_features = [col for col in available_features if not col.startswith('hr')]
    
    print(f"📊 Selected features: {len(available_features)}")
    
    # Prepare X and y
    X = combined_df[available_features].fillna(0)
    y = combined_df['hr']
    
    # Remove rows with NaN target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    print(f"🎯 Training data: {len(X):,} samples, {X.shape[1]} features")
    return X, y, available_features

def main():
    """Run MLP hyperparameter tuning."""
    print("🔧 MLP Hyperparameter Tuning for Heart Rate Prediction")
    print("=" * 60)
    
    # Load data
    X, y, feature_cols = load_and_prepare_data()
    
    # Create MLP model with default config
    mlp_config = {
        'hidden_layers': [64, 32, 16],
        'learning_rate': 0.001,
        'batch_size': 256,
        'dropout': 0.2,
        'l2_reg': 0.001,
        'epochs': 100,
        'patience': 15,
        'normalize': True
    }
    
    # Initialize model
    print("\n🤖 Initializing MLP model...")
    mlp_model = MLPModel(mlp_config)
    mlp_model.feature_columns = feature_cols
    
    # Run hyperparameter tuning
    print("\n🚀 Starting hyperparameter optimization...")
    print("📈 This will test different combinations of:")
    print("   • Network architecture (layers, sizes)")
    print("   • Learning parameters (rate, batch size)")
    print("   • Regularization (dropout, L2)")
    print("   • Training settings (epochs, patience)")
    
    try:
        best_params, best_mae = mlp_model.tune_hyperparameters(
            X, y, 
            n_trials=50,  # Start with 50 for testing
            cv_folds=5
        )
        
        print(f"\n🎉 Hyperparameter tuning completed!")
        print(f"🏆 Best cross-validation MAE: {best_mae:.3f} bpm")
        
        if best_mae < 1.17:
            print(f"🚀 SUCCESS: Beat Random Forest target! ({best_mae:.3f} < 1.17 bpm)")
            print("🎯 Ready to deploy optimized MLP in main pipeline")
        else:
            print(f"📊 Progress made: {best_mae:.3f} bpm (vs RF: 1.17 bpm)")
            print("💡 Consider running more trials for further optimization")
        
        return best_params, best_mae
        
    except KeyboardInterrupt:
        print("\n⏹️  Tuning interrupted by user")
        return None, None
    except Exception as e:
        print(f"\n❌ Error during tuning: {e}")
        return None, None

if __name__ == "__main__":
    main()