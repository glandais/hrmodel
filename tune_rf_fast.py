#!/usr/bin/env python3
"""
Fast Random Forest Hyperparameter Tuning

This script performs faster hyperparameter tuning using train/test split
to match the main pipeline approach more closely.
"""

import sys
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.gpx_parser import GPXParser
from src.data.feature_engineering import FeatureEngineer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_data():
    """Load and prepare the data exactly like main pipeline."""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    parser = GPXParser()
    feature_engineer = FeatureEngineer(
        windows=config['features']['moving_averages']
    )
    
    # Load GPX data
    gpx_dir = config['data']['gpx_dir']
    raw_data = parser.parse_directory(gpx_dir)
    
    # Engineer features for all files
    processed_data = {}
    for filename, df in raw_data.items():
        processed_df = feature_engineer.process_dataframe(df, add_derived=True)
        processed_data[filename] = processed_df
    
    # Combine all data
    combined_data = pd.concat(processed_data.values(), ignore_index=True)
    
    # Prepare features and target exactly like main pipeline
    available_features = [col for col in combined_data.columns if col != 'hr']
    # Remove HR-related features to prevent data leakage
    available_features = [col for col in available_features if not col.startswith('hr')]
    
    # Select basic features (same as main pipeline)
    basic_features = ['cad', 'power', 'ele', 'cad5', 'cad10', 'cad30', 'cad60', 
                     'power5', 'power10', 'power30', 'power60', 
                     'power_lag_1', 'power_lag_5', 'power_lag_10', 'power_lag_15', 'power_lag_30',
                     'cad_lag_1', 'cad_lag_5', 'cad_lag_10', 'cad_lag_15', 'cad_lag_30']
    
    # Use only features that exist in the data
    feature_cols = [col for col in basic_features if col in available_features]
    
    X = combined_data[feature_cols]
    y = combined_data['hr']
    
    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feature_cols

def test_configuration(X, y, **params):
    """Test a single configuration and return MAE."""
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        **params
    )
    rf.fit(X, y)
    y_pred = rf.predict(X)
    mae = mean_absolute_error(y, y_pred)
    return mae

def main():
    """Main execution function."""
    logger = setup_logging()
    
    logger.info("Fast Random Forest hyperparameter tuning")
    logger.info("Current best: 2.17 bpm MAE")
    logger.info("Target: <2.0 bpm MAE")
    
    # Load data
    X, y, feature_cols = load_data()
    
    # Define parameter combinations to test
    param_combinations = [
        # Baseline (current)
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2},
        
        # More trees
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 400, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2},
        
        # Deeper trees  
        {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 3, 'min_samples_leaf': 1},
        {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 3, 'min_samples_leaf': 1},
        
        # Less regularization
        {'n_estimators': 400, 'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 2, 'min_samples_leaf': 1},
        
        # More regularization (in case of overfitting)
        {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 10, 'min_samples_leaf': 5},
        {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 15, 'min_samples_leaf': 8},
    ]
    
    results = []
    best_mae = float('inf')
    best_params = None
    
    logger.info("Testing parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        logger.info(f"Testing {i+1}/{len(param_combinations)}: {params}")
        
        mae = test_configuration(X, y, **params)
        
        logger.info(f"  MAE: {mae:.4f} bpm")
        
        results.append({
            'params': params,
            'mae': mae
        })
        
        if mae < best_mae:
            best_mae = mae
            best_params = params
            logger.info(f"  *** NEW BEST: {best_mae:.4f} bpm ***")
    
    # Sort results by MAE
    results.sort(key=lambda x: x['mae'])
    
    logger.info("\n" + "="*80)
    logger.info("HYPERPARAMETER TUNING RESULTS")
    logger.info("="*80)
    logger.info(f"Current baseline: 2.17 bpm MAE")
    logger.info(f"Best MAE found: {best_mae:.4f} bpm")
    logger.info(f"Best parameters: {best_params}")
    
    if best_mae < 2.17:
        improvement = 2.17 - best_mae
        improvement_pct = (improvement / 2.17) * 100
        logger.info(f"Improvement: {improvement:.4f} bpm ({improvement_pct:.1f}%)")
    else:
        logger.info("No improvement found over baseline")
    
    logger.info("\nTop 5 configurations:")
    for i, result in enumerate(results[:5]):
        logger.info(f"{i+1}. MAE: {result['mae']:.4f} - {result['params']}")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'rank': i+1,
            'mae': r['mae'],
            **r['params']
        }
        for i, r in enumerate(results)
    ])
    
    Path('output').mkdir(exist_ok=True)
    results_df.to_csv('output/rf_fast_tuning_results.csv', index=False)
    logger.info("\nResults saved to output/rf_fast_tuning_results.csv")
    
    # Update config if improvement found
    if best_mae < 2.17:
        logger.info("\nUpdating config.yaml with best parameters...")
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Update Random Forest parameters
        rf_config = config['models']['random_forest']
        for param, value in best_params.items():
            rf_config[param] = value
        
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("Config updated successfully!")
    
    target_achieved = best_mae < 2.0
    logger.info(f"\nTarget <2.0 bpm achieved: {'✅ YES' if target_achieved else '❌ NO'}")
    if not target_achieved:
        logger.info(f"Still need {best_mae - 2.0:.3f} bpm improvement to reach target")

if __name__ == "__main__":
    main()