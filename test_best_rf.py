#!/usr/bin/env python3
"""
Test the best Random Forest configuration found during tuning.
"""

import sys
import yaml
import pandas as pd
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.gpx_parser import GPXParser
from src.data.feature_engineering import FeatureEngineer
from src.models.random_forest_model import RandomForestModel
from src.utils.metrics import MetricsCalculator

def main():
    """Test the best configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    parser = GPXParser()
    feature_engineer = FeatureEngineer(
        windows=config['features']['moving_averages']
    )
    metrics_calc = MetricsCalculator()
    
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
    
    # Prepare features and target
    available_features = [col for col in combined_data.columns if col != 'hr']
    available_features = [col for col in available_features if not col.startswith('hr')]
    
    basic_features = ['cad', 'power', 'ele', 'cad5', 'cad10', 'cad30', 'cad60', 
                     'power5', 'power10', 'power30', 'power60', 
                     'power_lag_1', 'power_lag_5', 'power_lag_10', 'power_lag_15', 'power_lag_30',
                     'cad_lag_1', 'cad_lag_5', 'cad_lag_10', 'cad_lag_15', 'cad_lag_30']
    
    feature_cols = [col for col in basic_features if col in available_features]
    X = combined_data[feature_cols]
    y = combined_data['hr']
    
    logger.info(f"Testing best Random Forest configuration")
    logger.info(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Best configuration found
    best_params = {
        'n_estimators': 200,
        'max_depth': 20, 
        'min_samples_split': 3,
        'min_samples_leaf': 1
    }
    
    logger.info(f"Best parameters: {best_params}")
    
    # Create and train model
    rf_model = RandomForestModel(
        feature_columns=feature_cols,
        **best_params
    )
    
    rf_model.train(X, y)
    
    # Make predictions
    y_pred = rf_model.predict(X)
    
    # Calculate metrics
    metrics = metrics_calc.calculate_metrics(y.values, y_pred)
    
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZED RANDOM FOREST RESULTS")
    logger.info("="*60)
    logger.info(f"MAE:         {metrics['mae']:.4f} bpm")
    logger.info(f"RMSE:        {metrics['rmse']:.4f} bpm")  
    logger.info(f"R²:          {metrics['r2']:.4f}")
    logger.info(f"MAPE:        {metrics['mape']:.4f}%")
    logger.info(f"Correlation: {metrics['correlation']:.4f}")
    
    # Compare to baseline
    baseline_mae = 2.17
    improvement = baseline_mae - metrics['mae']
    improvement_pct = (improvement / baseline_mae) * 100
    
    logger.info("\n" + "="*60)
    logger.info("COMPARISON TO BASELINE")
    logger.info("="*60)
    logger.info(f"Baseline MAE: {baseline_mae:.4f} bpm")
    logger.info(f"New MAE:      {metrics['mae']:.4f} bpm")
    logger.info(f"Improvement:  {improvement:.4f} bpm ({improvement_pct:.1f}%)")
    
    target_achieved = metrics['mae'] < 2.0
    logger.info(f"\nTarget <2.0 bpm achieved: {'✅ YES' if target_achieved else '❌ NO'}")
    
    # Update config with best parameters
    if metrics['mae'] < baseline_mae:
        logger.info("\nUpdating config.yaml with optimized parameters...")
        
        config['models']['random_forest'].update(best_params)
        
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("Config updated successfully!")
        logger.info("Run 'python main.py' to use the optimized model")

if __name__ == "__main__":
    main()