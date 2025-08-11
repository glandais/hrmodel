#!/usr/bin/env python3
"""
Random Forest Hyperparameter Tuning

This script performs systematic hyperparameter tuning for the Random Forest model
to achieve better performance than the current 2.17 bpm MAE.
"""

import sys
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from pathlib import Path
import logging
from itertools import product

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.gpx_parser import GPXParser
from src.data.feature_engineering import FeatureEngineer
from src.utils.metrics import MetricsCalculator

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_data():
    """Load and prepare the data."""
    logger = logging.getLogger(__name__)
    logger.info("Loading and preparing data...")
    
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
    logger.info(f"Loaded {len(raw_data)} files")
    
    # Engineer features for all files
    processed_data = {}
    for filename, df in raw_data.items():
        processed_df = feature_engineer.process_dataframe(df, add_derived=True)
        processed_data[filename] = processed_df
        logger.info(f"Processed {filename}: {processed_df.shape[0]} points, {processed_df.shape[1]} features")
    
    # Combine all data
    combined_data = pd.concat(processed_data.values(), ignore_index=True)
    
    # Prepare features and target
    available_features = [col for col in combined_data.columns if col != 'hr']
    # Remove HR-related features to prevent data leakage
    available_features = [col for col in available_features if not col.startswith('hr')]
    
    # Select basic features to avoid overfitting with too many features
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

def tune_random_forest_systematic():
    """Perform systematic hyperparameter tuning."""
    logger = logging.getLogger(__name__)
    logger.info("Starting systematic Random Forest hyperparameter tuning...")
    
    X, y, feature_cols = load_data()
    
    # Define parameter grid for systematic search
    param_combinations = [
        # Baseline (current best)
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
        
        # More trees
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
        {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
        
        # Deeper trees
        {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
        {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
        
        # Different feature selection
        {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2'},
        {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': None},
        
        # Regularization variants
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 250, 'max_depth': 18, 'min_samples_split': 7, 'min_samples_leaf': 3, 'max_features': 'sqrt'},
        
        # Aggressive overfitting prevention
        {'n_estimators': 400, 'max_depth': 12, 'min_samples_split': 15, 'min_samples_leaf': 8, 'max_features': 'sqrt'},
    ]
    
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    results = []
    best_mae = float('inf')
    best_params = None
    
    for i, params in enumerate(param_combinations):
        logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        # Create model
        rf = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )
        
        # 5-fold cross-validation
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring=mae_scorer, n_jobs=-1)
        mean_mae = -cv_scores.mean()  # Convert back from negative
        std_mae = cv_scores.std()
        
        logger.info(f"  MAE: {mean_mae:.4f} (+/- {std_mae:.4f})")
        
        results.append({
            'params': params,
            'mae_mean': mean_mae,
            'mae_std': std_mae,
            'mae_min': mean_mae - std_mae,
            'mae_max': mean_mae + std_mae
        })
        
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_params = params
            logger.info(f"  *** NEW BEST: MAE {best_mae:.4f} ***")
    
    # Sort results by MAE
    results.sort(key=lambda x: x['mae_mean'])
    
    logger.info("\n" + "="*80)
    logger.info("HYPERPARAMETER TUNING RESULTS")
    logger.info("="*80)
    logger.info(f"Current baseline: 2.17 bpm MAE")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best MAE: {best_mae:.4f} bpm")
    logger.info(f"Improvement: {2.17 - best_mae:.4f} bpm ({((2.17 - best_mae)/2.17)*100:.1f}%)")
    
    logger.info("\nTop 5 configurations:")
    for i, result in enumerate(results[:5]):
        logger.info(f"{i+1}. MAE: {result['mae_mean']:.4f} (+/- {result['mae_std']:.4f}) - {result['params']}")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'rank': i+1,
            'mae_mean': r['mae_mean'],
            'mae_std': r['mae_std'],
            'mae_min': r['mae_min'],
            'mae_max': r['mae_max'],
            **r['params']
        }
        for i, r in enumerate(results)
    ])
    
    results_df.to_csv('output/random_forest_tuning_results.csv', index=False)
    logger.info("\nResults saved to output/random_forest_tuning_results.csv")
    
    return best_params, best_mae

def tune_random_forest_randomized():
    """Perform randomized search for broader exploration."""
    logger = logging.getLogger(__name__)
    logger.info("Starting randomized hyperparameter search...")
    
    X, y, feature_cols = load_data()
    
    # Define parameter distributions
    param_distributions = {
        'n_estimators': [100, 150, 200, 250, 300, 400, 500],
        'max_depth': [10, 12, 15, 18, 20, 22, 25, None],
        'min_samples_split': [2, 3, 5, 7, 10, 15, 20],
        'min_samples_leaf': [1, 2, 3, 5, 8, 12],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
        'bootstrap': [True, False],
        'max_samples': [0.7, 0.8, 0.9, None]  # Only used when bootstrap=True
    }
    
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Randomized search
    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_distributions,
        n_iter=50,  # Try 50 random combinations
        cv=5,
        scoring=mae_scorer,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    logger.info("Performing randomized search with 50 iterations...")
    random_search.fit(X, y)
    
    best_mae = -random_search.best_score_
    best_params = random_search.best_params_
    
    logger.info("\n" + "="*80)
    logger.info("RANDOMIZED SEARCH RESULTS")
    logger.info("="*80)
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best MAE: {best_mae:.4f} bpm")
    logger.info(f"Current baseline: 2.17 bpm")
    logger.info(f"Improvement: {2.17 - best_mae:.4f} bpm ({((2.17 - best_mae)/2.17)*100:.1f}%)")
    
    # Save detailed results
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('mean_test_score', ascending=False)
    results_df.to_csv('output/random_forest_randomized_search.csv', index=False)
    
    return best_params, best_mae

def main():
    """Main execution function."""
    logger = setup_logging()
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    logger.info("Starting Random Forest hyperparameter tuning")
    logger.info("Current best: 2.17 bpm MAE")
    logger.info("Target: <2.0 bpm MAE")
    
    # First, systematic search
    best_systematic_params, best_systematic_mae = tune_random_forest_systematic()
    
    # Then, randomized search for broader exploration
    best_random_params, best_random_mae = tune_random_forest_randomized()
    
    # Compare results
    logger.info("\n" + "="*80)
    logger.info("FINAL COMPARISON")
    logger.info("="*80)
    logger.info(f"Systematic search - MAE: {best_systematic_mae:.4f}, Params: {best_systematic_params}")
    logger.info(f"Randomized search - MAE: {best_random_mae:.4f}, Params: {best_random_params}")
    
    if best_systematic_mae < best_random_mae:
        final_best_params = best_systematic_params
        final_best_mae = best_systematic_mae
        search_type = "Systematic"
    else:
        final_best_params = best_random_params
        final_best_mae = best_random_mae
        search_type = "Randomized"
    
    logger.info(f"\nOverall best from {search_type} search:")
    logger.info(f"MAE: {final_best_mae:.4f} bpm")
    logger.info(f"Improvement over baseline: {2.17 - final_best_mae:.4f} bpm ({((2.17 - final_best_mae)/2.17)*100:.1f}%)")
    logger.info(f"Parameters: {final_best_params}")
    
    # Update config.yaml with best parameters
    if final_best_mae < 2.17:  # Only update if we found improvement
        logger.info("\nUpdating config.yaml with best parameters...")
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Update Random Forest parameters
        rf_config = config['models']['random_forest']
        for param, value in final_best_params.items():
            rf_config[param] = value
        
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("Config updated successfully!")
    else:
        logger.info("No improvement found. Keeping current parameters.")
    
    target_achieved = final_best_mae < 2.0
    logger.info(f"\nTarget <2.0 bpm achieved: {'✅ YES' if target_achieved else '❌ NO'}")
    if not target_achieved:
        logger.info(f"Still need {final_best_mae - 2.0:.3f} bpm improvement to reach target")

if __name__ == "__main__":
    main()