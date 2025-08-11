#!/usr/bin/env python3
"""
HR Model Pipeline - Main Entry Point

This script runs the complete pipeline from GPX files to trained models.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.gpx_parser import GPXParser
from src.data.feature_engineering import FeatureEngineer
from src.models.ols_model import OLSModel
from src.models.ridge_model import RidgeModel
from src.models.elastic_net_model import ElasticNetModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.ensemble_model import EnsembleModel
from src.models.arima_model import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.state_space_model import StateSpaceModel
from src.models.deepar_model import DeepARModel
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import Visualizer


class HRModelPipeline:
    """Complete HR prediction pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.setup_directories()
        
        # Initialize components
        self.parser = GPXParser()
        self.feature_engineer = FeatureEngineer(
            windows=self.config['features']['moving_averages']
        )
        self.metrics_calc = MetricsCalculator()
        self.visualizer = Visualizer()
        
        # Data storage
        self.raw_data = {}
        self.processed_data = {}
        self.models = {}
        self.results = {}
        
        self.logger.info("Pipeline initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config['logging']['level'])
        log_file = self.config['logging']['file']
        
        # Create parent directory for log file if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.config['data']['output_dir'],
            self.config['data']['processed_data_dir']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_directories(self, model_name: str) -> dict:
        """Get all directories for a specific model."""
        model_base = Path(self.config['data']['model_output_pattern'].format(model_name=model_name))
        
        return {
            'base': model_base,
            'model_file': model_base / f'{model_name}_model.joblib',
            'predictions': model_base / 'predictions',
            'plots': model_base / 'plots',
            'reports': model_base / 'reports'
        }
    
    def load_data(self) -> None:
        """Load and parse GPX files."""
        self.logger.info("Loading GPX data...")
        
        gpx_dir = self.config['data']['gpx_dir']
        if not Path(gpx_dir).exists():
            raise FileNotFoundError(f"GPX directory not found: {gpx_dir}")
        
        self.raw_data = self.parser.parse_directory(gpx_dir)
        
        if not self.raw_data:
            raise ValueError("No GPX files found or parsed successfully")
        
        total_points = sum(len(df) for df in self.raw_data.values())
        self.logger.info(f"Loaded {len(self.raw_data)} files with {total_points:,} total points")
    
    def engineer_features(self) -> None:
        """Apply feature engineering to all datasets."""
        self.logger.info("Engineering features...")
        
        for filename, df in self.raw_data.items():
            processed_df = self.feature_engineer.process_dataframe(
                df,
                add_ma=True,
                add_derived=self.config['features']['add_derived']
            )
            
            self.processed_data[filename] = processed_df
            
            # Save processed data
            processed_data_dir = Path(self.config['data']['processed_data_dir'])
            processed_data_dir.mkdir(parents=True, exist_ok=True)
            output_file = processed_data_dir / f"{filename}.parquet"
            processed_df.to_parquet(output_file, compression='snappy')
            
            self.logger.info(f"Processed {filename}: {len(processed_df)} points, "
                           f"{processed_df.shape[1]} features")
    
    def prepare_training_data(self) -> tuple:
        """Combine all data and prepare for training."""
        self.logger.info("Preparing training data...")
        
        # Combine all datasets
        all_dataframes = []
        for filename, df in self.processed_data.items():
            df_copy = df.copy()
            df_copy['source_file'] = filename
            all_dataframes.append(df_copy)
        
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Define features based on configuration
        feature_cols = []
        
        # Base features
        feature_cols.extend(self.config['features']['base_features'])
        
        # Moving average features (exclude HR to prevent data leakage)
        for base_feature in ['cad', 'power']:  # Removed 'hr' - we can't use HR to predict HR
            for window in self.config['features']['moving_averages']:
                feature_cols.append(f"{base_feature}{window}")
        
        # Add power/cadence lag features (effort history)
        for base_feature in ['power', 'cad']:
            for lag in [1, 5, 10, 15, 30]:
                feature_cols.append(f"{base_feature}_lag_{lag}")
        
        # Filter to available features and exclude any HR-related features
        available_features = [col for col in feature_cols if col in combined_df.columns]
        
        # CRITICAL: Remove any HR-related features to prevent data leakage
        available_features = [col for col in available_features if not col.startswith('hr')]
        
        self.logger.info(f"Features selected (no HR leakage): {available_features}")
        
        # Prepare X and y
        X = combined_df[available_features].fillna(0)
        y = combined_df[self.config['features']['target']]
        
        # Remove rows where target is NaN
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        self.logger.info(f"Training data prepared: {len(X)} samples, {X.shape[1]} features")
        return X, y, available_features
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> None:
        """Train all configured models."""
        self.logger.info("Training models...")
        
        # Train OLS model
        if 'ols' in self.config['training']['models']:
            model_name = 'ols'
            self.logger.info(f"Training {model_name.upper()} model...")
            
            # Get model directories
            model_dirs = self.get_model_directories(model_name)
            model_dirs['base'].mkdir(parents=True, exist_ok=True)
            
            # Create and train model
            ols_model = OLSModel(
                feature_columns=feature_cols,
                normalize=self.config['models']['ols']['normalize']
            )
            ols_model.train(X, y)
            self.models[model_name] = ols_model
            
            # Save model
            ols_model.save(model_dirs['model_file'])
        
        # Train Ridge model
        if 'ridge' in self.config['training']['models']:
            model_name = 'ridge'
            self.logger.info(f"Training {model_name.upper()} model...")
            
            # Get model directories
            model_dirs = self.get_model_directories(model_name)
            model_dirs['base'].mkdir(parents=True, exist_ok=True)
            
            # Get Ridge configuration
            ridge_config = self.config['models']['ridge']
            alpha = ridge_config.get('alpha', 1.0)
            
            # If alpha_range is specified, use the middle value for now
            # (Later this could be enhanced with hyperparameter tuning)
            if 'alpha_range' in ridge_config:
                alpha_range = ridge_config['alpha_range']
                alpha = np.sqrt(alpha_range[0] * alpha_range[1])  # Geometric mean
            
            # Create and train model
            ridge_model = RidgeModel(
                feature_columns=feature_cols,
                alpha=alpha,
                normalize=ridge_config.get('normalize', True)
            )
            ridge_model.train(X, y)
            self.models[model_name] = ridge_model
            
            # Save model
            ridge_model.save(model_dirs['model_file'])
        
        # Train Elastic Net model
        if 'elastic_net' in self.config['training']['models']:
            model_name = 'elastic_net'
            self.logger.info(f"Training {model_name.upper()} model...")
            
            # Get model directories
            model_dirs = self.get_model_directories(model_name)
            model_dirs['base'].mkdir(parents=True, exist_ok=True)
            
            # Get Elastic Net configuration
            elastic_net_config = self.config['models']['elastic_net']
            alpha = elastic_net_config.get('alpha', 1.0)
            l1_ratio = elastic_net_config.get('l1_ratio', 0.5)
            
            # Create and train model
            elastic_net_model = ElasticNetModel(
                feature_columns=feature_cols,
                alpha=alpha,
                l1_ratio=l1_ratio,
                normalize=elastic_net_config.get('normalize', True)
            )
            elastic_net_model.train(X, y)
            self.models[model_name] = elastic_net_model
            
            # Save model
            elastic_net_model.save(model_dirs['model_file'])
        
        # Train Random Forest model
        if 'random_forest' in self.config['training']['models']:
            model_name = 'random_forest'
            self.logger.info(f"Training {model_name.upper()} model...")
            
            # Get model directories
            model_dirs = self.get_model_directories(model_name)
            model_dirs['base'].mkdir(parents=True, exist_ok=True)
            
            # Get Random Forest configuration
            rf_config = self.config['models']['random_forest']
            n_estimators = rf_config.get('n_estimators', 100)
            max_depth = rf_config.get('max_depth', None)
            min_samples_split = rf_config.get('min_samples_split', 5)
            min_samples_leaf = rf_config.get('min_samples_leaf', 2)
            
            # Create and train model
            rf_model = RandomForestModel(
                feature_columns=feature_cols,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                normalize=rf_config.get('normalize', False)
            )
            rf_model.train(X, y)
            self.models[model_name] = rf_model
            
            # Save model
            rf_model.save(model_dirs['model_file'])
        
        # Train XGBoost model
        if 'xgboost' in self.config['training']['models']:
            model_name = 'xgboost'
            self.logger.info(f"Training {model_name.upper()} model...")
            
            # Get model directories
            model_dirs = self.get_model_directories(model_name)
            model_dirs['base'].mkdir(parents=True, exist_ok=True)
            
            # Get XGBoost configuration
            xgb_config = self.config['models']['xgboost']
            n_estimators = xgb_config.get('n_estimators', 200)
            max_depth = xgb_config.get('max_depth', 6)
            learning_rate = xgb_config.get('learning_rate', 0.1)
            subsample = xgb_config.get('subsample', 0.8)
            colsample_bytree = xgb_config.get('colsample_bytree', 0.8)
            
            # Create and train model
            xgb_model = XGBoostModel(
                feature_columns=feature_cols,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                normalize=xgb_config.get('normalize', False)
            )
            xgb_model.train(X, y)
            self.models[model_name] = xgb_model
            
            # Save model
            xgb_model.save(model_dirs['model_file'])
        
        # Train LightGBM model
        if 'lightgbm' in self.config['training']['models']:
            model_name = 'lightgbm'
            self.logger.info(f"Training {model_name.upper()} model...")
            
            # Get model directories
            model_dirs = self.get_model_directories(model_name)
            model_dirs['base'].mkdir(parents=True, exist_ok=True)
            
            # Get LightGBM configuration
            lgb_config = self.config['models'].get('lightgbm', {})
            n_estimators = lgb_config.get('n_estimators', 200)
            max_depth = lgb_config.get('max_depth', -1)
            num_leaves = lgb_config.get('num_leaves', 31)
            learning_rate = lgb_config.get('learning_rate', 0.1)
            subsample = lgb_config.get('subsample', 0.8)
            colsample_bytree = lgb_config.get('colsample_bytree', 0.8)
            min_child_samples = lgb_config.get('min_child_samples', 20)
            reg_alpha = lgb_config.get('reg_alpha', 0.0)
            reg_lambda = lgb_config.get('reg_lambda', 0.0)
            
            # Create and train model
            lgb_model = LightGBMModel(
                feature_columns=feature_cols,
                n_estimators=n_estimators,
                max_depth=max_depth,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_samples=min_child_samples,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                normalize=lgb_config.get('normalize', False)
            )
            lgb_model.train(X, y)
            self.models[model_name] = lgb_model
            
            # Save model
            lgb_model.save(model_dirs['model_file'])
        
        # Train ARIMA model
        if 'arima' in self.config['training']['models']:
            model_name = 'arima'
            self.logger.info(f"Training {model_name.upper()} model...")
            
            # Get model directories
            model_dirs = self.get_model_directories(model_name)
            model_dirs['base'].mkdir(parents=True, exist_ok=True)
            
            # Get ARIMA configuration
            arima_config = self.config['models'].get('arima', {})
            order = tuple(arima_config.get('order', [2, 1, 2]))
            seasonal_order = tuple(arima_config.get('seasonal_order', [0, 0, 0, 0]))
            use_exog = arima_config.get('use_exog', True)
            
            # Create and train model
            arima_model = ARIMAModel(
                feature_columns=feature_cols,
                order=order,
                seasonal_order=seasonal_order,
                use_exog=use_exog,
                normalize=arima_config.get('normalize', False)
            )
            arima_model.train(X, y)
            self.models[model_name] = arima_model
            
            # Save model
            arima_model.save(model_dirs['model_file'])
        
        # Train Prophet model
        if 'prophet' in self.config['training']['models']:
            model_name = 'prophet'
            self.logger.info(f"Training {model_name.upper()} model...")
            
            # Get model directories
            model_dirs = self.get_model_directories(model_name)
            model_dirs['base'].mkdir(parents=True, exist_ok=True)
            
            # Get Prophet configuration
            prophet_config = self.config['models'].get('prophet', {})
            yearly_seasonality = prophet_config.get('yearly_seasonality', False)
            weekly_seasonality = prophet_config.get('weekly_seasonality', False)
            daily_seasonality = prophet_config.get('daily_seasonality', True)
            changepoint_prior_scale = prophet_config.get('changepoint_prior_scale', 0.05)
            seasonality_prior_scale = prophet_config.get('seasonality_prior_scale', 10.0)
            
            # Create and train model
            prophet_model = ProphetModel(
                feature_columns=feature_cols,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                normalize=prophet_config.get('normalize', False)
            )
            prophet_model.train(X, y)
            self.models[model_name] = prophet_model
            
            # Save model
            prophet_model.save(model_dirs['model_file'])
        
        # Train State Space model
        if 'state_space' in self.config['training']['models']:
            model_name = 'state_space'
            self.logger.info(f"Training {model_name.upper()} model...")
            
            # Get model directories
            model_dirs = self.get_model_directories(model_name)
            model_dirs['base'].mkdir(parents=True, exist_ok=True)
            
            # Get State Space configuration
            ss_config = self.config['models'].get('state_space', {})
            level = ss_config.get('level', True)
            trend = ss_config.get('trend', True)
            seasonal = ss_config.get('seasonal', None)
            use_exog = ss_config.get('use_exog', True)
            
            # Create and train model
            ss_model = StateSpaceModel(
                feature_columns=feature_cols,
                level=level,
                trend=trend,
                seasonal=seasonal,
                use_exog=use_exog,
                normalize=ss_config.get('normalize', False)
            )
            ss_model.train(X, y)
            self.models[model_name] = ss_model
            
            # Save model
            ss_model.save(model_dirs['model_file'])
        
        # Train DeepAR model
        if 'deepar' in self.config['training']['models']:
            model_name = 'deepar'
            self.logger.info(f"Training {model_name.upper()} model...")
            
            # Get model directories
            model_dirs = self.get_model_directories(model_name)
            model_dirs['base'].mkdir(parents=True, exist_ok=True)
            
            # Get DeepAR configuration
            deepar_config = self.config['models'].get('deepar', {})
            hidden_size = deepar_config.get('hidden_size', 64)
            num_layers = deepar_config.get('num_layers', 2)
            dropout = deepar_config.get('dropout', 0.1)
            learning_rate = deepar_config.get('learning_rate', 0.001)
            n_epochs = deepar_config.get('n_epochs', 50)
            batch_size = deepar_config.get('batch_size', 32)
            sequence_length = deepar_config.get('sequence_length', 30)
            
            # Create and train model
            deepar_model = DeepARModel(
                feature_columns=feature_cols,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                batch_size=batch_size,
                sequence_length=sequence_length,
                normalize=deepar_config.get('normalize', True)
            )
            deepar_model.train(X, y)
            self.models[model_name] = deepar_model
            
            # Save model
            deepar_model.save(model_dirs['model_file'])
        
        # Train Ensemble model
        if 'ensemble' in self.config['training']['models']:
            model_name = 'ensemble'
            self.logger.info(f"Training {model_name.upper()} model...")
            
            # Get model directories
            model_dirs = self.get_model_directories(model_name)
            model_dirs['base'].mkdir(parents=True, exist_ok=True)
            
            # Get Ensemble configuration
            ensemble_config = self.config['models']['ensemble']
            
            # Create and train model
            ensemble_model = EnsembleModel(
                feature_columns=feature_cols,
                rf_n_estimators=ensemble_config.get('rf_n_estimators', 200),
                rf_max_depth=ensemble_config.get('rf_max_depth', 15),
                rf_min_samples_split=ensemble_config.get('rf_min_samples_split', 5),
                rf_min_samples_leaf=ensemble_config.get('rf_min_samples_leaf', 2),
                rf_max_features=ensemble_config.get('rf_max_features', 'sqrt'),
                xgb_n_estimators=ensemble_config.get('xgb_n_estimators', 150),
                xgb_max_depth=ensemble_config.get('xgb_max_depth', 8),
                xgb_learning_rate=ensemble_config.get('xgb_learning_rate', 0.1),
                xgb_subsample=ensemble_config.get('xgb_subsample', 0.8),
                xgb_colsample_bytree=ensemble_config.get('xgb_colsample_bytree', 0.8),
                rf_weight=ensemble_config.get('rf_weight', 0.7),
                xgb_weight=ensemble_config.get('xgb_weight', 0.3)
            )
            
            # Train the ensemble model
            ensemble_model.train(X, y)
            self.models[model_name] = ensemble_model
    
    def evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Evaluate all trained models."""
        self.logger.info("Evaluating models...")
        
        for model_name, model in self.models.items():
            self.logger.info(f"Evaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            metrics = self.metrics_calc.calculate_metrics(y.values, y_pred)
            
            # Store results
            self.results[model_name] = {
                'metrics': metrics,
                'predictions': {'y_true': y.values, 'y_pred': y_pred},
                'model': model
            }
            
            self.logger.info(f"{model_name} - MAE: {metrics['mae']:.2f}, "
                           f"RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
    
    def apply_to_files(self) -> None:
        """Apply all trained models to individual files."""
        self.logger.info("Applying models to individual files...")
        
        for model_name, model in self.models.items():
            self.logger.info(f"Applying {model_name.upper()} model to files...")
            
            # Get model directories
            model_dirs = self.get_model_directories(model_name)
            model_dirs['predictions'].mkdir(parents=True, exist_ok=True)
            
            for filename, df in self.processed_data.items():
                self.logger.info(f"Processing {filename} with {model_name}...")
                
                # Prepare features
                X = model.prepare_features(df)
                
                # Make predictions
                predictions = model.predict(X)
                df_with_pred = df.copy()
                df_with_pred['predicted_hr'] = predictions
                
                # Calculate error if actual HR is available
                if 'hr' in df.columns:
                    df_with_pred['hr_error'] = df_with_pred['hr'] - df_with_pred['predicted_hr']
                
                # Save results
                output_file = model_dirs['predictions'] / f"{filename}.parquet"
                df_with_pred.to_parquet(output_file, compression='snappy')
                
                # Calculate metrics for this file
                if 'hr' in df.columns:
                    file_metrics = self.metrics_calc.calculate_metrics(
                        df['hr'].values, predictions
                    )
                    self.logger.info(f"{filename} - MAE: {file_metrics['mae']:.2f}, "
                                   f"RMSE: {file_metrics['rmse']:.2f}")
    
    def create_visualizations(self) -> None:
        """Create visualization plots using parallel processing."""
        if not self.config['evaluation']['create_plots']:
            return
        
        self.logger.info("Creating visualizations in parallel...")
        
        # Use ThreadPoolExecutor for parallel visualization generation
        max_workers = min(len(self.results), 8)  # Limit to 4 threads to avoid overwhelming the system
        
        def create_model_visualizations(model_item):
            """Create visualizations for a single model."""

            model_name, results = model_item
            self.logger.info(f"Creating visualizations for {model_name}")
            
            try:
                # Get model directories
                model_dirs = self.get_model_directories(model_name)
                model_plots_dir = model_dirs['plots']
                model_plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Global predictions plot
                self.visualizer.plot_predictions(
                    results['predictions']['y_true'],
                    results['predictions']['y_pred'],
                    title=f"{model_name.upper()} Model - Overall Performance",
                    save_path=model_plots_dir / 'predictions.png'
                )
                
                # Feature importance (if available)
                if hasattr(results['model'], 'get_feature_importance'):
                    importance_df = results['model'].get_feature_importance()
                    if importance_df is not None:
                        self.visualizer.plot_feature_importance(
                            importance_df,
                            title=f"{model_name.upper()} Feature Importance",
                            save_path=model_plots_dir / 'feature_importance.png'
                        )
                
                # Create time-series plots for each input file
                self.create_file_time_series_plots(model_name, model_plots_dir)

                self.logger.info(f"Completed visualizations for {model_name}")
                return f"✅ {model_name}"
                
            except Exception as e:
                error_msg = f"❌ {model_name}: {str(e)}"
                self.logger.error(f"Error creating visualizations for {model_name}: {e}", exc_info=True)
                return error_msg


        for model_result in self.results.items():
            create_model_visualizations(model_result)

    def create_file_time_series_plots(self, model_name: str, plots_dir: Path) -> None:
        """Create time-series plots for each input file."""
        self.logger.info(f"Creating time-series plots for {model_name}...")
        
        # Create subdirectory for file-specific plots
        files_plots_dir = plots_dir / 'files'
        files_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model directories
        model_dirs = self.get_model_directories(model_name)
        model_results_dir = model_dirs['predictions']
        
        if not model_results_dir.exists():
            self.logger.warning(f"No results directory found for {model_name}")
            return
        
        # Process each result file
        result_files = list(model_results_dir.glob('*.parquet'))
        
        for result_file in result_files:
            file_name = result_file.stem
            self.logger.info(f"Creating plots for {file_name}...")
            
            try:
                # Load the result file
                df = pd.read_parquet(result_file)
                
                # Check required columns
                if 'hr' not in df.columns or 'predicted_hr' not in df.columns:
                    self.logger.warning(f"Missing required columns in {file_name}")
                    continue
                
                # Basic time series plot
                self.visualizer.plot_time_series(
                    df,
                    columns=['hr', 'predicted_hr'],
                    title=f"{model_name.upper()} - {file_name}: HR vs Predicted HR",
                    save_path=files_plots_dir / f'{file_name}_timeseries.png'
                )
                
                # Time series with error plot
                self.visualizer.plot_time_series_with_error(
                    df,
                    title=f"{model_name.upper()} - {file_name}: HR with Prediction Error",
                    save_path=files_plots_dir / f'{file_name}_error.png'
                )
                
                self.logger.info(f"Created plots for {file_name}")
                
            except Exception as e:
                self.logger.error(f"Error creating plots for {file_name}: {e}")
    
    def create_report(self) -> None:
        """Create final report."""
        self.logger.info("Creating report...")
        
        # Compile metrics
        metrics_summary = {}
        for model_name, results in self.results.items():
            metrics_summary[model_name] = results['metrics']
        
        # Create comparison DataFrame
        comparison_df = self.metrics_calc.compare_models(metrics_summary)
        
        # Save report to main output directory
        report_path = Path(self.config['data']['output_dir']) / 'model_comparison.csv'
        comparison_df.to_csv(report_path)
        
        # Print summary
        print("\\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(comparison_df.round(4).to_string())
        print("\\n" + "="*60)
        
        best_model = comparison_df.index[0]
        best_mae = comparison_df.loc[best_model, 'mae']
        best_r2 = comparison_df.loc[best_model, 'r2']
        
        print(f"Best Model: {best_model}")
        print(f"Best MAE: {best_mae:.2f} bpm")
        print(f"Best R²: {best_r2:.4f}")
        print("="*60)
    
    def run(self) -> None:
        """Run the complete pipeline."""
        start_time = datetime.now()
        self.logger.info("Starting HR Model Pipeline")
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Feature engineering
            self.engineer_features()
            
            # Step 3: Prepare training data
            X, y, feature_cols = self.prepare_training_data()
            
            # Step 4: Train models
            self.train_models(X, y, feature_cols)
            
            # Step 5: Evaluate models
            self.evaluate_models(X, y)
            
            # Step 6: Apply to individual files
            self.apply_to_files()
            
            # Step 7: Create visualizations
            self.create_visualizations()
            
            # Step 8: Create report
            self.create_report()
            
            # Completion
            duration = datetime.now() - start_time
            self.logger.info(f"Pipeline completed successfully in {duration}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='HR Model Training Pipeline')
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run pipeline
    pipeline = HRModelPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()