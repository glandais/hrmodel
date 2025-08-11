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

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.gpx_parser import GPXParser
from src.data.feature_engineering import FeatureEngineer
from src.models.ols_model import OLSModel
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
                add_derived=self.config['features']['add_derived'],
                add_zones=self.config['features']['add_hr_zones']
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
        
        # Moving average features
        for base_feature in ['cad', 'power', 'hr']:  # Note: hr moving averages might be used as features
            if base_feature == self.config['features']['target']:
                continue  # Skip target variable moving averages
            for window in self.config['features']['moving_averages']:
                feature_cols.append(f"{base_feature}{window}")
        
        # Filter to available features
        available_features = [col for col in feature_cols if col in combined_df.columns]
        
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
        
        # For now, start with OLS
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
        """Create visualization plots."""
        if not self.config['evaluation']['create_plots']:
            return
        
        self.logger.info("Creating visualizations...")
        
        for model_name, results in self.results.items():
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