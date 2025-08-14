#!/usr/bin/env python3
"""
Comprehensive Feature Impact Analysis for Random Forest Model
Analyzes how different feature groups affect MAE performance
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.gpx_parser import GPXParser
from src.data.feature_engineering import FeatureEngineer
from src.models.random_forest_model import RandomForestModel
from src.utils.metrics import MetricsCalculator


class FeatureImpactAnalyzer:
    """Analyzes feature impact on Random Forest MAE performance."""
    
    def __init__(self, X, y, feature_names, model_params):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.model_params = model_params
        self.baseline_mae = None
        self.metrics_calc = MetricsCalculator()
        
        # Define feature groups
        self.feature_groups = self._define_feature_groups()
        
    def _define_feature_groups(self) -> Dict[str, List[str]]:
        """Define logical feature groups based on available features."""
        groups = {
            "Raw Sensors": [],
            "Cadence Features": [],
            "Power Features": [],
            "Elevation Features": [],
            "Moving Averages - Cadence": [],
            "Moving Averages - Power": [],
            "Lag Features - Cadence": [],
            "Lag Features - Power": [],
            "Power Intensity": [],
            "Cadence Variability": [],
            "Physiological Response": [],
            "Fatigue Indicators": [],
            "Power-Cadence Interaction": [],
            "Terrain Features": [],
            "Temporal Features": []
        }
        
        for feature in self.feature_names:
            # Raw sensors
            if feature in ['cad', 'power', 'ele']:
                groups["Raw Sensors"].append(feature)
                
            # Moving averages
            elif feature.startswith('cad') and any(str(w) in feature for w in [5, 10, 30, 60]):
                groups["Moving Averages - Cadence"].append(feature)
            elif feature.startswith('power') and any(str(w) in feature for w in [5, 10, 30, 60]):
                groups["Moving Averages - Power"].append(feature)
                
            # Lag features
            elif 'cad_lag' in feature:
                groups["Lag Features - Cadence"].append(feature)
            elif 'power_lag' in feature:
                groups["Lag Features - Power"].append(feature)
                
            # Power intensity features
            elif any(x in feature for x in ['power_cv', 'power_zone', 'power_normalized', 
                                            'power_above', 'critical_power', 'anaerobic']):
                groups["Power Intensity"].append(feature)
                
            # Cadence variability
            elif any(x in feature for x in ['cad_std', 'cad_optimal', 'cad_too', 
                                            'cad_smoothness', 'cad_change']):
                groups["Cadence Variability"].append(feature)
                
            # Physiological response
            elif any(x in feature for x in ['effort_response', 'effort_ramp', 'intensity_factor']):
                groups["Physiological Response"].append(feature)
                
            # Fatigue indicators
            elif any(x in feature for x in ['cumulative_work', 'fatigue_index', 'intense_work']):
                groups["Fatigue Indicators"].append(feature)
                
            # Power-cadence interaction
            elif any(x in feature for x in ['power_per_cad', 'torque_proxy', 'cad_efficiency']):
                groups["Power-Cadence Interaction"].append(feature)
                
            # Terrain features
            elif any(x in feature for x in ['gradient', 'ele_gradient', 'terrain', 
                                            'climbing', 'vam_proxy', 'elevation_gain']):
                groups["Terrain Features"].append(feature)
                
            # Temporal features
            elif any(x in feature for x in ['time_elapsed', 'hour', 'minute', 
                                            'workout_progress', 'circadian']):
                groups["Temporal Features"].append(feature)
                
            # Elevation (not captured above)
            elif 'ele' in feature:
                groups["Elevation Features"].append(feature)
                
            # Generic cadence (not captured above)
            elif 'cad' in feature:
                groups["Cadence Features"].append(feature)
                
            # Generic power (not captured above)
            elif 'power' in feature:
                groups["Power Features"].append(feature)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        
        return groups
    
    def calculate_baseline_performance(self) -> float:
        """Calculate baseline MAE with all features."""
        logging.info("Calculating baseline performance with all features...")
        
        model = RandomForestModel(
            feature_columns=self.feature_names,
            **self.model_params
        )
        model.train(self.X, self.y)
        
        y_pred = model.predict(self.X)
        metrics = self.metrics_calc.calculate_metrics(self.y.values, y_pred)
        self.baseline_mae = metrics['mae']
        
        logging.info(f"Baseline MAE: {self.baseline_mae:.4f} bpm")
        
        # Store feature importances
        self.baseline_importances = model.get_feature_importance()
        
        return self.baseline_mae
    
    def permutation_importance_analysis(self, n_repeats=10) -> pd.DataFrame:
        """Perform permutation importance analysis."""
        logging.info("Performing permutation importance analysis...")
        
        # Train model
        model = RandomForestModel(
            feature_columns=self.feature_names,
            **self.model_params
        )
        model.train(self.X, self.y)
        
        # Calculate permutation importance
        result = permutation_importance(
            model.model, self.X, self.y,
            n_repeats=n_repeats,
            random_state=42,
            scoring='neg_mean_absolute_error'
        )
        
        # Create DataFrame with results
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': -result.importances_mean,  # Convert to positive MAE increase
            'importance_std': result.importances_std
        })
        
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        
        logging.info("Top 10 features by permutation importance (MAE increase):")
        for _, row in importance_df.head(10).iterrows():
            logging.info(f"  {row['feature']}: +{row['importance_mean']:.4f} ± {row['importance_std']:.4f} bpm")
        
        return importance_df
    
    def drop_column_importance(self) -> pd.DataFrame:
        """Calculate importance by dropping each feature and measuring MAE increase."""
        logging.info("Performing drop-column importance analysis...")
        
        results = []
        
        for feature in self.feature_names:
            # Create feature set without current feature
            X_dropped = self.X.drop(columns=[feature])
            feature_cols_dropped = [f for f in self.feature_names if f != feature]
            
            # Train model without feature
            model = RandomForestModel(
                feature_columns=feature_cols_dropped,
                **self.model_params
            )
            model.train(X_dropped, self.y)
            
            # Calculate MAE
            y_pred = model.predict(X_dropped)
            metrics = self.metrics_calc.calculate_metrics(self.y.values, y_pred)
            mae_without = metrics['mae']
            
            # Calculate MAE increase
            mae_increase = mae_without - self.baseline_mae
            
            results.append({
                'feature': feature,
                'mae_without': mae_without,
                'mae_increase': mae_increase,
                'relative_increase': (mae_increase / self.baseline_mae) * 100
            })
            
            logging.info(f"  Dropped {feature}: MAE = {mae_without:.4f} (+{mae_increase:.4f} bpm)")
        
        drop_importance_df = pd.DataFrame(results).sort_values('mae_increase', ascending=False)
        
        logging.info("\nTop 10 features by drop-column importance:")
        for _, row in drop_importance_df.head(10).iterrows():
            logging.info(f"  {row['feature']}: +{row['mae_increase']:.4f} bpm ({row['relative_increase']:.1f}%)")
        
        return drop_importance_df
    
    def group_importance_analysis(self) -> pd.DataFrame:
        """Analyze importance of feature groups."""
        logging.info("\nAnalyzing feature group importance...")
        
        group_results = []
        
        for group_name, group_features in self.feature_groups.items():
            if not group_features:
                continue
                
            # Create feature set without group
            features_to_keep = [f for f in self.feature_names if f not in group_features]
            
            if not features_to_keep:
                logging.warning(f"  Skipping {group_name} - would remove all features")
                continue
                
            X_without_group = self.X[features_to_keep]
            
            # Train model without group
            model = RandomForestModel(
                feature_columns=features_to_keep,
                **self.model_params
            )
            model.train(X_without_group, self.y)
            
            # Calculate MAE
            y_pred = model.predict(X_without_group)
            metrics = self.metrics_calc.calculate_metrics(self.y.values, y_pred)
            mae_without = metrics['mae']
            
            # Calculate impact
            mae_increase = mae_without - self.baseline_mae
            
            group_results.append({
                'group': group_name,
                'n_features': len(group_features),
                'mae_without': mae_without,
                'mae_increase': mae_increase,
                'relative_increase': (mae_increase / self.baseline_mae) * 100,
                'features': ', '.join(group_features[:3]) + ('...' if len(group_features) > 3 else '')
            })
            
            logging.info(f"  {group_name} ({len(group_features)} features): "
                        f"MAE = {mae_without:.4f} (+{mae_increase:.4f} bpm, "
                        f"+{(mae_increase/self.baseline_mae)*100:.1f}%)")
        
        group_importance_df = pd.DataFrame(group_results).sort_values('mae_increase', ascending=False)
        
        return group_importance_df
    
    def incremental_feature_analysis(self) -> pd.DataFrame:
        """Analyze how MAE improves as features are added incrementally."""
        logging.info("\nPerforming incremental feature analysis...")
        
        # Get feature importance ranking
        feature_ranking = self.baseline_importances['feature'].tolist()
        
        incremental_results = []
        
        for n_features in [1, 2, 3, 5, 10, 15, 20, len(feature_ranking)]:
            if n_features > len(feature_ranking):
                break
                
            # Select top n features
            selected_features = feature_ranking[:n_features]
            X_selected = self.X[selected_features]
            
            # Train model with selected features
            model = RandomForestModel(
                feature_columns=selected_features,
                **self.model_params
            )
            model.train(X_selected, self.y)
            
            # Calculate MAE
            y_pred = model.predict(X_selected)
            metrics = self.metrics_calc.calculate_metrics(self.y.values, y_pred)
            
            incremental_results.append({
                'n_features': n_features,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'top_features': ', '.join(selected_features[:3]) + ('...' if n_features > 3 else '')
            })
            
            logging.info(f"  Top {n_features} features: MAE = {metrics['mae']:.4f}, R² = {metrics['r2']:.4f}")
        
        incremental_df = pd.DataFrame(incremental_results)
        
        return incremental_df
    
    def visualize_results(self, perm_importance, drop_importance, group_importance, incremental_results):
        """Create comprehensive visualization of feature impact analysis."""
        logging.info("\nCreating visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Top 15 Features by Permutation Importance
        ax1 = plt.subplot(2, 3, 1)
        top_perm = perm_importance.head(15)
        ax1.barh(range(len(top_perm)), top_perm['importance_mean'].values)
        ax1.set_yticks(range(len(top_perm)))
        ax1.set_yticklabels(top_perm['feature'].values)
        ax1.set_xlabel('MAE Increase (bpm)')
        ax1.set_title('Top 15 Features - Permutation Importance')
        ax1.invert_yaxis()
        
        # 2. Top 15 Features by Drop-Column Importance
        ax2 = plt.subplot(2, 3, 2)
        top_drop = drop_importance.head(15)
        ax2.barh(range(len(top_drop)), top_drop['mae_increase'].values)
        ax2.set_yticks(range(len(top_drop)))
        ax2.set_yticklabels(top_drop['feature'].values)
        ax2.set_xlabel('MAE Increase (bpm)')
        ax2.set_title('Top 15 Features - Drop-Column Importance')
        ax2.invert_yaxis()
        
        # 3. Feature Group Importance
        ax3 = plt.subplot(2, 3, 3)
        group_sorted = group_importance.sort_values('mae_increase', ascending=True).tail(10)
        ax3.barh(range(len(group_sorted)), group_sorted['mae_increase'].values)
        ax3.set_yticks(range(len(group_sorted)))
        ax3.set_yticklabels([f"{row['group']}\n({row['n_features']} features)" 
                             for _, row in group_sorted.iterrows()])
        ax3.set_xlabel('MAE Increase (bpm)')
        ax3.set_title('Top 10 Feature Groups Impact')
        
        # 4. Incremental Feature Addition
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(incremental_results['n_features'], incremental_results['mae'], 
                marker='o', linewidth=2, markersize=8)
        ax4.axhline(y=self.baseline_mae, color='r', linestyle='--', label='Baseline (all features)')
        ax4.set_xlabel('Number of Features')
        ax4.set_ylabel('MAE (bpm)')
        ax4.set_title('MAE vs Number of Features (by importance)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Feature Importance Distribution
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(perm_importance['importance_mean'].values, bins=30, edgecolor='black')
        ax5.set_xlabel('Permutation Importance (MAE increase)')
        ax5.set_ylabel('Number of Features')
        ax5.set_title('Distribution of Feature Importance')
        ax5.axvline(x=perm_importance['importance_mean'].median(), color='r', 
                   linestyle='--', label=f'Median: {perm_importance["importance_mean"].median():.3f}')
        ax5.legend()
        
        # 6. Group Importance Heatmap
        ax6 = plt.subplot(2, 3, 6)
        group_pivot = group_importance[['group', 'relative_increase']].set_index('group')
        group_pivot_sorted = group_pivot.sort_values('relative_increase', ascending=False).head(12)
        
        # Create color map
        colors = plt.cm.RdYlGn_r(group_pivot_sorted['relative_increase'].values / 
                                 group_pivot_sorted['relative_increase'].max())
        
        bars = ax6.bar(range(len(group_pivot_sorted)), group_pivot_sorted['relative_increase'].values, 
                      color=colors)
        ax6.set_xticks(range(len(group_pivot_sorted)))
        ax6.set_xticklabels(group_pivot_sorted.index, rotation=45, ha='right')
        ax6.set_ylabel('Relative MAE Increase (%)')
        ax6.set_title('Feature Group Relative Importance')
        
        # Add value labels on bars
        for bar, value in zip(bars, group_pivot_sorted['relative_increase'].values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('feature_impact_analysis.png', dpi=300, bbox_inches='tight')
        logging.info("Visualizations saved to feature_impact_analysis.png")
        
        return fig


def main():
    """Main analysis function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("FEATURE IMPACT ANALYSIS FOR RANDOM FOREST MODEL")
    logger.info("="*80)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    parser = GPXParser()
    feature_engineer = FeatureEngineer(
        windows=config['features']['moving_averages']
    )
    
    # Load and process data
    logger.info("\nLoading and processing GPX data...")
    gpx_dir = config['data']['gpx_dir']
    raw_data = parser.parse_directory(gpx_dir)
    
    # Engineer features
    processed_data = {}
    for filename, df in raw_data.items():
        processed_df = feature_engineer.process_dataframe(df, add_derived=True)
        processed_data[filename] = processed_df
    
    # Combine all data
    combined_data = pd.concat(processed_data.values(), ignore_index=True)
    
    # Select features (exclude HR-related features to prevent leakage)
    all_features = [col for col in combined_data.columns if col != 'hr' and not col.startswith('hr')]
    
    # Use basic features as in the best model
    basic_features = ['cad', 'power', 'ele', 'cad5', 'cad10', 'cad30', 'cad60',
                     'power5', 'power10', 'power30', 'power60',
                     'power_lag_1', 'power_lag_5', 'power_lag_10', 'power_lag_15', 'power_lag_30',
                     'cad_lag_1', 'cad_lag_5', 'cad_lag_10', 'cad_lag_15', 'cad_lag_30']
    
    # Add additional derived features if available
    additional_features = [col for col in all_features 
                          if col not in basic_features and 
                          any(x in col for x in ['gradient', 'power_cv', 'cad_std', 'effort', 
                                                 'cumulative', 'intensity', 'torque', 'terrain',
                                                 'zone', 'normalized', 'variability'])]
    
    # Combine features (limit to most relevant)
    feature_cols = basic_features + additional_features[:30]  # Limit to avoid overfitting
    feature_cols = [col for col in feature_cols if col in combined_data.columns]
    
    X = combined_data[feature_cols].fillna(0)
    y = combined_data['hr']
    
    # Remove rows with missing target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Features analyzed: {len(feature_cols)}")
    
    # Best model parameters
    model_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 3,
        'min_samples_leaf': 1
    }
    
    # Initialize analyzer
    analyzer = FeatureImpactAnalyzer(X, y, feature_cols, model_params)
    
    # Calculate baseline
    baseline_mae = analyzer.calculate_baseline_performance()
    
    # Perform analyses
    perm_importance = analyzer.permutation_importance_analysis(n_repeats=5)
    drop_importance = analyzer.drop_column_importance()
    group_importance = analyzer.group_importance_analysis()
    incremental_results = analyzer.incremental_feature_analysis()
    
    # Visualize results
    fig = analyzer.visualize_results(perm_importance, drop_importance, 
                                     group_importance, incremental_results)
    
    # Summary report
    logger.info("\n" + "="*80)
    logger.info("SUMMARY OF FEATURE IMPACT ANALYSIS")
    logger.info("="*80)
    
    logger.info(f"\nBaseline Performance (all {len(feature_cols)} features):")
    logger.info(f"  MAE: {baseline_mae:.4f} bpm")
    
    logger.info("\nMost Important Individual Features:")
    for i, row in enumerate(perm_importance.head(5).iterrows(), 1):
        logger.info(f"  {i}. {row[1]['feature']}: +{row[1]['importance_mean']:.4f} bpm when permuted")
    
    logger.info("\nMost Important Feature Groups:")
    for i, row in enumerate(group_importance.head(5).iterrows(), 1):
        logger.info(f"  {i}. {row[1]['group']} ({row[1]['n_features']} features): "
                   f"+{row[1]['mae_increase']:.4f} bpm ({row[1]['relative_increase']:.1f}% increase)")
    
    logger.info("\nMinimal Feature Set Performance:")
    min_features = incremental_results[incremental_results['mae'] < baseline_mae + 0.5].iloc[0]
    logger.info(f"  {min_features['n_features']} features achieve MAE of {min_features['mae']:.4f} bpm")
    logger.info(f"  (Only {min_features['mae'] - baseline_mae:.4f} bpm worse than all features)")
    
    # Save detailed results
    logger.info("\nSaving detailed results to CSV files...")
    perm_importance.to_csv('feature_importance_permutation.csv', index=False)
    drop_importance.to_csv('feature_importance_drop.csv', index=False)
    group_importance.to_csv('feature_group_importance.csv', index=False)
    incremental_results.to_csv('incremental_feature_results.csv', index=False)
    
    logger.info("\nAnalysis complete! Check the generated files for detailed results.")
    

if __name__ == "__main__":
    main()