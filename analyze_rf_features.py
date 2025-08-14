#!/usr/bin/env python3
"""
Feature Impact Analysis for Random Forest Model
Analyzes how different feature groups affect MAE performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class FeatureImpactAnalyzer:
    """Analyzes feature impact on Random Forest MAE performance."""
    
    def __init__(self, X, y, feature_names, model_params):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.model_params = model_params
        self.baseline_mae = None
        
        # Define feature groups
        self.feature_groups = self._define_feature_groups()
        
    def _define_feature_groups(self) -> Dict[str, List[str]]:
        """Define logical feature groups based on available features."""
        groups = {
            "Raw Sensors": [],
            "Cadence Features": [],
            "Power Features": [],
            "Moving Averages - Cadence": [],
            "Moving Averages - Power": [],
            "Lag Features - Cadence": [],
            "Lag Features - Power": [],
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
                
            # Generic cadence
            elif 'cad' in feature:
                groups["Cadence Features"].append(feature)
                
            # Generic power
            elif 'power' in feature:
                groups["Power Features"].append(feature)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        
        return groups
    
    def calculate_baseline_performance(self) -> float:
        """Calculate baseline MAE with all features."""
        logging.info("Calculating baseline performance with all features...")
        
        model = RandomForestRegressor(**self.model_params, random_state=42, n_jobs=2)
        model.fit(self.X, self.y)
        
        y_pred = model.predict(self.X)
        self.baseline_mae = mean_absolute_error(self.y, y_pred)
        self.baseline_rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        self.baseline_r2 = r2_score(self.y, y_pred)
        
        logging.info(f"Baseline MAE: {self.baseline_mae:.4f} bpm")
        logging.info(f"Baseline RMSE: {self.baseline_rmse:.4f} bpm")
        logging.info(f"Baseline R²: {self.baseline_r2:.4f}")
        
        # Store feature importances
        self.baseline_importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.model = model
        
        return self.baseline_mae
    
    def permutation_importance_analysis(self, n_repeats=10) -> pd.DataFrame:
        """Perform permutation importance analysis."""
        logging.info("\nPerforming permutation importance analysis...")
        
        # Calculate permutation importance (reduced parallelization to avoid memory issues)
        result = permutation_importance(
            self.model, self.X, self.y,
            n_repeats=n_repeats,
            random_state=42,
            scoring='neg_mean_absolute_error',
            n_jobs=1  # Disable parallelization to avoid memory issues
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
        logging.info("\nPerforming drop-column importance analysis...")
        
        results = []
        
        for i, feature in enumerate(self.feature_names):
            if (i + 1) % 5 == 0:
                logging.info(f"  Processing feature {i+1}/{len(self.feature_names)}...")
            
            # Create feature set without current feature
            X_dropped = self.X.drop(columns=[feature])
            
            # Train model without feature
            model = RandomForestRegressor(**self.model_params, random_state=42, n_jobs=2)
            model.fit(X_dropped, self.y)
            
            # Calculate MAE
            y_pred = model.predict(X_dropped)
            mae_without = mean_absolute_error(self.y, y_pred)
            
            # Calculate MAE increase
            mae_increase = mae_without - self.baseline_mae
            
            results.append({
                'feature': feature,
                'mae_without': mae_without,
                'mae_increase': mae_increase,
                'relative_increase': (mae_increase / self.baseline_mae) * 100
            })
        
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
            model = RandomForestRegressor(**self.model_params, random_state=42, n_jobs=2)
            model.fit(X_without_group, self.y)
            
            # Calculate MAE
            y_pred = model.predict(X_without_group)
            mae_without = mean_absolute_error(self.y, y_pred)
            
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
            model = RandomForestRegressor(**self.model_params, random_state=42, n_jobs=2)
            model.fit(X_selected, self.y)
            
            # Calculate metrics
            y_pred = model.predict(X_selected)
            mae = mean_absolute_error(self.y, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y, y_pred))
            r2 = r2_score(self.y, y_pred)
            
            incremental_results.append({
                'n_features': n_features,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'top_features': ', '.join(selected_features[:3]) + ('...' if n_features > 3 else '')
            })
            
            logging.info(f"  Top {n_features} features: MAE = {mae:.4f}, R² = {r2:.4f}")
        
        incremental_df = pd.DataFrame(incremental_results)
        
        return incremental_df
    
    def visualize_results(self, perm_importance, drop_importance, group_importance, incremental_results):
        """Create comprehensive visualization of feature impact analysis."""
        logging.info("\nCreating visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Top 15 Features by Permutation Importance
        ax1 = plt.subplot(2, 3, 1)
        top_perm = perm_importance.head(15)
        bars1 = ax1.barh(range(len(top_perm)), top_perm['importance_mean'].values, color='steelblue')
        ax1.set_yticks(range(len(top_perm)))
        ax1.set_yticklabels(top_perm['feature'].values, fontsize=9)
        ax1.set_xlabel('MAE Increase (bpm)')
        ax1.set_title('Top 15 Features - Permutation Importance', fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        
        # 2. Top 15 Features by Drop-Column Importance
        ax2 = plt.subplot(2, 3, 2)
        top_drop = drop_importance.head(15)
        bars2 = ax2.barh(range(len(top_drop)), top_drop['mae_increase'].values, color='coral')
        ax2.set_yticks(range(len(top_drop)))
        ax2.set_yticklabels(top_drop['feature'].values, fontsize=9)
        ax2.set_xlabel('MAE Increase (bpm)')
        ax2.set_title('Top 15 Features - Drop-Column Importance', fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature Group Importance
        ax3 = plt.subplot(2, 3, 3)
        group_sorted = group_importance.sort_values('mae_increase', ascending=True).tail(10)
        colors = plt.cm.RdYlGn_r(group_sorted['mae_increase'].values / group_sorted['mae_increase'].max())
        bars3 = ax3.barh(range(len(group_sorted)), group_sorted['mae_increase'].values, color=colors)
        ax3.set_yticks(range(len(group_sorted)))
        ax3.set_yticklabels([f"{row['group']}\n({row['n_features']} features)" 
                             for _, row in group_sorted.iterrows()], fontsize=9)
        ax3.set_xlabel('MAE Increase (bpm)')
        ax3.set_title('Feature Groups Impact on MAE', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Incremental Feature Addition
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(incremental_results['n_features'], incremental_results['mae'], 
                marker='o', linewidth=2, markersize=8, color='darkgreen', label='MAE')
        ax4.axhline(y=self.baseline_mae, color='r', linestyle='--', label=f'Baseline ({len(self.feature_names)} features)')
        ax4.axhline(y=2.0, color='orange', linestyle=':', label='Target (2.0 bpm)')
        ax4.set_xlabel('Number of Features')
        ax4.set_ylabel('MAE (bpm)')
        ax4.set_title('MAE vs Number of Features', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Feature Importance Distribution
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(perm_importance['importance_mean'].values, bins=30, edgecolor='black', color='skyblue')
        ax5.set_xlabel('Permutation Importance (MAE increase)')
        ax5.set_ylabel('Number of Features')
        ax5.set_title('Distribution of Feature Importance', fontweight='bold')
        median_val = perm_importance['importance_mean'].median()
        ax5.axvline(x=median_val, color='r', linestyle='--', label=f'Median: {median_val:.3f}')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. R² Performance with Feature Addition
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(incremental_results['n_features'], incremental_results['r2'], 
                marker='s', linewidth=2, markersize=8, color='purple')
        ax6.axhline(y=self.baseline_r2, color='r', linestyle='--', label=f'Baseline R² ({self.baseline_r2:.4f})')
        ax6.axhline(y=0.95, color='green', linestyle=':', label='Target (0.95)')
        ax6.set_xlabel('Number of Features')
        ax6.set_ylabel('R² Score')
        ax6.set_title('R² Score vs Number of Features', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Random Forest Feature Impact Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('rf_feature_impact_analysis.png', dpi=300, bbox_inches='tight')
        logging.info("Visualizations saved to rf_feature_impact_analysis.png")
        
        return fig


def load_data_from_parquet():
    """Load data from existing parquet files in output/processed_data."""
    # Find available parquet files in the correct location
    parquet_files = list(Path('output/processed_data').glob('*.parquet'))
    
    if not parquet_files:
        raise FileNotFoundError("No parquet files found in output/processed_data/")
    
    logging.info(f"Found {len(parquet_files)} parquet files in output/processed_data/")
    
    # Load all parquet files
    dfs = []
    for file in parquet_files:
        logging.info(f"  Loading {file.name}...")
        df = pd.read_parquet(file)
        dfs.append(df)
    
    # Combine all data
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sample if dataset is too large for faster analysis
    if len(combined_df) > 20000:
        logging.info(f"Dataset has {len(combined_df)} samples, sampling 20000 for faster analysis...")
        combined_df = combined_df.sample(n=20000, random_state=42)
    
    return combined_df


def main():
    """Main analysis function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("FEATURE IMPACT ANALYSIS FOR RANDOM FOREST MODEL")
    logger.info("="*80)
    
    try:
        # Load data from parquet files
        logger.info("\nLoading data from parquet files...")
        combined_data = load_data_from_parquet()
        logger.info(f"Loaded {len(combined_data)} samples")
        
    except FileNotFoundError:
        logger.warning("No parquet files found. Creating sample data for demonstration...")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 10000
        
        # Create synthetic features
        combined_data = pd.DataFrame({
            'hr': 120 + np.random.randn(n_samples) * 20,
            'cad': 80 + np.random.randn(n_samples) * 10,
            'power': 200 + np.random.randn(n_samples) * 50,
            'ele': 100 + np.random.randn(n_samples) * 20,
        })
        
        # Add moving averages
        for window in [5, 10, 30, 60]:
            combined_data[f'cad{window}'] = combined_data['cad'].rolling(window, min_periods=1).mean()
            combined_data[f'power{window}'] = combined_data['power'].rolling(window, min_periods=1).mean()
        
        # Add lag features
        for lag in [1, 5, 10, 15, 30]:
            combined_data[f'cad_lag_{lag}'] = combined_data['cad'].shift(lag)
            combined_data[f'power_lag_{lag}'] = combined_data['power'].shift(lag)
        
        # Fill NaN values
        combined_data = combined_data.fillna(method='bfill').fillna(method='ffill')
        
        # Make HR dependent on features
        combined_data['hr'] = (
            60 + 
            0.5 * combined_data['cad'] + 
            0.15 * combined_data['power'] + 
            0.05 * combined_data['ele'] +
            np.random.randn(n_samples) * 5
        )
    
    # Define features to use
    basic_features = ['cad', 'power', 'ele', 'cad5', 'cad10', 'cad30', 'cad60',
                     'power5', 'power10', 'power30', 'power60',
                     'power_lag_1', 'power_lag_5', 'power_lag_10', 'power_lag_15', 'power_lag_30',
                     'cad_lag_1', 'cad_lag_5', 'cad_lag_10', 'cad_lag_15', 'cad_lag_30']
    
    # Select available features
    feature_cols = [col for col in basic_features if col in combined_data.columns]
    
    # If we don't have all features, use what we have
    if len(feature_cols) < len(basic_features):
        logger.warning(f"Only {len(feature_cols)}/{len(basic_features)} features available")
        feature_cols = [col for col in combined_data.columns if col != 'hr' and not col.startswith('hr')][:21]
    
    X = combined_data[feature_cols].fillna(0)
    y = combined_data['hr']
    
    # Remove rows with missing target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Features analyzed: {feature_cols}")
    
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
    
    logger.info(f"\n📊 Baseline Performance (all {len(feature_cols)} features):")
    logger.info(f"  • MAE: {baseline_mae:.4f} bpm")
    logger.info(f"  • RMSE: {analyzer.baseline_rmse:.4f} bpm")
    logger.info(f"  • R²: {analyzer.baseline_r2:.4f}")
    
    logger.info("\n🎯 Most Important Individual Features (Permutation):")
    for i, row in enumerate(perm_importance.head(5).iterrows(), 1):
        logger.info(f"  {i}. {row[1]['feature']}: +{row[1]['importance_mean']:.4f} bpm MAE when permuted")
    
    logger.info("\n🔧 Most Important Feature Groups:")
    for i, row in enumerate(group_importance.head(5).iterrows(), 1):
        logger.info(f"  {i}. {row[1]['group']} ({row[1]['n_features']} features): "
                   f"+{row[1]['mae_increase']:.4f} bpm ({row[1]['relative_increase']:.1f}% increase)")
    
    logger.info("\n📈 Performance with Minimal Features:")
    min_features = incremental_results[incremental_results['mae'] < baseline_mae + 0.5].iloc[0]
    logger.info(f"  • {min_features['n_features']} features achieve MAE of {min_features['mae']:.4f} bpm")
    logger.info(f"  • Only {min_features['mae'] - baseline_mae:.4f} bpm worse than all features")
    logger.info(f"  • R² = {min_features['r2']:.4f}")
    
    # Key insights
    logger.info("\n💡 KEY INSIGHTS:")
    
    # 1. Feature efficiency
    optimal_features = incremental_results[incremental_results['mae'] < 2.0]
    if not optimal_features.empty:
        min_for_target = optimal_features.iloc[0]['n_features']
        logger.info(f"  • Only {min_for_target} features needed to achieve <2.0 bpm MAE target")
    
    # 2. Feature group ranking
    top_group = group_importance.iloc[0]
    logger.info(f"  • {top_group['group']} is the most critical feature group "
               f"(+{top_group['mae_increase']:.2f} bpm without it)")
    
    # 3. Diminishing returns
    if len(incremental_results) > 3:
        mae_5 = incremental_results[incremental_results['n_features'] == 5].iloc[0]['mae']
        mae_10 = incremental_results[incremental_results['n_features'] == 10].iloc[0]['mae']
        improvement = mae_5 - mae_10
        logger.info(f"  • Diminishing returns: features 6-10 only improve MAE by {improvement:.3f} bpm")
    
    # Save detailed results
    logger.info("\n💾 Saving detailed results to CSV files...")
    perm_importance.to_csv('rf_feature_importance_permutation.csv', index=False)
    drop_importance.to_csv('rf_feature_importance_drop.csv', index=False)
    group_importance.to_csv('rf_feature_group_importance.csv', index=False)
    incremental_results.to_csv('rf_incremental_feature_results.csv', index=False)
    
    # Save baseline importances
    analyzer.baseline_importances.to_csv('rf_baseline_feature_importances.csv', index=False)
    
    logger.info("\n✅ Analysis complete! Check the generated files for detailed results.")
    logger.info("  • rf_feature_impact_analysis.png - Visual summary")
    logger.info("  • CSV files - Detailed numerical results")
    

if __name__ == "__main__":
    main()