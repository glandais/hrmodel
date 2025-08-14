#!/usr/bin/env python3
"""
Fast Feature Impact Analysis for Random Forest Model
Provides key insights without exhaustive computation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_data_from_parquet():
    """Load data from existing parquet files in output/processed_data."""
    parquet_files = list(Path('output/processed_data').glob('*.parquet'))
    
    if not parquet_files:
        raise FileNotFoundError("No parquet files found in output/processed_data/")
    
    logging.info(f"Found {len(parquet_files)} parquet files")
    
    # Load all parquet files
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)
    
    # Combine all data
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sample for faster analysis
    if len(combined_df) > 10000:
        logging.info(f"Dataset has {len(combined_df)} samples, sampling 10000 for analysis...")
        combined_df = combined_df.sample(n=10000, random_state=42)
    
    return combined_df


def main():
    """Main analysis function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("FAST FEATURE IMPACT ANALYSIS FOR RANDOM FOREST MODEL")
    logger.info("="*80)
    
    # Load data
    logger.info("\nLoading data from output/processed_data...")
    combined_data = load_data_from_parquet()
    
    # Define features
    basic_features = ['cad', 'power', 'ele', 'cad5', 'cad10', 'cad30', 'cad60',
                     'power5', 'power10', 'power30', 'power60',
                     'power_lag_1', 'power_lag_5', 'power_lag_10', 'power_lag_15', 'power_lag_30',
                     'cad_lag_1', 'cad_lag_5', 'cad_lag_10', 'cad_lag_15', 'cad_lag_30']
    
    feature_cols = [col for col in basic_features if col in combined_data.columns]
    
    X = combined_data[feature_cols].fillna(0)
    y = combined_data['hr']
    
    # Remove rows with missing target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Best model parameters
    model_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 3,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': 2
    }
    
    # 1. BASELINE MODEL
    logger.info("\n" + "="*60)
    logger.info("1. BASELINE PERFORMANCE (All Features)")
    logger.info("="*60)
    
    model_all = RandomForestRegressor(**model_params)
    model_all.fit(X, y)
    y_pred_all = model_all.predict(X)
    
    mae_all = mean_absolute_error(y, y_pred_all)
    rmse_all = np.sqrt(mean_squared_error(y, y_pred_all))
    r2_all = r2_score(y, y_pred_all)
    
    logger.info(f"MAE:  {mae_all:.4f} bpm")
    logger.info(f"RMSE: {rmse_all:.4f} bpm")
    logger.info(f"R²:   {r2_all:.4f}")
    
    # 2. FEATURE IMPORTANCE FROM MODEL
    logger.info("\n" + "="*60)
    logger.info("2. FEATURE IMPORTANCE (Built-in)")
    logger.info("="*60)
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_all.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:20s}: {row['importance']:.4f}")
    
    # 3. FEATURE GROUP ANALYSIS
    logger.info("\n" + "="*60)
    logger.info("3. FEATURE GROUP ANALYSIS")
    logger.info("="*60)
    
    feature_groups = {
        "Raw Sensors": ['cad', 'power', 'ele'],
        "Cadence MA": ['cad5', 'cad10', 'cad30', 'cad60'],
        "Power MA": ['power5', 'power10', 'power30', 'power60'],
        "Cadence Lags": ['cad_lag_1', 'cad_lag_5', 'cad_lag_10', 'cad_lag_15', 'cad_lag_30'],
        "Power Lags": ['power_lag_1', 'power_lag_5', 'power_lag_10', 'power_lag_15', 'power_lag_30']
    }
    
    group_results = []
    for group_name, group_features in feature_groups.items():
        # Filter to available features
        group_features = [f for f in group_features if f in feature_cols]
        if not group_features:
            continue
            
        # Calculate group importance
        group_importance = importance_df[importance_df['feature'].isin(group_features)]['importance'].sum()
        
        # Train model without this group
        features_without = [f for f in feature_cols if f not in group_features]
        if features_without:
            X_without = X[features_without]
            model_without = RandomForestRegressor(**model_params)
            model_without.fit(X_without, y)
            y_pred_without = model_without.predict(X_without)
            mae_without = mean_absolute_error(y, y_pred_without)
            mae_increase = mae_without - mae_all
        else:
            mae_without = np.nan
            mae_increase = np.nan
        
        group_results.append({
            'group': group_name,
            'n_features': len(group_features),
            'total_importance': group_importance,
            'mae_without': mae_without,
            'mae_increase': mae_increase,
            'relative_increase': (mae_increase / mae_all * 100) if mae_all > 0 else 0
        })
        
        logger.info(f"{group_name:15s} ({len(group_features)} features): "
                   f"Importance={group_importance:.3f}, "
                   f"MAE increase={mae_increase:.3f} bpm ({mae_increase/mae_all*100:.1f}%)")
    
    group_df = pd.DataFrame(group_results)
    
    # 4. INCREMENTAL FEATURE ANALYSIS
    logger.info("\n" + "="*60)
    logger.info("4. INCREMENTAL FEATURE ADDITION")
    logger.info("="*60)
    
    feature_ranking = importance_df['feature'].tolist()
    incremental_results = []
    
    for n_features in [1, 3, 5, 10, 15, len(feature_ranking)]:
        if n_features > len(feature_ranking):
            break
            
        selected_features = feature_ranking[:n_features]
        X_selected = X[selected_features]
        
        model_selected = RandomForestRegressor(**model_params)
        model_selected.fit(X_selected, y)
        y_pred_selected = model_selected.predict(X_selected)
        
        mae = mean_absolute_error(y, y_pred_selected)
        r2 = r2_score(y, y_pred_selected)
        
        incremental_results.append({
            'n_features': n_features,
            'mae': mae,
            'r2': r2
        })
        
        logger.info(f"Top {n_features:2d} features: MAE={mae:.4f} bpm, R²={r2:.4f}")
    
    incremental_df = pd.DataFrame(incremental_results)
    
    # 5. KEY INSIGHTS
    logger.info("\n" + "="*60)
    logger.info("5. KEY INSIGHTS")
    logger.info("="*60)
    
    # Most critical features
    top_features = importance_df.head(5)
    logger.info("\n🎯 Most Critical Individual Features:")
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        logger.info(f"  {i}. {row['feature']} (importance: {row['importance']:.3f})")
    
    # Most critical group
    top_group = group_df.nlargest(1, 'mae_increase').iloc[0]
    logger.info(f"\n📊 Most Critical Feature Group:")
    logger.info(f"  {top_group['group']} - Removing it increases MAE by {top_group['mae_increase']:.3f} bpm")
    
    # Minimal features for good performance
    good_perf = incremental_df[incremental_df['mae'] < mae_all * 1.2]
    if not good_perf.empty:
        min_features = good_perf.iloc[0]
        logger.info(f"\n⚡ Efficiency:")
        logger.info(f"  Only {min_features['n_features']} features needed for MAE < {min_features['mae']:.3f} bpm")
        logger.info(f"  (Within 20% of best performance using all {len(feature_cols)} features)")
    
    # Feature groups by importance
    logger.info("\n📈 Feature Groups Ranked by Impact:")
    for i, row in group_df.nlargest(5, 'mae_increase').iterrows():
        logger.info(f"  {row['group']:15s}: +{row['mae_increase']:.3f} bpm MAE without it")
    
    # 6. VISUALIZATION
    logger.info("\n" + "="*60)
    logger.info("6. CREATING VISUALIZATIONS")
    logger.info("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Top 15 feature importances
    ax1 = axes[0, 0]
    top_15 = importance_df.head(15)
    ax1.barh(range(len(top_15)), top_15['importance'].values, color='steelblue')
    ax1.set_yticks(range(len(top_15)))
    ax1.set_yticklabels(top_15['feature'].values)
    ax1.set_xlabel('Feature Importance')
    ax1.set_title('Top 15 Features by Importance')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Feature group impacts
    ax2 = axes[0, 1]
    group_sorted = group_df.sort_values('mae_increase')
    colors = plt.cm.RdYlGn_r(group_sorted['mae_increase'].values / group_sorted['mae_increase'].max())
    ax2.barh(range(len(group_sorted)), group_sorted['mae_increase'].values, color=colors)
    ax2.set_yticks(range(len(group_sorted)))
    ax2.set_yticklabels([f"{row['group']}\n({row['n_features']} feat.)" 
                         for _, row in group_sorted.iterrows()])
    ax2.set_xlabel('MAE Increase (bpm)')
    ax2.set_title('Feature Group Impact (MAE increase when removed)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Incremental feature performance
    ax3 = axes[1, 0]
    ax3.plot(incremental_df['n_features'], incremental_df['mae'], 
            marker='o', linewidth=2, markersize=8, color='darkgreen')
    ax3.axhline(y=mae_all, color='r', linestyle='--', 
               label=f'All features ({mae_all:.3f} bpm)')
    ax3.axhline(y=2.0, color='orange', linestyle=':', label='Target (2.0 bpm)')
    ax3.set_xlabel('Number of Features')
    ax3.set_ylabel('MAE (bpm)')
    ax3.set_title('MAE vs Number of Features')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Group importance pie chart
    ax4 = axes[1, 1]
    group_df_sorted = group_df.sort_values('total_importance', ascending=False)
    wedges, texts, autotexts = ax4.pie(
        group_df_sorted['total_importance'].values,
        labels=group_df_sorted['group'].values,
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Set3(range(len(group_df_sorted)))
    )
    ax4.set_title('Feature Group Contribution to Model')
    
    plt.suptitle('Random Forest Feature Impact Analysis - Fast Mode', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('rf_feature_analysis_fast.png', dpi=150, bbox_inches='tight')
    logger.info("Saved visualization to rf_feature_analysis_fast.png")
    
    # Save results
    importance_df.to_csv('rf_feature_importances.csv', index=False)
    group_df.to_csv('rf_group_importances.csv', index=False)
    incremental_df.to_csv('rf_incremental_performance.csv', index=False)
    
    logger.info("\n✅ Analysis complete! Results saved to:")
    logger.info("  • rf_feature_analysis_fast.png")
    logger.info("  • rf_feature_importances.csv")
    logger.info("  • rf_group_importances.csv")
    logger.info("  • rf_incremental_performance.csv")


if __name__ == "__main__":
    main()