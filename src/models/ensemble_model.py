from .base_model import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

class EnsembleModel(BaseModel):
    def __init__(self, feature_columns=None, 
                 # Random Forest params
                 rf_n_estimators=200, rf_max_depth=15, rf_min_samples_split=5,
                 rf_min_samples_leaf=2, rf_max_features='sqrt',
                 # XGBoost params  
                 xgb_n_estimators=150, xgb_max_depth=8, xgb_learning_rate=0.1,
                 xgb_subsample=0.8, xgb_colsample_bytree=0.8,
                 # Ensemble params
                 rf_weight=0.7, xgb_weight=0.3):
        
        super().__init__(feature_columns)
        
        # Initialize base models with optimized hyperparameters
        self.rf_model = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            max_features=rf_max_features,
            random_state=42,
            n_jobs=-1
        )
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=xgb_n_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=xgb_subsample,
            colsample_bytree=xgb_colsample_bytree,
            random_state=42,
            n_jobs=-1
        )
        
        # Ensemble weights (should sum to 1.0)
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        
        # Normalize weights
        total_weight = self.rf_weight + self.xgb_weight
        self.rf_weight /= total_weight
        self.xgb_weight /= total_weight
        
    def fit(self, X, y, output_dir=None):
        if self.feature_columns:
            X = X[self.feature_columns]
        
        print(f"Training ensemble with {X.shape[1]} features on {X.shape[0]} samples")
        print(f"Weights: RF={self.rf_weight:.2f}, XGB={self.xgb_weight:.2f}")
        
        # Train both models
        print("Training Random Forest...")
        self.rf_model.fit(X, y)
        
        print("Training XGBoost...")
        self.xgb_model.fit(X, y)
        
        # Get individual predictions for validation
        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        ensemble_pred = self.rf_weight * rf_pred + self.xgb_weight * xgb_pred
        
        # Calculate individual model performance
        rf_mae = mean_absolute_error(y, rf_pred)
        xgb_mae = mean_absolute_error(y, xgb_pred)
        ensemble_mae = mean_absolute_error(y, ensemble_pred)
        
        print(f"Training MAE - RF: {rf_mae:.3f}, XGB: {xgb_mae:.3f}, Ensemble: {ensemble_mae:.3f}")
        
        # Save models if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            joblib.dump(self.rf_model, os.path.join(output_dir, 'rf_model.pkl'))
            joblib.dump(self.xgb_model, os.path.join(output_dir, 'xgb_model.pkl'))
            joblib.dump({'rf_weight': self.rf_weight, 'xgb_weight': self.xgb_weight}, 
                       os.path.join(output_dir, 'ensemble_weights.pkl'))
    
    def predict(self, X):
        if self.feature_columns:
            X = X[self.feature_columns]
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        
        # Weighted ensemble prediction
        ensemble_pred = self.rf_weight * rf_pred + self.xgb_weight * xgb_pred
        
        return ensemble_pred
    
    def get_feature_importance(self):
        if not self.feature_columns:
            return None
        
        # Get feature importance from both models
        rf_importance = self.rf_model.feature_importances_
        
        # XGBoost feature importance
        xgb_importance = self.xgb_model.feature_importances_
        
        # Weighted average of feature importances
        ensemble_importance = (self.rf_weight * rf_importance + 
                              self.xgb_weight * xgb_importance)
        
        # Return DataFrame matching other models' format
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': ensemble_importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        return importance_df
    
    def create_feature_importance_plot(self, output_dir, top_n=20):
        importance = self.get_feature_importance()
        if importance is None:
            return
        
        # Create comparison plot with individual model importances
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
        
        # Random Forest importance
        rf_importance = pd.Series(self.rf_model.feature_importances_, 
                                 index=self.feature_columns).sort_values(ascending=False)
        rf_importance.head(top_n).plot(kind='barh', ax=ax1, color='lightblue')
        ax1.set_title(f'Random Forest Feature Importance (Weight: {self.rf_weight:.2f})')
        ax1.set_xlabel('Importance')
        
        # XGBoost importance
        xgb_importance = pd.Series(self.xgb_model.feature_importances_, 
                                  index=self.feature_columns).sort_values(ascending=False)
        xgb_importance.head(top_n).plot(kind='barh', ax=ax2, color='lightcoral')
        ax2.set_title(f'XGBoost Feature Importance (Weight: {self.xgb_weight:.2f})')
        ax2.set_xlabel('Importance')
        
        # Ensemble importance
        importance.head(top_n).plot(kind='barh', ax=ax3, color='lightgreen')
        ax3.set_title('Ensemble Feature Importance (Weighted Average)')
        ax3.set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save importance scores
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'rf_importance': self.rf_model.feature_importances_,
            'xgb_importance': self.xgb_model.feature_importances_,
            'ensemble_importance': importance.values
        }).sort_values('ensemble_importance', ascending=False)
        
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance_comparison.csv'), index=False)
        
        return importance
    
    def train(self, X, y):
        """Train method to comply with BaseModel interface."""
        self.fit(X, y)