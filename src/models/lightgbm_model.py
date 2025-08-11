import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")


class LightGBMModel(BaseModel):
    """LightGBM regression model for HR prediction.
    
    LightGBM is a gradient boosting framework that uses tree-based learning algorithms.
    It is designed to be distributed and efficient with the following advantages:
    - Faster training speed and higher efficiency
    - Lower memory usage
    - Better accuracy than XGBoost in many cases
    - Support for parallel and GPU learning
    - Capable of handling large-scale data
    """
    
    def __init__(self, feature_columns=None, 
                 n_estimators=200, max_depth=-1, num_leaves=31,
                 learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                 min_child_samples=20, reg_alpha=0.0, reg_lambda=0.0,
                 normalize=False, random_state=42):
        """Initialize LightGBM model.
        
        Args:
            feature_columns: List of feature column names
            n_estimators: Number of boosting iterations
            max_depth: Maximum tree depth (-1 means no limit)
            num_leaves: Maximum number of leaves in one tree (should be <= 2^max_depth)
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of the training instance
            colsample_bytree: Subsample ratio of columns when constructing each tree
            min_child_samples: Minimum number of data needed in a child (leaf)
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            normalize: Whether to normalize features
            random_state: Random seed for reproducibility
        """
        super().__init__("LightGBM", feature_columns)
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.normalize = normalize
        
        self.scaler = StandardScaler() if normalize else None
        
        # LightGBM parameters
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,  # Suppress LightGBM output
            objective='regression',
            metric='mae',  # Use MAE as primary metric
            boosting_type='gbdt',  # Gradient Boosting Decision Tree
            subsample_freq=1  # Frequency of subsample
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the LightGBM model."""
        X_prepared = self.prepare_features(X)
        
        # Remove rows with NaN in target
        mask = ~y.isna()
        X_clean = X_prepared[mask]
        y_clean = y[mask]
        
        if self.normalize and self.scaler:
            X_clean = self.scaler.fit_transform(X_clean)
        
        # Train LightGBM model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress LightGBM warnings
            self.model.fit(
                X_clean, y_clean,
                callbacks=[lgb.log_evaluation(0)]  # Suppress iteration output
            )
        
        self.is_trained = True
        
        # Store metrics
        self.metrics['training_samples'] = len(X_clean)
        self.metrics['n_features'] = X_clean.shape[1]
        self.metrics['n_estimators'] = self.n_estimators
        self.metrics['num_leaves'] = self.num_leaves
        self.metrics['max_depth'] = self.max_depth
        self.metrics['learning_rate'] = self.learning_rate
        self.metrics['subsample'] = self.subsample
        self.metrics['colsample_bytree'] = self.colsample_bytree
        self.metrics['min_child_samples'] = self.min_child_samples
        
        # Evaluate on training data
        train_metrics = self.evaluate(X[mask], y[mask])
        self.metrics.update({f'train_{k}': v for k, v in train_metrics.items()})
        
        print(f"LightGBM model trained on {len(X_clean)} samples with {X_clean.shape[1]} features")
        print(f"Trees: {self.n_estimators}, Leaves: {self.num_leaves}, LR: {self.learning_rate}")
        print(f"Training R²: {self.metrics['train_r2']:.4f}, MAE: {self.metrics['train_mae']:.2f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the LightGBM model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_prepared = self.prepare_features(X)
        
        if self.normalize and self.scaler:
            X_prepared = self.scaler.transform(X_prepared)
        
        return self.model.predict(X_prepared)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the LightGBM model."""
        if not self.is_trained:
            return None
        
        # Get feature importance (split-based and gain-based)
        importance_split = self.model.feature_importances_  # Default is 'split'
        importance_gain = self.model.booster_.feature_importance(importance_type='gain')
        
        # Normalize gain importance to sum to 1 for comparison
        if importance_gain.sum() > 0:
            importance_gain = importance_gain / importance_gain.sum()
        
        # Create importance dataframe
        feature_names = self.feature_columns[:len(importance_split)]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_split': importance_split,
            'importance_gain': importance_gain,
            'importance': importance_split  # Use split as primary for compatibility
        })
        
        # Sort by split importance and add derived columns
        importance_df = importance_df.sort_values('importance_split', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        importance_df['cumulative_importance'] = importance_df['importance_split'].cumsum()
        
        # Calculate percentage importance
        total_importance = importance_df['importance_split'].sum()
        if total_importance > 0:
            importance_df['importance_pct'] = (importance_df['importance_split'] / total_importance * 100)
        
        return importance_df
    
    def get_model_info(self) -> dict:
        """Get detailed model information."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "model_type": "LightGBM",
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "n_features": len(self.feature_columns),
            "training_samples": self.metrics.get('training_samples', 0)
        }
        
        # Get actual number of trees from booster
        if hasattr(self.model, 'booster_'):
            info['actual_trees'] = self.model.booster_.num_trees()
            info['actual_iterations'] = self.model.booster_.current_iteration()
        
        return info
    
    def plot_tree(self, tree_index=0, **kwargs):
        """Plot a specific tree from the model.
        
        Args:
            tree_index: Index of the tree to plot (0-based)
            **kwargs: Additional arguments for lgb.plot_tree
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting trees")
        
        try:
            import matplotlib.pyplot as plt
            lgb.plot_tree(self.model, tree_index=tree_index, **kwargs)
            plt.show()
        except ImportError:
            print("Matplotlib is required for plotting trees")
    
    def plot_importance(self, importance_type='split', max_features=20, **kwargs):
        """Plot feature importance.
        
        Args:
            importance_type: Type of importance ('split' or 'gain')
            max_features: Maximum number of features to display
            **kwargs: Additional arguments for lgb.plot_importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting importance")
        
        try:
            import matplotlib.pyplot as plt
            lgb.plot_importance(
                self.model, 
                importance_type=importance_type,
                max_num_features=max_features,
                **kwargs
            )
            plt.show()
        except ImportError:
            print("Matplotlib is required for plotting importance")