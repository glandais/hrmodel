"""
Multi-Layer Perceptron (MLP) model for heart rate prediction.

This model uses TensorFlow/Keras to create a deep neural network for predicting
heart rate from cycling sensor data. The architecture is designed to capture
non-linear relationships and temporal dependencies in the data.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import joblib
import os
import optuna
from .base_model import BaseModel

class MLPModel(BaseModel):
    """Multi-Layer Perceptron model for heart rate prediction."""
    
    def __init__(self, config, feature_columns=None):
        super().__init__("mlp", feature_columns)
        self.config = config  # Store config separately
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Configure TensorFlow for better performance
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    
    def _build_model(self, input_dim):
        """Build the MLP architecture using current config parameters."""
        hidden_layers = self.config.get('hidden_layers', [64, 32, 16])
        dropout = self.config.get('dropout', 0.2)
        l2_reg = self.config.get('l2_reg', 0.001)
        
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(input_dim,)))
        
        # Add hidden layers dynamically based on config
        for i, layer_size in enumerate(hidden_layers):
            model.add(keras.layers.Dense(
                layer_size,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            ))
            
            # Add batch normalization for all but last hidden layer
            if i < len(hidden_layers) - 1:
                model.add(keras.layers.BatchNormalization())
            
            # Add dropout
            model.add(keras.layers.Dropout(dropout))
        
        # Output layer - single HR prediction
        model.add(keras.layers.Dense(1, activation='linear'))
        
        # Compile with adaptive learning rate
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config.get('learning_rate', 0.001),
            beta_1=0.9,
            beta_2=0.999
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean squared error for regression
            metrics=['mae']  # Track MAE during training
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the MLP model."""
        print(f"Training MLP model with {X_train.shape[0]} samples, {X_train.shape[1]} features...")
        
        # Prepare features using BaseModel method
        X_train_prepared = self.prepare_features(X_train)
        
        # Remove rows with NaN in target (follow BaseModel pattern)
        mask = ~y_train.isna()
        X_train_clean = X_train_prepared[mask]
        y_train_clean = y_train[mask]
        
        # Scale features (critical for neural networks)
        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            # Use 20% of training data for validation
            val_split = 0.2
            validation_data = None
        
        # Build model
        self.model = self._build_model(X_train_clean.shape[1])
        
        # Print model summary
        print("MLP Model Architecture:")
        self.model.summary()
        
        # Define callbacks
        callbacks = []
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'val_loss',
            patience=self.config.get('patience', 15),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction on plateau
        lr_reducer = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        # Train the model with (possibly tuned) parameters
        epochs = self.config.get('max_epochs', self.config.get('epochs', 100))
        batch_size = self.config.get('batch_size', 256)
        
        self.history = self.model.fit(
            X_train_scaled, y_train_clean,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split if validation_data is None else None,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Mark as trained
        self.is_trained = True
        
        # Store metrics
        self.metrics['training_samples'] = len(X_train_clean)
        self.metrics['n_features'] = X_train_clean.shape[1]
        
        print("MLP training completed!")
        return self
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features using BaseModel method
        X_prepared = self.prepare_features(X)
        X_scaled = self.scaler.transform(X_prepared)
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()  # Return 1D array
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
            'bias': np.mean(y_pred - y_test),
        }
        
        return metrics
    
    def save(self, filepath):
        """Save the trained model and scaler (overrides BaseModel.save)."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the keras model
        model_path = str(filepath).replace('.joblib', '_keras_model.keras')
        if self.model:
            self.model.save(model_path)
        
        # Save scaler and metadata (compatible with BaseModel structure)
        model_data = {
            'name': self.name,
            'model_type': 'mlp',
            'scaler': self.scaler,
            'config': self.config,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'is_trained': self.is_trained,
            'history': self.history.history if self.history else None,
            'keras_model_path': model_path
        }
        
        joblib.dump(model_data, filepath)
        print(f"MLP model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and scaler."""
        model_data = joblib.load(filepath)
        
        self.scaler = model_data['scaler']
        self.config.update(model_data.get('config', {}))
        
        # Load keras model
        keras_model_path = model_data.get('keras_model_path')
        if keras_model_path and os.path.exists(keras_model_path):
            self.model = keras.models.load_model(keras_model_path)
            print(f"MLP model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Keras model file not found: {keras_model_path}")
    
    def get_feature_importance(self):
        """
        Get feature importance for MLP.
        
        Since neural networks don't have direct feature importance,
        we use permutation importance as an approximation.
        """
        if self.model is None:
            return {}
        
        try:
            # For neural networks, we can analyze the first layer weights
            # as a rough approximation of feature importance
            first_layer = self.model.layers[0]
            if hasattr(first_layer, 'get_weights'):
                weights = first_layer.get_weights()[0]  # Input weights
                
                # Calculate average absolute weight for each input feature
                feature_importance = np.mean(np.abs(weights), axis=1)
                
                # Normalize to sum to 1
                feature_importance = feature_importance / np.sum(feature_importance)
                
                return {f'feature_{i}': importance 
                       for i, importance in enumerate(feature_importance)}
            else:
                return {}
        except Exception as e:
            print(f"Warning: Could not extract feature importance: {e}")
            return {}
    
    def plot_training_history(self, save_path=None):
        """Plot training history if available."""
        if not self.history:
            print("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Model Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (bpm)')
        ax2.set_title('Model MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def tune_hyperparameters(self, X, y, n_trials=50, cv_folds=5):
        """
        Tune hyperparameters using Optuna (Bayesian optimization).
        
        This method systematically searches for the best combination of:
        - Network architecture (layer sizes, depth)
        - Learning parameters (learning rate, batch size) 
        - Regularization (dropout, L2 penalty)
        - Training settings (epochs, patience)
        
        Target: Beat Random Forest's 1.17 bpm MAE
        """
        print(f"🔧 Starting MLP hyperparameter tuning with {n_trials} trials...")
        print("🎯 Target: < 1.17 bpm MAE (beat Random Forest)")
        
        # Prepare data
        X_prepared = self.prepare_features(X)
        mask = ~y.isna()
        X_clean = X_prepared[mask].values
        y_clean = y[mask].values
        
        print(f"📊 Training data: {len(X_clean)} samples, {X_clean.shape[1]} features")
        
        def objective(trial):
            """Optuna objective function - what we're trying to minimize."""
            
            # 1. ARCHITECTURE TUNING
            # Number of hidden layers (1-4 layers for balance of complexity vs speed)
            n_layers = trial.suggest_int('n_layers', 1, 4)
            
            # Hidden layer sizes (powers of 2 for efficient computation)
            hidden_layers = []
            for i in range(n_layers):
                # Decreasing size: first layer largest, subsequent layers smaller
                max_size = max(16, 128 // (2**i))  # 128, 64, 32, 16
                size = trial.suggest_categorical(f'layer_{i}_size', [16, 32, 64, 128, 256])
                size = min(size, max_size)  # Ensure decreasing pattern
                hidden_layers.append(size)
            
            # 2. LEARNING DYNAMICS TUNING  
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  # Log scale
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
            
            # 3. REGULARIZATION TUNING
            dropout = trial.suggest_float('dropout', 0.0, 0.5)  # 0-50% dropout
            l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True)  # L2 penalty
            
            # 4. TRAINING TUNING
            max_epochs = trial.suggest_int('max_epochs', 50, 200)
            patience = trial.suggest_int('patience', 10, 30)
            
            # Cross-validation to get robust performance estimate
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_clean)):
                X_train_fold, X_val_fold = X_clean[train_idx], X_clean[val_idx]
                y_train_fold, y_val_fold = y_clean[train_idx], y_clean[val_idx]
                
                # Scale features for this fold
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)
                
                # Build model with trial parameters
                model = keras.Sequential()
                model.add(keras.layers.Input(shape=(X_clean.shape[1],)))
                
                # Add hidden layers with trial-suggested architecture
                for i, layer_size in enumerate(hidden_layers):
                    model.add(keras.layers.Dense(
                        layer_size, 
                        activation='relu',
                        kernel_regularizer=keras.regularizers.l2(l2_reg)
                    ))
                    if i < len(hidden_layers) - 1:  # Batch norm except last layer
                        model.add(keras.layers.BatchNormalization())
                    model.add(keras.layers.Dropout(dropout))
                
                # Output layer
                model.add(keras.layers.Dense(1, activation='linear'))
                
                # Compile with trial parameters
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='mse',
                    metrics=['mae']
                )
                
                # Train with early stopping
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=0
                )
                
                # Suppress training output for cleaner tuning logs
                history = model.fit(
                    X_train_scaled, y_train_fold,
                    epochs=max_epochs,
                    batch_size=batch_size,
                    validation_data=(X_val_scaled, y_val_fold),
                    callbacks=[early_stopping],
                    verbose=0  # Silent training
                )
                
                # Evaluate on validation fold
                val_pred = model.predict(X_val_scaled, verbose=0).flatten()
                fold_mae = mean_absolute_error(y_val_fold, val_pred)
                cv_scores.append(fold_mae)
                
                # Clean up to prevent memory issues
                del model
                tf.keras.backend.clear_session()
            
            # Return average CV score (what Optuna minimizes)
            avg_mae = np.mean(cv_scores)
            return avg_mae
        
        # Run Optuna optimization
        study = optuna.create_study(
            direction='minimize',
            study_name='mlp_hr_prediction',
            sampler=optuna.samplers.TPESampler(seed=42)  # Bayesian optimization
        )
        
        print("🚀 Starting hyperparameter search...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best results
        best_params = study.best_params
        best_mae = study.best_value
        
        print(f"\n🏆 BEST HYPERPARAMETERS FOUND:")
        print(f"   📉 Best MAE: {best_mae:.3f} bpm")
        print(f"   🏗️  Architecture: {[best_params.get(f'layer_{i}_size', 0) for i in range(best_params['n_layers'])]}")
        print(f"   📚 Learning rate: {best_params['learning_rate']:.2e}")
        print(f"   📦 Batch size: {best_params['batch_size']}")
        print(f"   🛡️  Dropout: {best_params['dropout']:.3f}")
        print(f"   ⚖️  L2 regularization: {best_params['l2_reg']:.2e}")
        print(f"   ⏱️  Max epochs: {best_params['max_epochs']}")
        print(f"   ⏳ Patience: {best_params['patience']}")
        
        # Update model config with best parameters
        self.config.update(best_params)
        
        # Store hidden layers as list format
        hidden_layers = [best_params.get(f'layer_{i}_size', 0) 
                        for i in range(best_params['n_layers']) 
                        if best_params.get(f'layer_{i}_size', 0) > 0]
        self.config['hidden_layers'] = hidden_layers
        
        if best_mae < 1.17:
            print(f"🎉 SUCCESS: Beat Random Forest target! ({best_mae:.3f} < 1.17 bpm)")
        else:
            print(f"📊 Progress: Current best vs RF: {best_mae:.3f} vs 1.17 bpm")
        
        return best_params, best_mae