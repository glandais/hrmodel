import pandas as pd
import numpy as np
import warnings
from .base_model import BaseModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")
    # Create dummy classes when torch is not available
    class nn:
        class Module:
            pass
        class LSTM:
            pass
        class Linear:
            pass
        class Softplus:
            pass
    torch = None


class DeepARNetwork(nn.Module if TORCH_AVAILABLE else object):
    """DeepAR neural network architecture."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super(DeepARNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers for distribution parameters
        # We predict mean and standard deviation for Gaussian distribution
        self.fc_mu = nn.Linear(hidden_size, 1)  # Mean
        self.fc_sigma = nn.Linear(hidden_size, 1)  # Standard deviation
        
        # Activation for sigma (must be positive)
        self.softplus = nn.Softplus()
        
    def forward(self, x, hidden=None):
        """Forward pass through the network."""
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Get the last output for prediction
        if len(lstm_out.shape) == 3:
            # Batch mode
            last_output = lstm_out[:, -1, :]
        else:
            # Single sample
            last_output = lstm_out[-1, :]
        
        # Predict distribution parameters
        mu = self.fc_mu(last_output)
        sigma = self.softplus(self.fc_sigma(last_output)) + 1e-6  # Ensure positive
        
        return mu, sigma, hidden
    
    def sample(self, mu, sigma):
        """Sample from the predicted distribution."""
        # Sample from Gaussian distribution
        eps = torch.randn_like(mu)
        return mu + sigma * eps


class DeepARModel(BaseModel):
    """DeepAR model for probabilistic time series forecasting.
    
    DeepAR is Amazon's deep learning approach for time series that:
    - Uses RNNs (LSTM/GRU) for temporal dependencies
    - Produces probabilistic forecasts (not just point estimates)
    - Handles multiple related time series
    - Learns from many similar time series jointly
    """
    
    def __init__(self, feature_columns=None,
                 hidden_size=64, num_layers=2, dropout=0.1,
                 learning_rate=0.001, n_epochs=50, batch_size=32,
                 sequence_length=30, normalize=True):
        """Initialize DeepAR model.
        
        Args:
            feature_columns: List of feature column names
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimization
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            sequence_length: Length of input sequences
            normalize: Whether to normalize features
        """
        super().__init__("DeepAR", feature_columns)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Install with: pip install torch")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        self.model = None
        self.scaler = StandardScaler() if normalize else None
        self.target_scaler = StandardScaler()  # Always scale target
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training history
        self.train_losses = []
    
    def _create_sequences(self, X, y):
        """Create sequences for time series training."""
        sequences_X = []
        sequences_y = []
        
        for i in range(len(X) - self.sequence_length):
            seq_X = X[i:i + self.sequence_length]
            seq_y = y[i + self.sequence_length]
            sequences_X.append(seq_X)
            sequences_y.append(seq_y)
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the DeepAR model."""
        # Prepare features
        X_prepared = self.prepare_features(X)
        
        # Remove NaN values
        mask = ~(y.isna() | X_prepared.isna().any(axis=1))
        X_clean = X_prepared[mask].values
        y_clean = y[mask].values.reshape(-1, 1)
        
        # Scale features and target
        if self.normalize and self.scaler:
            X_clean = self.scaler.fit_transform(X_clean)
        y_clean = self.target_scaler.fit_transform(y_clean).flatten()
        
        # Create sequences
        X_sequences, y_sequences = self._create_sequences(X_clean, y_clean)
        
        if len(X_sequences) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        y_tensor = torch.FloatTensor(y_sequences).to(self.device)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_size = X_clean.shape[1]
        self.model = DeepARNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss function (negative log-likelihood for Gaussian)
        def gaussian_nll_loss(mu, sigma, target):
            """Negative log-likelihood for Gaussian distribution."""
            distribution = torch.distributions.Normal(mu.squeeze(), sigma.squeeze())
            return -distribution.log_prob(target).mean()
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_X, batch_y in dataloader:
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                mu, sigma, _ = self.model(batch_X)
                
                # Calculate loss
                loss = gaussian_nll_loss(mu, sigma, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
            self.train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.n_epochs}], Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        # Evaluate on training data
        self.model.eval()
        with torch.no_grad():
            mu_train, _, _ = self.model(X_tensor)
            train_pred = mu_train.cpu().numpy().flatten()
            
            # Inverse transform predictions
            train_pred = self.target_scaler.inverse_transform(
                train_pred.reshape(-1, 1)
            ).flatten()
            y_original = self.target_scaler.inverse_transform(
                y_sequences.reshape(-1, 1)
            ).flatten()
            
            # Calculate metrics
            self.metrics['training_samples'] = len(X_sequences)
            self.metrics['n_features'] = input_size
            self.metrics['train_mae'] = mean_absolute_error(y_original, train_pred)
            self.metrics['train_rmse'] = np.sqrt(mean_squared_error(y_original, train_pred))
            self.metrics['train_r2'] = r2_score(y_original, train_pred)
        
        print(f"DeepAR model trained on {len(X_sequences)} sequences")
        print(f"Hidden size: {self.hidden_size}, Layers: {self.num_layers}")
        print(f"Training R²: {self.metrics['train_r2']:.4f}, MAE: {self.metrics['train_mae']:.2f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the DeepAR model."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X_prepared = self.prepare_features(X).values
        
        # Scale features
        if self.normalize and self.scaler:
            X_prepared = self.scaler.transform(X_prepared)
        
        predictions = []
        
        # We need at least sequence_length points to make a prediction
        if len(X_prepared) < self.sequence_length:
            # Pad with zeros if not enough data
            padding = np.zeros((self.sequence_length - len(X_prepared), X_prepared.shape[1]))
            X_prepared = np.vstack([padding, X_prepared])
        
        # Make predictions using sliding window
        self.model.eval()
        with torch.no_grad():
            for i in range(len(X_prepared) - self.sequence_length + 1):
                # Get sequence
                seq = X_prepared[i:i + self.sequence_length]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                
                # Predict
                mu, sigma, _ = self.model(seq_tensor)
                
                # Use mean as point estimate
                pred = mu.cpu().numpy().flatten()[0]
                predictions.append(pred)
        
        # Handle case where we need more predictions
        while len(predictions) < len(X):
            # Use last prediction for remaining points
            predictions.append(predictions[-1] if predictions else 0)
        
        # Trim to match input length
        predictions = np.array(predictions[:len(X)])
        
        # Inverse transform predictions
        predictions = self.target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
        
        # Clip to reasonable HR range
        predictions = np.clip(predictions, 40, 220)
        
        return predictions
    
    def predict_probabilistic(self, X: pd.DataFrame, n_samples=100):
        """Make probabilistic predictions (with uncertainty)."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Similar to predict, but return multiple samples
        X_prepared = self.prepare_features(X).values
        
        if self.normalize and self.scaler:
            X_prepared = self.scaler.transform(X_prepared)
        
        all_predictions = []
        
        for _ in range(n_samples):
            predictions = []
            
            self.model.eval()
            with torch.no_grad():
                for i in range(max(1, len(X_prepared) - self.sequence_length + 1)):
                    if i + self.sequence_length <= len(X_prepared):
                        seq = X_prepared[i:i + self.sequence_length]
                    else:
                        # Pad if needed
                        seq = X_prepared[-self.sequence_length:]
                    
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                    
                    # Predict distribution parameters
                    mu, sigma, _ = self.model(seq_tensor)
                    
                    # Sample from distribution
                    sample = self.model.sample(mu, sigma)
                    pred = sample.cpu().numpy().flatten()[0]
                    predictions.append(pred)
            
            # Pad or trim
            while len(predictions) < len(X):
                predictions.append(predictions[-1] if predictions else 0)
            predictions = predictions[:len(X)]
            
            # Inverse transform
            predictions = self.target_scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            ).flatten()
            
            all_predictions.append(predictions)
        
        all_predictions = np.array(all_predictions)
        
        # Return mean and confidence intervals
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        lower_bound = np.percentile(all_predictions, 2.5, axis=0)
        upper_bound = np.percentile(all_predictions, 97.5, axis=0)
        
        return {
            'mean': np.clip(mean_pred, 40, 220),
            'std': std_pred,
            'lower_95': np.clip(lower_bound, 40, 220),
            'upper_95': np.clip(upper_bound, 40, 220)
        }
    
    def get_model_info(self) -> dict:
        """Get detailed model information."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "model_type": "DeepAR",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "sequence_length": self.sequence_length,
            "n_epochs": self.n_epochs,
            "final_loss": self.train_losses[-1] if self.train_losses else None,
            "device": str(self.device),
            "training_samples": self.metrics.get('training_samples', 0)
        }
        
        if self.model is not None:
            info['n_parameters'] = sum(p.numel() for p in self.model.parameters())
        
        return info
    
    def save(self, filepath: str) -> None:
        """Save the model to a file."""
        model_data = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'model_config': {
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'input_size': self.model.lstm.input_size if self.model else None
            },
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'train_losses': self.train_losses,
            'sequence_length': self.sequence_length,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"DeepAR model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the model from a file."""
        model_data = joblib.load(filepath)
        
        # Restore configuration
        config = model_data['model_config']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # Recreate model architecture
        if config['input_size']:
            self.model = DeepARNetwork(
                input_size=config['input_size'],
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(model_data['model_state_dict'])
        
        self.scaler = model_data['scaler']
        self.target_scaler = model_data['target_scaler']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']
        self.train_losses = model_data['train_losses']
        self.sequence_length = model_data['sequence_length']
        self.is_trained = model_data['is_trained']
        
        print(f"DeepAR model loaded from {filepath}")