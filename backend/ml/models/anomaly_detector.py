# ml/models/anomaly_detector.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection in health data"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class HealthAnomalyDetector:
    """Detect anomalies in health data using multiple methods"""
    
    def __init__(self, method: str = 'autoencoder'):
        """
        Initialize anomaly detector
        
        Args:
            method: 'autoencoder', 'isolation_forest', or 'ensemble'
        """
        self.method = method
        self.models = {}
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_trained = False
        
    def build_models(self, input_dim: int):
        """Build anomaly detection models"""
        
        if self.method == 'autoencoder':
            self.models['autoencoder'] = Autoencoder(input_dim, encoding_dim=32)
        
        elif self.method == 'isolation_forest':
            self.models['isolation_forest'] = IsolationForest(
                n_estimators=200,
                contamination=0.1,
                random_state=42,
                max_samples='auto'
            )
        
        elif self.method == 'ensemble':
            self.models['autoencoder'] = Autoencoder(input_dim, encoding_dim=32)
            self.models['isolation_forest'] = IsolationForest(
                n_estimators=200,
                contamination=0.1,
                random_state=42
            )
    
    def train(self, X_train: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """
        Train anomaly detection models
        
        Args:
            X_train: Normal training data (without anomalies)
            epochs: Number of training epochs for autoencoder
            batch_size: Batch size for training
        """
        
        print(f"Training anomaly detector using {self.method}...")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X_train)
        
        if 'autoencoder' in self.models:
            self._train_autoencoder(X_scaled, epochs, batch_size)
        
        if 'isolation_forest' in self.models:
            self._train_isolation_forest(X_scaled)
        
        # Set anomaly threshold based on reconstruction error
        self._set_threshold(X_scaled)
        
        self.is_trained = True
        print("Anomaly detector trained successfully!")
    
    def _train_autoencoder(self, X: np.ndarray, epochs: int, batch_size: int):
        """Train autoencoder model"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.models['autoencoder'].to(device)
        
        X_tensor = torch.FloatTensor(X).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X), batch_size):
                batch = X_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = epoch_loss / (len(X) / batch_size)
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    def _train_isolation_forest(self, X: np.ndarray):
        """Train Isolation Forest model"""
        
        model = self.models['isolation_forest']
        model.fit(X)
        print("Isolation Forest trained")
    
    def _set_threshold(self, X: np.ndarray):
        """Set anomaly detection threshold"""
        
        if 'autoencoder' in self.models:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = self.models['autoencoder'].to(device)
            model.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                reconstructed = model(X_tensor)
                mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
                reconstruction_errors = mse.cpu().numpy()
            
            # Set threshold at 95th percentile
            self.threshold = np.percentile(reconstruction_errors, 95)
            print(f"Anomaly threshold set to: {self.threshold:.4f}")
    
    def detect_anomalies(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect anomalies in data
        
        Args:
            X: Data to check for anomalies
            
        Returns:
            Dictionary with anomaly scores and binary predictions
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies")
        
        X_scaled = self.scaler.transform(X)
        results = {'anomaly_scores': [], 'is_anomaly': [], 'method_scores': {}}
        
        if 'autoencoder' in self.models:
            ae_scores, ae_anomalies = self._detect_autoencoder(X_scaled)
            results['method_scores']['autoencoder'] = ae_scores
            results['anomaly_scores'].append(ae_scores)
            results['is_anomaly'].append(ae_anomalies)
        
        if 'isolation_forest' in self.models:
            if_scores, if_anomalies = self._detect_isolation_forest(X_scaled)
            results['method_scores']['isolation_forest'] = if_scores
            results['anomaly_scores'].append(if_scores)
            results['is_anomaly'].append(if_anomalies)
        
        # Combine scores if ensemble
        if self.method == 'ensemble':
            # Average anomaly scores
            results['anomaly_scores'] = np.mean(results['anomaly_scores'], axis=0)
            # Flag as anomaly if any method detects it
            results['is_anomaly'] = np.any(results['is_anomaly'], axis=0).astype(int)
        else:
            results['anomaly_scores'] = results['anomaly_scores'][0]
            results['is_anomaly'] = results['is_anomaly'][0]
        
        return results
    
    def _detect_autoencoder(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using autoencoder"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.models['autoencoder'].to(device)
        model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            reconstructed = model(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            reconstruction_errors = mse.cpu().numpy()
        
        # Normalize scores to 0-1
        anomaly_scores = reconstruction_errors / (self.threshold * 2)
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        is_anomaly = (reconstruction_errors > self.threshold).astype(int)
        
        return anomaly_scores, is_anomaly
    
    def _detect_isolation_forest(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using Isolation Forest"""
        
        model = self.models['isolation_forest']
        
        # Predict (-1 for anomaly, 1 for normal)
        predictions = model.predict(X)
        is_anomaly = (predictions == -1).astype(int)
        
        # Get anomaly scores (lower is more anomalous)
        anomaly_scores = -model.score_samples(X)
        # Normalize to 0-1
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        
        return anomaly_scores, is_anomaly
    
    def explain_anomaly(self, X: np.ndarray, feature_names: List[str], top_n: int = 5) -> List[Dict]:
        """
        Explain which features contribute most to anomaly detection
        
        Args:
            X: Anomalous data points
            feature_names: Names of features
            top_n: Number of top contributing features to return
            
        Returns:
            List of dictionaries with anomaly explanations
        """
        
        if not self.is_trained or 'autoencoder' not in self.models:
            return []
        
        X_scaled = self.scaler.transform(X)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.models['autoencoder'].to(device)
        model.eval()
        
        explanations = []
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            reconstructed = model(X_tensor)
            
            # Calculate per-feature reconstruction errors
            feature_errors = ((X_tensor - reconstructed) ** 2).cpu().numpy()
            
            for i in range(len(X)):
                # Get top contributing features
                top_indices = np.argsort(feature_errors[i])[-top_n:][::-1]
                
                explanation = {
                    'sample_index': i,
                    'total_error': feature_errors[i].sum(),
                    'top_features': [
                        {
                            'feature': feature_names[idx],
                            'error': float(feature_errors[i][idx]),
                            'original_value': float(X_scaled[i][idx]),
                            'reconstructed_value': float(reconstructed[i][idx].cpu().numpy())
                        }
                        for idx in top_indices
                    ]
                }
                explanations.append(explanation)
        
        return explanations
    
    def get_anomaly_severity(self, anomaly_score: float) -> str:
        """Classify anomaly severity"""
        
        if anomaly_score < 0.3:
            return 'normal'
        elif anomaly_score < 0.6:
            return 'mild'
        elif anomaly_score < 0.8:
            return 'moderate'
        else:
            return 'severe'
    
    def save_model(self, save_path: str = "ml/trained_models/anomaly_detector.pkl"):
        """Save trained model"""
        
        model_data = {
            'method': self.method,
            'threshold': self.threshold,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        # Save sklearn models
        for name, model in self.models.items():
            if name == 'isolation_forest':
                model_data[name] = model
        
        joblib.dump(model_data, save_path)
        
        # Save PyTorch models separately
        if 'autoencoder' in self.models:
            ae_path = save_path.replace('.pkl', '_autoencoder.pth')
            torch.save(self.models['autoencoder'].state_dict(), ae_path)
        
        print(f"Anomaly detector saved to {save_path}")
    
    def load_model(self, load_path: str = "ml/trained_models/anomaly_detector.pkl"):
        """Load trained model"""
        
        model_data = joblib.load(load_path)
        
        self.method = model_data['method']
        self.threshold = model_data['threshold']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        # Load sklearn models
        if 'isolation_forest' in model_data:
            self.models['isolation_forest'] = model_data['isolation_forest']
        
        # Load PyTorch models
        if 'autoencoder' in self.method or self.method == 'ensemble':
            # Would need input_dim to rebuild
            pass
        
        print(f"Anomaly detector loaded from {load_path}")

# Example usage
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from ml.data.synthetic_generator import save_synthetic_data
    from ml.data.feature_engineering import create_ml_features
    
    # Generate data
    print("Generating synthetic data...")
    datasets = save_synthetic_data()
    
    # Create features
    print("Creating features...")
    ml_features = create_ml_features(datasets)
    
    X = ml_features['X_train']
    feature_names = ml_features['feature_names']
    
    # Train anomaly detector
    print("\nTraining anomaly detector...")
    detector = HealthAnomalyDetector(method='ensemble')
    detector.build_models(X.shape[1])
    detector.train(X, epochs=50)
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    results = detector.detect_anomalies(X[:100])
    
    print(f"\nFound {results['is_anomaly'].sum()} anomalies out of 100 samples")
    print(f"Average anomaly score: {results['anomaly_scores'].mean():.4f}")
    
    # Explain anomalies
    anomaly_indices = np.where(results['is_anomaly'] == 1)[0]
    if len(anomaly_indices) > 0:
        print(f"\nExplaining first anomaly (index {anomaly_indices[0]})...")
        explanations = detector.explain_anomaly(X[anomaly_indices[:1]], feature_names)
        
        for exp in explanations:
            print(f"\nTotal reconstruction error: {exp['total_error']:.4f}")
            print("Top contributing features:")
            for feat in exp['top_features']:
                print(f"  {feat['feature']}: error={feat['error']:.4f}")
    
    # Save model
    detector.save_model()