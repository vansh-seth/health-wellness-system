# ml/models/anomaly_detector_v2.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import joblib

class ImprovedAutoencoder(nn.Module):
    """Enhanced Autoencoder with better regularization for anomaly detection"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super(ImprovedAutoencoder, self).__init__()
        
        # Encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
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
    """Advanced anomaly detection for health data"""
    
    def __init__(self, method: str = 'ensemble'):
        """
        Initialize anomaly detector
        
        Args:
            method: 'autoencoder', 'isolation_forest', or 'ensemble'
        """
        self.method = method
        self.models = {}
        self.scaler = RobustScaler()  # More robust to outliers
        self.threshold = None
        self.is_trained = False
        self.feature_importance = None
        
    def build_models(self, input_dim: int):
        """Build anomaly detection models"""
        
        if self.method in ['autoencoder', 'ensemble']:
            self.models['autoencoder'] = ImprovedAutoencoder(input_dim, encoding_dim=min(32, input_dim // 3))
        
        if self.method in ['isolation_forest', 'ensemble']:
            self.models['isolation_forest'] = IsolationForest(
                n_estimators=200,
                contamination=0.05,  # Assume 5% anomalies
                max_samples=256,
                random_state=42,
                n_jobs=-1
            )
    
    def train(self, X_train: np.ndarray, epochs: int = 100, batch_size: int = 32, 
             validation_split: float = 0.2):
        """
        Train anomaly detection models with validation
        
        Args:
            X_train: Normal training data (without anomalies)
            epochs: Number of training epochs for autoencoder
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
        """
        
        print(f"\nTraining anomaly detector using {self.method}...")
        print(f"Training samples: {len(X_train):,}")
        
        # Clean data
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Split for validation
        if validation_split > 0:
            split_idx = int(len(X_scaled) * (1 - validation_split))
            X_train_split = X_scaled[:split_idx]
            X_val_split = X_scaled[split_idx:]
        else:
            X_train_split = X_scaled
            X_val_split = None
        
        if 'autoencoder' in self.models:
            self._train_autoencoder(X_train_split, X_val_split, epochs, batch_size)
        
        if 'isolation_forest' in self.models:
            self._train_isolation_forest(X_scaled)
        
        # Set anomaly threshold
        self._set_threshold(X_scaled)
        
        self.is_trained = True
        print("\n✓ Anomaly detector trained successfully!")
    
    def _train_autoencoder(self, X_train: np.ndarray, X_val: Optional[np.ndarray], 
                          epochs: int, batch_size: int):
        """Train autoencoder with early stopping"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.models['autoencoder'].to(device)
        
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device) if X_val is not None else None
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        print("\nTraining Autoencoder:")
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Mini-batch training
            indices = torch.randperm(len(X_train_tensor))
            
            for i in range(0, len(X_train_tensor), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch = X_train_tensor[batch_indices]
                
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            
            # Validation
            if X_val_tensor is not None:
                model.eval()
                with torch.no_grad():
                    val_reconstructed = model(X_val_tensor)
                    val_loss = criterion(val_reconstructed, X_val_tensor).item()
                model.train()
                
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch:3d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch}")
                        break
            else:
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch:3d}: Train Loss = {avg_train_loss:.4f}")
    
    def _train_isolation_forest(self, X: np.ndarray):
        """Train Isolation Forest model"""
        print("\nTraining Isolation Forest...")
        model = self.models['isolation_forest']
        model.fit(X)
        print("  ✓ Isolation Forest trained")
    
    def _set_threshold(self, X: np.ndarray):
        """Set anomaly detection threshold based on training data"""
        
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
            mean_error = np.mean(reconstruction_errors)
            std_error = np.std(reconstruction_errors)
            
            print(f"\nAnomaly Detection Thresholds:")
            print(f"  Mean reconstruction error: {mean_error:.4f}")
            print(f"  Std reconstruction error: {std_error:.4f}")
            print(f"  95th percentile threshold: {self.threshold:.4f}")
    
    def detect_anomalies(self, X: np.ndarray, return_details: bool = False) -> Dict[str, np.ndarray]:
        """
        Detect anomalies in data
        
        Args:
            X: Data to check for anomalies
            return_details: Whether to return detailed scores per method
            
        Returns:
            Dictionary with anomaly scores, predictions, and severity
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies")
        
        # Clean and scale data
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        results = {
            'anomaly_scores': [],
            'is_anomaly': [],
            'severity': [],
            'method_scores': {}
        }
        
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
        if self.method == 'ensemble' and len(results['anomaly_scores']) > 1:
            # Weighted average (autoencoder gets more weight)
            results['anomaly_scores'] = (
                results['anomaly_scores'][0] * 0.6 + 
                results['anomaly_scores'][1] * 0.4
            )
            # Flag as anomaly if either method detects it
            results['is_anomaly'] = np.any(results['is_anomaly'], axis=0).astype(int)
        else:
            results['anomaly_scores'] = results['anomaly_scores'][0]
            results['is_anomaly'] = results['is_anomaly'][0]
        
        # Calculate severity
        results['severity'] = np.array([
            self.get_anomaly_severity(score) 
            for score in results['anomaly_scores']
        ])
        
        if not return_details:
            # Remove detailed method scores to save memory
            del results['method_scores']
        
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
        max_score = self.threshold * 3  # 3x threshold is definitely anomalous
        anomaly_scores = np.clip(reconstruction_errors / max_score, 0, 1)
        
        is_anomaly = (reconstruction_errors > self.threshold).astype(int)
        
        return anomaly_scores, is_anomaly
    
    def _detect_isolation_forest(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using Isolation Forest"""
        
        model = self.models['isolation_forest']
        
        # Predict (-1 for anomaly, 1 for normal)
        predictions = model.predict(X)
        is_anomaly = (predictions == -1).astype(int)
        
        # Get anomaly scores (lower is more anomalous)
        raw_scores = model.score_samples(X)
        # Invert and normalize to 0-1 (higher = more anomalous)
        anomaly_scores = 1 / (1 + np.exp(raw_scores))  # Sigmoid transformation
        
        return anomaly_scores, is_anomaly
    
    def explain_anomaly(self, X: np.ndarray, feature_names: List[str], 
                       top_n: int = 5) -> List[Dict]:
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
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
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
                
                total_error = feature_errors[i].sum()
                
                explanation = {
                    'sample_index': i,
                    'total_error': float(total_error),
                    'anomaly_score': float(total_error / (self.threshold * len(feature_names))),
                    'severity': self.get_anomaly_severity(total_error / len(feature_names)),
                    'top_features': []
                }
                
                for idx in top_indices:
                    if idx < len(feature_names):
                        # Calculate contribution percentage
                        contribution = (feature_errors[i][idx] / total_error) * 100
                        
                        explanation['top_features'].append({
                            'feature': feature_names[idx],
                            'error': float(feature_errors[i][idx]),
                            'contribution_pct': float(contribution),
                            'original_value': float(X_scaled[i][idx]),
                            'reconstructed_value': float(reconstructed[i][idx].cpu().numpy()),
                            'deviation': abs(float(X_scaled[i][idx] - reconstructed[i][idx].cpu().numpy()))
                        })
                
                explanations.append(explanation)
        
        return explanations
    
    def get_anomaly_severity(self, anomaly_score: float) -> str:
        """Classify anomaly severity"""
        
        if anomaly_score < 0.3:
            return 'normal'
        elif anomaly_score < 0.5:
            return 'mild'
        elif anomaly_score < 0.7:
            return 'moderate'
        elif anomaly_score < 0.85:
            return 'high'
        else:
            return 'critical'
    
    def get_health_insights(self, anomaly_explanation: Dict) -> List[str]:
        """Generate human-readable health insights from anomaly explanation"""
        
        insights = []
        severity = anomaly_explanation['severity']
        
        if severity == 'normal':
            insights.append("✓ All health metrics are within normal ranges")
            return insights
        
        insights.append(f"⚠️ {severity.upper()} anomaly detected")
        
        # Analyze top features
        for feat_info in anomaly_explanation['top_features'][:3]:
            feature = feat_info['feature']
            contrib = feat_info['contribution_pct']
            
            if contrib > 30:
                if 'sleep' in feature.lower():
                    insights.append(f"• Sleep pattern significantly different from normal ({contrib:.0f}% contribution)")
                elif 'stress' in feature.lower():
                    insights.append(f"• Stress levels unusually high/low ({contrib:.0f}% contribution)")
                elif 'mood' in feature.lower():
                    insights.append(f"• Mood pattern deviates from baseline ({contrib:.0f}% contribution)")
                elif 'exercise' in feature.lower():
                    insights.append(f"• Exercise behavior changed significantly ({contrib:.0f}% contribution)")
                elif 'calories' in feature.lower():
                    insights.append(f"• Calorie intake unusual compared to normal ({contrib:.0f}% contribution)")
                else:
                    insights.append(f"• {feature} shows abnormal pattern ({contrib:.0f}% contribution)")
        
        if severity in ['high', 'critical']:
            insights.append("⚕️ Consider consulting with a healthcare provider")
        
        return insights
    
    def save_model(self, save_path: str = "ml/trained_models/anomaly_detector_v2.pkl"):
        """Save trained model"""
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'method': self.method,
            'threshold': self.threshold,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'version': 'v2'
        }
        
        # Save sklearn models
        if 'isolation_forest' in self.models:
            model_data['isolation_forest'] = self.models['isolation_forest']
        
        joblib.dump(model_data, save_path)
        
        # Save PyTorch models separately
        if 'autoencoder' in self.models:
            ae_path = save_path.replace('.pkl', '_autoencoder.pth')
            torch.save(self.models['autoencoder'].state_dict(), ae_path)
            print(f"  Autoencoder saved to {ae_path}")
        
        print(f"  Anomaly detector saved to {save_path}")
    
    def load_model(self, load_path: str = "ml/trained_models/anomaly_detector_v2.pkl", 
                  input_dim: Optional[int] = None):
        """Load trained model"""
        
        model_data = joblib.load(load_path)
        
        self.method = model_data['method']
        self.threshold = model_data['threshold']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        # Load sklearn models
        if 'isolation_forest' in model_data:
            self.models['isolation_forest'] = model_data['isolation_forest']
        
        # Load PyTorch models (need input_dim to rebuild)
        if input_dim is not None and self.method in ['autoencoder', 'ensemble']:
            self.build_models(input_dim)
            ae_path = load_path.replace('.pkl', '_autoencoder.pth')
            if Path(ae_path).exists():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.models['autoencoder'].load_state_dict(
                    torch.load(ae_path, map_location=device)
                )
                print(f"  Autoencoder loaded from {ae_path}")
        
        print(f"  Anomaly detector loaded from {load_path}")


# Example usage and training
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from ml.data.synthetic_generator import save_synthetic_data
    from ml.data.feature_engineering_v2 import create_ml_features
    
    print("="*80)
    print("HEALTH ANOMALY DETECTOR - TRAINING & DEMO")
    print("="*80)
    
    # Generate data
    print("\nStep 1: Generating synthetic data...")
    datasets = save_synthetic_data()
    
    # Create features
    print("\nStep 2: Creating features...")
    ml_features = create_ml_features(datasets, sample_rate=0.5, lookback_days=28)
    
    X = ml_features['X_train']
    feature_names = ml_features['feature_names']
    
    print(f"\nDataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    # Train anomaly detector
    print("\n" + "="*80)
    print("Step 3: Training Anomaly Detector")
    print("="*80)
    
    detector = HealthAnomalyDetector(method='ensemble')
    detector.build_models(X.shape[1])
    detector.train(X, epochs=50, batch_size=64, validation_split=0.2)
    
    # Detect anomalies
    print("\n" + "="*80)
    print("Step 4: Detecting Anomalies")
    print("="*80)
    
    test_samples = X[:200]
    results = detector.detect_anomalies(test_samples, return_details=True)
    
    num_anomalies = results['is_anomaly'].sum()
    print(f"\nFound {num_anomalies} anomalies out of {len(test_samples)} samples ({num_anomalies/len(test_samples)*100:.1f}%)")
    print(f"Average anomaly score: {results['anomaly_scores'].mean():.4f}")
    
    # Show severity distribution
    severity_counts = pd.Series(results['severity']).value_counts()
    print("\nSeverity Distribution:")
    for severity, count in severity_counts.items():
        print(f"  {severity:10s}: {count:3d} ({count/len(test_samples)*100:5.1f}%)")
    
    # Explain top anomalies
    anomaly_indices = np.where(results['is_anomaly'] == 1)[0]
    if len(anomaly_indices) > 0:
        print("\n" + "="*80)
        print("Step 5: Explaining Top Anomalies")
        print("="*80)
        
        # Get top 3 anomalies by score
        top_anomaly_indices = anomaly_indices[
            np.argsort(results['anomaly_scores'][anomaly_indices])[-3:][::-1]
        ]
        
        explanations = detector.explain_anomaly(
            test_samples[top_anomaly_indices], 
            feature_names, 
            top_n=5
        )
        
        for i, exp in enumerate(explanations, 1):
            print(f"\n{'─'*80}")
            print(f"ANOMALY #{i} (Sample Index: {exp['sample_index']})")
            print(f"{'─'*80}")
            print(f"Severity: {exp['severity'].upper()}")
            print(f"Anomaly Score: {exp['anomaly_score']:.4f}")
            print(f"Total Reconstruction Error: {exp['total_error']:.4f}")
            
            # Get health insights
            insights = detector.get_health_insights(exp)
            print("\nHealth Insights:")
            for insight in insights:
                print(f"  {insight}")
            
            print("\nTop Contributing Features:")
            for j, feat in enumerate(exp['top_features'], 1):
                print(f"\n  {j}. {feat['feature']}")
                print(f"     Contribution: {feat['contribution_pct']:.1f}%")
                print(f"     Error: {feat['error']:.4f}")
                print(f"     Original: {feat['original_value']:.3f}")
                print(f"     Reconstructed: {feat['reconstructed_value']:.3f}")
                print(f"     Deviation: {feat['deviation']:.3f}")
    
    # Save model
    print("\n" + "="*80)
    print("Step 6: Saving Model")
    print("="*80)
    detector.save_model()
    
    print("\n✓ Anomaly detector ready for use!")
    print("\nTo use in your application:")
    print("  from ml.models.anomaly_detector_v2 import HealthAnomalyDetector")
    print("  detector = HealthAnomalyDetector()")
    print("  detector.load_model(input_dim=X.shape[1])")
    print("  results = detector.detect_anomalies(X_new)")