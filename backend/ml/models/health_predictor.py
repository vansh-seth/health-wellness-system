# ml/models/health_predictor.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
import joblib
from pathlib import Path

class HealthPredictor:
    """Main health prediction model combining multiple ML approaches"""
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize health predictor
        
        Args:
            model_type: 'ensemble', 'deep_learning', or 'traditional'
        """
        self.model_type = model_type
        self.models = {}
        self.is_trained = False
        
    def build_models(self, input_dim: int):
        """Build prediction models for different health metrics"""
        
        if self.model_type == 'ensemble':
            # Ensemble models for different predictions
            self.models = {
                'sleep_quality': GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42
                ),
                'mood_score': GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42
                ),
                'stress_level': GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42
                ),
                'exercise_binary': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42
                )
            }
        
        elif self.model_type == 'deep_learning':
            # Neural network models
            self.models = {
                'sleep_quality': self._build_neural_net(input_dim, 1),
                'mood_score': self._build_neural_net(input_dim, 1),
                'stress_level': self._build_neural_net(input_dim, 1),
                'exercise_binary': self._build_neural_net(input_dim, 1, binary=True)
            }
    
    def _build_neural_net(self, input_dim: int, output_dim: int, binary: bool = False) -> nn.Module:
        """Build a neural network for prediction"""
        
        class HealthNet(nn.Module):
            def __init__(self, input_dim, output_dim, binary=False):
                super(HealthNet, self).__init__()
                self.binary = binary
                
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.2),
                    
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(64, output_dim)
                )
            
            def forward(self, x):
                output = self.network(x)
                if self.binary:
                    output = torch.sigmoid(output)
                return output
        
        return HealthNet(input_dim, output_dim, binary)
    
    def train(self, X_train: np.ndarray, y_train: Dict[str, np.ndarray], 
              X_val: Optional[np.ndarray] = None, y_val: Optional[Dict[str, np.ndarray]] = None):
        """
        Train all prediction models
        
        Args:
            X_train: Training features
            y_train: Dictionary of training targets
            X_val: Validation features
            y_val: Dictionary of validation targets
        """
        
        print("Training health prediction models...")
        
        for target_name, model in self.models.items():
            print(f"\nTraining {target_name} predictor...")
            
            if target_name not in y_train:
                print(f"Warning: No training data for {target_name}")
                continue
            
            if self.model_type == 'ensemble' or self.model_type == 'traditional':
                # Train sklearn models
                model.fit(X_train, y_train[target_name])
                
                # Evaluate
                train_score = model.score(X_train, y_train[target_name])
                print(f"Training score: {train_score:.4f}")
                
                if X_val is not None and y_val is not None:
                    val_score = model.score(X_val, y_val[target_name])
                    print(f"Validation score: {val_score:.4f}")
            
            elif self.model_type == 'deep_learning':
                # Train neural network
                self._train_neural_net(
                    model, X_train, y_train[target_name],
                    X_val, y_val[target_name] if y_val else None
                )
        
        self.is_trained = True
        print("\nAll models trained successfully!")
    
    def _train_neural_net(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                         epochs: int = 100, batch_size: int = 32):
        """Train a neural network model"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss() if not model.binary else nn.BCELoss()
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            if epoch % 10 == 0 and X_val is not None:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    print(f"Epoch {epoch}: Val Loss = {val_loss.item():.4f}")
                scheduler.step(val_loss)
                model.train()
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions for all health metrics
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of predictions for each health metric
        """
        
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        predictions = {}
        
        for target_name, model in self.models.items():
            if self.model_type == 'ensemble' or self.model_type == 'traditional':
                # Sklearn prediction
                if target_name == 'exercise_binary':
                    predictions[target_name] = model.predict_proba(X)[:, 1]
                else:
                    predictions[target_name] = model.predict(X)
            
            elif self.model_type == 'deep_learning':
                # Neural network prediction
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(device)
                    outputs = model(X_tensor)
                    predictions[target_name] = outputs.cpu().numpy().flatten()
        
        return predictions
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Get feature importance for tree-based models"""
        
        if self.model_type != 'ensemble':
            return {}
        
        importance_dict = {}
        
        for target_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_dict[target_name] = {
                    feature_names[i]: float(importances[i])
                    for i in range(len(feature_names))
                }
                # Sort by importance
                importance_dict[target_name] = dict(
                    sorted(importance_dict[target_name].items(), 
                          key=lambda x: x[1], reverse=True)
                )
        
        return importance_dict
    
    def save_models(self, save_dir: str = "ml/trained_models"):
        """Save trained models to disk"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for target_name, model in self.models.items():
            model_path = save_path / f"{target_name}_predictor.pkl"
            
            if self.model_type in ['ensemble', 'traditional']:
                joblib.dump(model, model_path)
            elif self.model_type == 'deep_learning':
                torch.save(model.state_dict(), model_path.with_suffix('.pth'))
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'targets': list(self.models.keys())
        }
        joblib.dump(metadata, save_path / 'predictor_metadata.pkl')
        
        print(f"Models saved to {save_path}")
    
    def load_models(self, load_dir: str = "ml/trained_models"):
        """Load trained models from disk"""
        
        load_path = Path(load_dir)
        
        # Load metadata
        metadata = joblib.load(load_path / 'predictor_metadata.pkl')
        self.model_type = metadata['model_type']
        self.is_trained = metadata['is_trained']
        
        # Load models
        for target_name in metadata['targets']:
            model_path = load_path / f"{target_name}_predictor.pkl"
            
            if self.model_type in ['ensemble', 'traditional']:
                self.models[target_name] = joblib.load(model_path)
            elif self.model_type == 'deep_learning':
                # Would need to rebuild architecture first
                pass
        
        print(f"Models loaded from {load_path}")

# Example usage
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from ml.data.synthetic_generator import save_synthetic_data
    from ml.data.feature_engineering import create_ml_features
    from sklearn.model_selection import train_test_split
    
    # Generate data
    print("Generating synthetic data...")
    datasets = save_synthetic_data()
    
    # Create features
    print("Creating features...")
    ml_features = create_ml_features(datasets)
    
    X = ml_features['X_train']
    y_dict = {
        'sleep_quality': ml_features['y_train'][:, 0],
        'mood_score': ml_features['y_train'][:, 1],
        'stress_level': ml_features['y_train'][:, 2],
        'exercise_binary': ml_features['y_train'][:, 3]
    }
    
    # Split data
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    y_train = {k: v[:len(X_train)] for k, v in y_dict.items()}
    y_val = {k: v[len(X_train):] for k, v in y_dict.items()}
    
    # Train predictor
    print("\nTraining health predictor...")
    predictor = HealthPredictor(model_type='ensemble')
    predictor.build_models(X.shape[1])
    predictor.train(X_train, y_train, X_val, y_val)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict(X_val[:10])
    
    print("\nSample predictions:")
    for target, preds in predictions.items():
        print(f"{target}: {preds[:5]}")
    
    # Feature importance
    print("\nTop 10 important features:")
    importance = predictor.get_feature_importance(ml_features['feature_names'])
    for target, features in importance.items():
        print(f"\n{target}:")
        for feat, imp in list(features.items())[:10]:
            print(f"  {feat}: {imp:.4f}")
    
    # Save models
    predictor.save_models()