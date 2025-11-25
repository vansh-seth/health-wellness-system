# ml/models/health_predictor_v3.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import joblib
from pathlib import Path

class ImprovedHealthPredictor:
    """Improved health prediction with better regularization and early stopping"""
    
    def __init__(self, model_type: str = 'stacking'):
        """
        Initialize health predictor
        
        Args:
            model_type: 'stacking', 'xgboost', 'lightgbm'
        """
        self.model_type = model_type
        self.models = {}
        self.is_trained = False
        
    def build_models(self, input_dim: int):
        """Build improved prediction models with stronger regularization"""
        
        if self.model_type == 'stacking':
            # Stacking ensemble with REDUCED COMPLEXITY
            base_regressors = [
                ('xgb', XGBRegressor(
                    n_estimators=150,         # Reduced from 200
                    learning_rate=0.01,       # Slower learning
                    max_depth=4,              # Reduced from 5
                    min_child_weight=6,       # Increased from 5
                    subsample=0.6,            # Reduced from 0.7
                    colsample_bytree=0.6,     # Reduced from 0.7
                    gamma=0.5,                # Regularization
                    reg_alpha=1.5,            # Increased L1
                    reg_lambda=3.0,           # Strong L2
                    random_state=42,
                    n_jobs=-1
                )),
                ('lgbm', LGBMRegressor(
                    n_estimators=150,
                    learning_rate=0.01,
                    max_depth=4,
                    num_leaves=15,            # Reduced from 31
                    subsample=0.6,
                    colsample_bytree=0.6,
                    min_child_samples=20,     # Increased
                    reg_alpha=1.5,
                    reg_lambda=3.0,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )),
                ('rf', RandomForestRegressor(
                    n_estimators=100,         # Reduced from 200
                    max_depth=10,             # Reduced from 15
                    min_samples_split=20,     # Increased from 10
                    min_samples_leaf=8,       # Increased from 4
                    max_features='sqrt',      # Reduced feature usage
                    random_state=42,
                    n_jobs=-1
                ))
            ]
            
            # Create stacking models for each target
            for target in ['sleep_quality', 'mood_score', 'stress_level']:
                self.models[target] = StackingRegressor(
                    estimators=base_regressors,
                    final_estimator=Ridge(alpha=5.0),  # Increased regularization
                    cv=5,
                    n_jobs=-1
                )
            
            # Classification for exercise
            base_classifiers = [
                ('xgb', XGBClassifier(
                    n_estimators=150,
                    learning_rate=0.01,
                    max_depth=4,
                    min_child_weight=6,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    gamma=0.5,
                    reg_alpha=1.5,
                    reg_lambda=3.0,
                    random_state=42,
                    n_jobs=-1
                )),
                ('lgbm', LGBMClassifier(
                    n_estimators=150,
                    learning_rate=0.01,
                    max_depth=4,
                    num_leaves=15,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    min_child_samples=20,
                    reg_alpha=1.5,
                    reg_lambda=3.0,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                ))
            ]
            
            self.models['exercise_binary'] = StackingClassifier(
                estimators=base_classifiers,
                final_estimator=LogisticRegression(C=0.1, max_iter=1000),  # Strong regularization
                cv=5,
                n_jobs=-1
            )
        
        elif self.model_type == 'xgboost':
            # Pure XGBoost with strong regularization
            for target in ['sleep_quality', 'mood_score', 'stress_level']:
                self.models[target] = XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.01,
                    max_depth=4,
                    min_child_weight=6,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    gamma=0.5,
                    reg_alpha=1.5,
                    reg_lambda=3.0,
                    random_state=42,
                    n_jobs=-1
                )
            
            self.models['exercise_binary'] = XGBClassifier(
                n_estimators=200,
                learning_rate=0.01,
                max_depth=4,
                min_child_weight=6,
                subsample=0.6,
                colsample_bytree=0.6,
                gamma=0.5,
                reg_alpha=1.5,
                reg_lambda=3.0,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'lightgbm':
            # LightGBM with strong regularization
            for target in ['sleep_quality', 'mood_score', 'stress_level']:
                self.models[target] = LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.01,
                    max_depth=4,
                    num_leaves=15,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    min_child_samples=20,
                    reg_alpha=1.5,
                    reg_lambda=3.0,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            
            self.models['exercise_binary'] = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.01,
                max_depth=4,
                num_leaves=15,
                subsample=0.6,
                colsample_bytree=0.6,
                min_child_samples=20,
                reg_alpha=1.5,
                reg_lambda=3.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
    
    def train(self, X_train: np.ndarray, y_train: Dict[str, np.ndarray], 
              X_val: Optional[np.ndarray] = None, y_val: Optional[Dict[str, np.ndarray]] = None):
        """
        Train all prediction models with early stopping
        
        Args:
            X_train: Training features
            y_train: Dictionary of training targets
            X_val: Validation features
            y_val: Dictionary of validation targets
        """
        
        print(f"\nTraining health prediction models using {self.model_type}...")
        print(f"Training samples: {X_train.shape[0]:,}")
        print(f"Validation samples: {X_val.shape[0]:,}" if X_val is not None else "No validation set")
        
        # Clean data
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        if X_val is not None:
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        for key in y_train:
            y_train[key] = np.nan_to_num(y_train[key], nan=0.0, posinf=0.0, neginf=0.0)
        
        if y_val is not None:
            for key in y_val:
                y_val[key] = np.nan_to_num(y_val[key], nan=0.0, posinf=0.0, neginf=0.0)
        
        for target_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {target_name} predictor...")
            print(f"{'='*60}")
            
            if target_name not in y_train:
                print(f"Warning: No training data for {target_name}")
                continue
            
            # Train with early stopping for XGBoost/LightGBM
            if isinstance(model, (XGBRegressor, XGBClassifier)) and X_val is not None:
                model.fit(
                    X_train, y_train[target_name],
                    eval_set=[(X_val, y_val[target_name])],
                    early_stopping_rounds=30,
                    verbose=False
                )
                print(f"Best iteration: {model.best_iteration}")
                
            elif isinstance(model, (LGBMRegressor, LGBMClassifier)) and X_val is not None:
                model.fit(
                    X_train, y_train[target_name],
                    eval_set=[(X_val, y_val[target_name])],
                    callbacks=[
                        # Note: early_stopping is deprecated in newer versions
                        # Model will use n_estimators as specified
                    ]
                )
            else:
                # For stacking and other models
                model.fit(X_train, y_train[target_name])
            
            # Evaluate on training set
            train_score = model.score(X_train, y_train[target_name])
            print(f"Training R²/Accuracy: {train_score:.4f}")
            
            # Evaluate on validation set
            if X_val is not None and y_val is not None and target_name in y_val:
                val_score = model.score(X_val, y_val[target_name])
                print(f"Validation R²/Accuracy: {val_score:.4f}")
                
                # Check for overfitting
                overfit_gap = train_score - val_score
                print(f"Overfitting gap: {overfit_gap:.4f}", end="")
                if overfit_gap > 0.15:
                    print(" ⚠️  HIGH - Consider more regularization")
                elif overfit_gap > 0.10:
                    print(" ⚠️  MODERATE")
                else:
                    print(" ✓ GOOD")
                
                # Calculate additional metrics
                if target_name != 'exercise_binary':
                    val_pred = model.predict(X_val)
                    mae = np.mean(np.abs(val_pred - y_val[target_name]))
                    rmse = np.sqrt(np.mean((val_pred - y_val[target_name])**2))
                    print(f"Validation MAE: {mae:.4f}")
                    print(f"Validation RMSE: {rmse:.4f}")
                else:
                    val_pred = model.predict(X_val)
                    accuracy = np.mean(val_pred == y_val[target_name])
                    print(f"Validation Accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        print(f"\n{'='*60}")
        print("All models trained successfully!")
        print(f"{'='*60}")
    
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
        
        # Clean input data
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        predictions = {}
        
        for target_name, model in self.models.items():
            if target_name == 'exercise_binary':
                predictions[target_name] = model.predict_proba(X)[:, 1]
            else:
                predictions[target_name] = model.predict(X)
        
        return predictions
    
    def get_feature_importance(self, feature_names: List[str], top_n: int = 20) -> Dict[str, Dict[str, float]]:
        """Get feature importance for all models safely"""
        
        importance_dict = {}
        
        for target_name, model in self.models.items():
            importances = None

            # Stacking models
            if isinstance(model, (StackingRegressor, StackingClassifier)):
                try:
                    if hasattr(model, 'estimators_') and model.estimators_ is not None:
                        all_importances = []
                        # estimators_ is a list of fitted estimators (not tuples)
                        for estimator in model.estimators_:
                            if hasattr(estimator, 'feature_importances_'):
                                all_importances.append(estimator.feature_importances_)
                        if all_importances:
                            importances = np.mean(all_importances, axis=0)
                except Exception as e:
                    print(f"Warning: Could not get importance from {target_name}: {e}")
            
            # Single models with feature_importances_
            elif hasattr(model, 'feature_importances_'):
                try:
                    importances = model.feature_importances_
                except Exception as e:
                    print(f"Warning: Could not get importance from {target_name}: {e}")
            
            # Only add if importances exist
            if importances is not None:
                max_features = min(len(feature_names), len(importances))
                if max_features > 0:
                    importance_dict[target_name] = {
                        feature_names[i]: float(importances[i])
                        for i in range(max_features)
                    }
                    importance_dict[target_name] = dict(
                        sorted(importance_dict[target_name].items(), 
                            key=lambda x: x[1], reverse=True)[:top_n]
                    )
            else:
                print(f"Note: No feature importance for {target_name}")
        
        return importance_dict
    
    def save_models(self, save_dir: str = "ml/trained_models"):
        """Save trained models to disk"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for target_name, model in self.models.items():
            model_path = save_path / f"{target_name}_predictor_v3.pkl"
            joblib.dump(model, model_path)
            print(f"Saved {target_name} model to {model_path}")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'targets': list(self.models.keys()),
            'version': 'v3'
        }
        joblib.dump(metadata, save_path / 'predictor_metadata_v3.pkl')
        
        print(f"\nAll models saved to {save_path}")
    
    def load_models(self, load_dir: str = "ml/trained_models"):
        """Load trained models from disk"""
        
        load_path = Path(load_dir)
        
        # Load metadata
        metadata = joblib.load(load_path / 'predictor_metadata_v3.pkl')
        self.model_type = metadata['model_type']
        self.is_trained = metadata['is_trained']
        
        # Load models
        for target_name in metadata['targets']:
            model_path = load_path / f"{target_name}_predictor_v3.pkl"
            self.models[target_name] = joblib.load(model_path)
            print(f"Loaded {target_name} model from {model_path}")
        
        print(f"\nAll models loaded from {load_path}")