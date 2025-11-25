import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.data.synthetic_generator import save_synthetic_data
from ml.data.preprocessor import HealthDataPreprocessor
from ml.data.feature_engineering import create_ml_features
from ml.models.health_predictor import HealthPredictor
from ml.models.fusion_model import HealthDataFusion
from ml.models.recommendation_engine import create_recommendation_system
from ml.models.anomaly_detector import HealthAnomalyDetector
from sklearn.model_selection import train_test_split

def train_all_models(use_existing_data: bool = False, save_models: bool = True):
    """Complete training pipeline for all models"""
    
    print("="*60)
    print("Health & Wellness ML Training Pipeline")
    print("="*60)
    
    # Step 1: Generate or load data
    if not use_existing_data:
        print("\n[1/6] Generating synthetic training data...")
        datasets = save_synthetic_data()
    else:
        print("\n[1/6] Loading existing data...")
        # Load from CSV files
        data_dir = Path("ml/data/synthetic")
        datasets = {
            'users': pd.read_csv(data_dir / 'users.csv'),
            'meals': pd.read_csv(data_dir / 'meals.csv'),
            'exercises': pd.read_csv(data_dir / 'exercises.csv'),
            'sleep': pd.read_csv(data_dir / 'sleep.csv'),
            'mood': pd.read_csv(data_dir / 'mood.csv')
        }
    
    # Step 2: Preprocess data
    print("\n[2/6] Preprocessing data...")
    preprocessor = HealthDataPreprocessor()
    transformed_datasets = preprocessor.fit_transform(datasets)
    
    if save_models:
        preprocessor.save()
    
    # Step 3: Feature engineering
    print("\n[3/6] Engineering features...")
    ml_features = create_ml_features(datasets)
    
    X = ml_features['X_train']
    y_dict = {
        'sleep_quality': ml_features['y_train'][:, 0],
        'mood_score': ml_features['y_train'][:, 1],
        'stress_level': ml_features['y_train'][:, 2],
        'exercise_binary': ml_features['y_train'][:, 3]
    }
    feature_names = ml_features['feature_names']
    
    # Split data
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    y_train = {k: v[:len(X_train)] for k, v in y_dict.items()}
    y_val = {k: v[len(X_train):] for k, v in y_dict.items()}
    
    # Step 4: Train health predictor
    print("\n[4/6] Training health predictor...")
    predictor = HealthPredictor(model_type='ensemble')
    predictor.build_models(X.shape[1])
    predictor.train(X_train, y_train, X_val, y_val)
    
    if save_models:
        predictor.save_models()
    
    # Step 5: Train fusion model
    print("\n[5/6] Training multi-modal fusion model...")
    fusion = HealthDataFusion()
    fusion.train_fusion_model(ml_features['final_features'], epochs=50)
    
    # Step 6: Train recommendation engine
    print("\n[6/6] Training recommendation engine...")
    rec_engine = create_recommendation_system(datasets, ml_features)
    
    # Train anomaly detector
    print("\n[Bonus] Training anomaly detector...")
    anomaly_detector = HealthAnomalyDetector(method='ensemble')
    anomaly_detector.build_models(X.shape[1])
    anomaly_detector.train(X_train, epochs=50)
    
    if save_models:
        anomaly_detector.save_model()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    return {
        'predictor': predictor,
        'fusion': fusion,
        'recommendation_engine': rec_engine,
        'anomaly_detector': anomaly_detector,
        'preprocessor': preprocessor,
        'feature_names': feature_names
    }

if __name__ == "__main__":
    models = train_all_models(use_existing_data=False, save_models=True)
    print("\nAll models trained and saved successfully!")