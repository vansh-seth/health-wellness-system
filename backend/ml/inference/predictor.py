import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import joblib

class HealthInferenceEngine:
    """Real-time health predictions"""
    
    def __init__(self, models_dir: str = "ml/trained_models"):
        self.models_dir = Path(models_dir)
        self.predictor = None
        self.preprocessor = None
        self.feature_engineer = None
        self.loaded = False
    
    def load_models(self):
        """Load all trained models"""
        
        try:
            # Load predictor
            from ml.models.health_predictor import HealthPredictor
            self.predictor = HealthPredictor()
            self.predictor.load_models(str(self.models_dir))
            
            # Load preprocessor
            from ml.data.preprocessor import HealthDataPreprocessor
            self.preprocessor = HealthDataPreprocessor()
            self.preprocessor.load(str(self.models_dir / "preprocessor.pkl"))
            
            self.loaded = True
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.loaded = False
    
    def predict(self, user_data: Dict[str, Any]) -> Dict[str, float]:
        """Make real-time predictions for a user"""
        
        if not self.loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Prepare features from user data
        # This would involve feature engineering on the fly
        
        # Make predictions
        predictions = self.predictor.predict(user_features)
        
        return {
            'sleep_quality': float(predictions['sleep_quality'][0]),
            'mood_score': float(predictions['mood_score'][0]),
            'stress_level': float(predictions['stress_level'][0]),
            'exercise_probability': float(predictions['exercise_binary'][0])
        }