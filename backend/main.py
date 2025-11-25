# backend/main.py - Fixed ML integration for inference
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
import jwt
import bcrypt
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from pathlib import Path

# ML imports
import sys
sys.path.append(str(Path(__file__).parent))

from ml.models.health_predictor_v2 import ImprovedHealthPredictor
from ml.models.anomaly_detector_v2 import HealthAnomalyDetector
from ml.models.recommendation_engine_v2 import (
    HealthRecommendationEngine,
    encode_categorical_features
)
from ml.data.feature_engineering_v2 import ImprovedHealthFeatureEngineer

load_dotenv()

app = FastAPI(title="AI-Powered Health & Wellness API", version="3.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-this")
JWT_ALGORITHM = "HS256"

client = MongoClient(MONGO_URL)
db = client.health_wellness
users_collection = db.users
health_logs_collection = db.health_logs

security = HTTPBearer()

# Global ML models
ml_models = {
    'predictor': None,
    'anomaly_detector': None,
    'recommendation_engine': None,
    'feature_engineer': None,
    'feature_names': None,
    'is_loaded': False
}

# Pydantic Models (keeping all existing models)
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    age: int
    gender: str
    height: float
    weight: float
    activity_level: str
    health_goals: List[str]
    medical_conditions: Optional[List[str]] = []
    allergies: Optional[List[str]] = []

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class MealLog(BaseModel):
    meal_name: str
    calories: int
    protein: Optional[float] = 0
    carbs: Optional[float] = 0
    fat: Optional[float] = 0
    meal_type: str

class ExerciseLog(BaseModel):
    exercise_name: str
    duration_minutes: int
    calories_burned: Optional[int] = 0
    exercise_type: str

class SleepLog(BaseModel):
    bedtime: datetime
    wake_time: datetime
    sleep_quality: int

class MoodLog(BaseModel):
    mood_score: int
    stress_level: int
    energy_level: int
    notes: Optional[str] = ""

# Helper functions (keeping all existing)
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_jwt_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# FIXED ML Helper Functions
def prepare_user_data_for_ml(user_id: str, days_lookback: int = 28) -> Optional[Dict]:
    """Prepare user data for ML models"""
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return None
        
        start_date = (datetime.utcnow() - timedelta(days=days_lookback)).date().isoformat()
        logs = list(health_logs_collection.find({
            "user_id": user_id,
            "date": {"$gte": start_date}
        }).sort("date", 1))
        
        if not logs:
            return None
        
        meals, exercises, sleep, mood = [], [], [], []
        
        for log in logs:
            base_data = {
                'user_id': user_id,
                'date': log['date'],
                'timestamp': log['timestamp'],
                **log['data']
            }
            
            if log['log_type'] == 'meal':
                meals.append(base_data)
            elif log['log_type'] == 'exercise':
                exercises.append(base_data)
            elif log['log_type'] == 'sleep':
                sleep.append(base_data)
            elif log['log_type'] == 'mood':
                mood.append(base_data)
        
        datasets = {
            'users': pd.DataFrame([{
                'user_id': user_id,
                'age': user['age'],
                'gender': user['gender'],
                'height': user['height'],
                'weight': user['weight'],
                'activity_level': user['activity_level']
            }]),
            'meals': pd.DataFrame(meals) if meals else pd.DataFrame(),
            'exercises': pd.DataFrame(exercises) if exercises else pd.DataFrame(),
            'sleep': pd.DataFrame(sleep) if sleep else pd.DataFrame(),
            'mood': pd.DataFrame(mood) if mood else pd.DataFrame()
        }
        
        return datasets
        
    except Exception as e:
        print(f"Error preparing user data: {e}")
        return None

def calculate_user_features_for_inference(user_id: str, datasets: Dict) -> Optional[pd.DataFrame]:
    """
    Calculate features for inference (NO target columns needed)
    """
    try:
        feature_engineer = ml_models['feature_engineer']
        if not feature_engineer:
            return None
        
        # Create aggregated user features
        user_features = feature_engineer.create_user_aggregated_features(datasets)
        
        # Create time series WITHOUT targets
        if len(datasets['sleep']) > 0 and len(datasets['mood']) > 0:
            # Create basic time series data
            sleep_df = datasets['sleep'].copy()
            mood_df = datasets['mood'].copy()
            
            # Merge sleep and mood by date
            daily_data = sleep_df.merge(
                mood_df[['date', 'mood_score', 'stress_level', 'energy_level']],
                on='date',
                how='outer'
            )
            
            # Add user_id
            daily_data['user_id'] = user_id
            
            # Sort by date
            daily_data = daily_data.sort_values('date')
            
            # Add rolling features manually (simpler version for inference)
            for col in ['sleep_quality', 'mood_score', 'stress_level']:
                if col in daily_data.columns:
                    daily_data[f'{col}_rolling_mean_7d'] = daily_data[col].rolling(7, min_periods=1).mean()
                    daily_data[f'{col}_rolling_std_7d'] = daily_data[col].rolling(7, min_periods=1).std()
            
            # Get most recent data point
            latest_data = daily_data.tail(1).copy()
            
            # Add temporal features
            if 'date' in latest_data.columns:
                latest_data['date'] = pd.to_datetime(latest_data['date'])
                latest_data['day_of_week'] = latest_data['date'].dt.dayofweek
                latest_data['day_of_month'] = latest_data['date'].dt.day
                latest_data['month'] = latest_data['date'].dt.month
                latest_data['is_weekend'] = (latest_data['day_of_week'] >= 5).astype(int)
            
            # Merge with user features
            final_features = latest_data.merge(user_features, on='user_id', how='left')
            
            # Add exercise data if available
            if len(datasets['exercises']) > 0:
                recent_exercises = datasets['exercises'].groupby('date').agg({
                    'duration_minutes': 'sum',
                    'calories_burned': 'sum'
                }).reset_index()
                
                if len(recent_exercises) > 0:
                    final_features['avg_exercise_duration'] = recent_exercises['duration_minutes'].mean()
                    final_features['total_calories_burned'] = recent_exercises['calories_burned'].sum()
            
            # Add meal data if available
            if len(datasets['meals']) > 0:
                recent_meals = datasets['meals'].groupby('date').agg({
                    'calories': 'sum'
                }).reset_index()
                
                if len(recent_meals) > 0:
                    final_features['avg_daily_calories'] = recent_meals['calories'].mean()
            
            return final_features
        
        return None
        
    except Exception as e:
        print(f"Error calculating features for inference: {e}")
        import traceback
        traceback.print_exc()
        return None

# Helper to patch model paths
def load_model_with_custom_path(model_instance, models_dir):
    """Load models from custom directory"""
    import pickle
    import joblib
    
    print(f"⚠️ Model trying to load from hardcoded path")
    print(f"   Loading from: {models_dir}")
    
    # Manually load all model files
    models_dir = Path(models_dir)
    
    # Load metadata
    metadata_file = models_dir / "predictor_metadata_v3.pkl"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Cannot find {metadata_file}")
    
    # Try loading metadata with both pickle and joblib
    try:
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
    except:
        metadata = joblib.load(metadata_file)
    
    # Initialize the models dictionary
    model_instance.models = {}
    model_instance.feature_names = metadata.get('feature_names', [])
    
    # Load each target's models - FIXED FILENAME PATTERN
    targets = ['sleep_quality', 'mood_score', 'stress_level', 'exercise_binary']
    
    for target in targets:
        # Try different naming patterns
        model_file = models_dir / f"{target}_predictor_v3.pkl"
        
        if model_file.exists():
            try:
                # Try joblib first (more common for sklearn models)
                try:
                    model_instance.models[target] = joblib.load(model_file)
                    print(f"   ✓ Loaded {target} model (joblib)")
                except:
                    # Fallback to pickle
                    with open(model_file, 'rb') as f:
                        model_instance.models[target] = pickle.load(f)
                    print(f"   ✓ Loaded {target} model (pickle)")
            except Exception as e:
                print(f"   ✗ Failed to load {target}: {e}")
        else:
            print(f"   ⚠️ Model file not found: {model_file}")
    
    if not model_instance.models:
        raise ValueError("No models were loaded successfully")
    
    # CRITICAL: Set the trained flag so predict() works
    model_instance.is_trained = True
    
    # Also set any other flags that might exist
    if hasattr(model_instance, '_is_fitted'):
        model_instance._is_fitted = True
    
    print(f"✅ Loaded {len(model_instance.models)} predictor models from custom path")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load ML models on startup"""
    global ml_models
    
    print("🚀 Starting AI-Powered Health & Wellness System...")
    
    # Try both possible model directories
    models_dir = Path("ml/trained_models_copy")
    if not models_dir.exists():
        models_dir = Path("ml/trained_models")
        if not models_dir.exists():
            print("⚠️ Trained models folder not found. Please run training scripts first.")
            print(f"   Tried: ml/trained_models_copy and ml/trained_models")
            return
    
    print(f"📁 Using models directory: {models_dir}")
    
    # Load predictor
    predictor_path = models_dir / "predictor_metadata_v3.pkl"
    print(f"🔍 Looking for predictor at: {predictor_path.absolute()}")
    print(f"   File exists: {predictor_path.exists()}")
    
    if predictor_path.exists():
        try:
            print("📊 Loading prediction model...")
            predictor = ImprovedHealthPredictor(model_type='stacking')
            
            # Try loading with custom path
            try:
                load_model_with_custom_path(predictor, models_dir)
                ml_models['predictor'] = predictor
                print("✅ Predictor loaded successfully")
            except Exception as inner_e:
                print(f"   Custom path loading failed: {inner_e}")
                # Fallback to original method
                predictor.load_models()
                ml_models['predictor'] = predictor
                print("✅ Predictor loaded successfully (standard path)")
                
        except Exception as e:
            print(f"❌ Failed to load predictor: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Predictor file not found at: {predictor_path}")
    
    # Load anomaly detector
    anomaly_path = models_dir / "anomaly_detector_v3.pkl"
    if anomaly_path.exists():
        try:
            print("🔍 Loading anomaly detector...")
            anomaly_detector = HealthAnomalyDetector(method='ensemble')
            ml_models['anomaly_detector'] = anomaly_detector
            print("✅ Anomaly detector loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load anomaly detector: {e}")
    
    # Load recommendation engine - FIXED: Try v2 filename
    rec_path_v3 = models_dir / "recommendation_engine_v3.pkl"
    rec_path_v2 = models_dir / "recommendation_engine_v2.pkl"
    rec_path = rec_path_v3 if rec_path_v3.exists() else rec_path_v2
    
    if rec_path.exists():
        try:
            print(f"🎯 Loading recommendation engine from {rec_path.name}...")
            rec_engine = HealthRecommendationEngine()
            
            # Try loading with custom path awareness
            try:
                # Manually load the recommendation engine
                import pickle
                with open(rec_path, 'rb') as f:
                    rec_data = pickle.load(f)
                
                # Set attributes
                if isinstance(rec_data, dict):
                    for key, value in rec_data.items():
                        setattr(rec_engine, key, value)
                    print("✅ Recommendation engine loaded successfully (custom)")
                else:
                    # If it's the object itself
                    rec_engine = rec_data
                    print("✅ Recommendation engine loaded successfully")
                    
                ml_models['recommendation_engine'] = rec_engine
                
            except Exception as custom_error:
                print(f"   Custom loading failed: {custom_error}")
                # Fallback to standard method
                rec_engine.load_model()
                ml_models['recommendation_engine'] = rec_engine
                print("✅ Recommendation engine loaded successfully (standard)")
                
        except Exception as e:
            print(f"❌ Failed to load recommendation engine: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Recommendation engine file missing: tried {rec_path_v3} and {rec_path_v2}")
    
    # Initialize feature engineer
    print("🔧 Initializing feature engineer...")
    ml_models['feature_engineer'] = ImprovedHealthFeatureEngineer()
    
    # Load feature names
    feature_names_path = models_dir / "feature_names_v3.txt"
    if feature_names_path.exists():
        with open(feature_names_path, 'r') as f:
            ml_models['feature_names'] = [line.strip() for line in f]
    
    ml_models['is_loaded'] = any([
        ml_models['predictor'], 
        ml_models['anomaly_detector'], 
        ml_models['recommendation_engine']
    ])
    
    print(f"✅ ML startup complete. Status: {'Ready' if ml_models['is_loaded'] else 'Not Ready'}")

# Authentication routes (keeping all existing)
@app.get("/")
async def root():
    # Check what model files exist
    model_files = []
    models_dir = Path("ml/trained_models_copy")
    if models_dir.exists():
        model_files = [f.name for f in models_dir.glob("*.pkl")]
    
    return {
        "message": "AI-Powered Health & Wellness API v3.0",
        "ml_status": "✅ Ready" if ml_models['is_loaded'] else "❌ Not Loaded",
        "loaded_models": [k for k, v in ml_models.items() if v and k not in ['feature_names', 'is_loaded']],
        "available_model_files": model_files,
        "features": [
            "Health Predictions",
            "Anomaly Detection",
            "Personalized Recommendations",
            "Multi-modal Data Fusion"
        ]
    }

@app.post("/auth/register")
async def register_user(user: UserRegister):
    existing_user = users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    user_data = user.dict()
    user_data["password"] = hash_password(user.password)
    user_data["created_at"] = datetime.utcnow()
    
    result = users_collection.insert_one(user_data)
    token = create_jwt_token(str(result.inserted_id))
    
    return {
        "message": "User created successfully",
        "token": token,
        "user_id": str(result.inserted_id)
    }

@app.post("/auth/login")
async def login_user(login_data: UserLogin):
    user = users_collection.find_one({"email": login_data.email})
    if not user or not verify_password(login_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_jwt_token(str(user["_id"]))
    return {
        "message": "Login successful",
        "token": token,
        "user_id": str(user["_id"])
    }

@app.get("/profile")
async def get_profile(current_user: str = Depends(get_current_user)):
    user = users_collection.find_one({"_id": ObjectId(current_user)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.pop("password", None)
    user["_id"] = str(user["_id"])
    return user

@app.put("/profile")
async def update_profile(updates: dict, current_user: str = Depends(get_current_user)):
    updates.pop("password", None)
    updates.pop("email", None)
    updates["updated_at"] = datetime.utcnow()
    
    result = users_collection.update_one(
        {"_id": ObjectId(current_user)},
        {"$set": updates}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "Profile updated successfully"}

# Health logging routes (keeping all existing)
@app.post("/health/log/meal")
async def log_meal(meal: MealLog, current_user: str = Depends(get_current_user)):
    log_data = {
        "user_id": current_user,
        "log_type": "meal",
        "timestamp": datetime.utcnow(),
        "date": datetime.utcnow().date().isoformat(),
        "data": meal.dict()
    }
    
    result = health_logs_collection.insert_one(log_data)
    return {"message": "Meal logged successfully", "log_id": str(result.inserted_id)}

@app.post("/health/log/exercise")
async def log_exercise(exercise: ExerciseLog, current_user: str = Depends(get_current_user)):
    log_data = {
        "user_id": current_user,
        "log_type": "exercise",
        "timestamp": datetime.utcnow(),
        "date": datetime.utcnow().date().isoformat(),
        "data": exercise.dict()
    }
    
    result = health_logs_collection.insert_one(log_data)
    return {"message": "Exercise logged successfully", "log_id": str(result.inserted_id)}

@app.post("/health/log/sleep")
async def log_sleep(sleep: SleepLog, current_user: str = Depends(get_current_user)):
    sleep_duration = (sleep.wake_time - sleep.bedtime).total_seconds() / 3600
    
    log_data = {
        "user_id": current_user,
        "log_type": "sleep",
        "timestamp": datetime.utcnow(),
        "date": sleep.bedtime.date().isoformat(),
        "data": {
            **sleep.dict(),
            "sleep_duration_hours": sleep_duration
        }
    }
    
    result = health_logs_collection.insert_one(log_data)
    return {"message": "Sleep logged successfully", "log_id": str(result.inserted_id)}

@app.post("/health/log/mood")
async def log_mood(mood: MoodLog, current_user: str = Depends(get_current_user)):
    log_data = {
        "user_id": current_user,
        "log_type": "mood",
        "timestamp": datetime.utcnow(),
        "date": datetime.utcnow().date().isoformat(),
        "data": mood.dict()
    }
    
    result = health_logs_collection.insert_one(log_data)
    return {"message": "Mood logged successfully", "log_id": str(result.inserted_id)}

# FIXED ML-POWERED ENDPOINTS

@app.get("/ai/predictions")
async def get_predictions(current_user: str = Depends(get_current_user)):
    """Get AI-powered health predictions"""
    
    if not ml_models['predictor']:
        raise HTTPException(status_code=503, detail="Prediction model not available")
    
    try:
        datasets = prepare_user_data_for_ml(current_user, days_lookback=28)
        if not datasets or len(datasets['sleep']) < 7:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data. Please log at least 7 days of sleep and mood data."
            )
        
        # Use FIXED feature calculation
        features_df = calculate_user_features_for_inference(current_user, datasets)
        if features_df is None or len(features_df) == 0:
            raise HTTPException(status_code=400, detail="Unable to calculate features")
        
        # Prepare features (remove non-feature columns)
        exclude_cols = ['user_id', 'date', 'timestamp', 'bedtime', 'wake_time']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X_df = features_df[feature_cols].copy()
        
        # CRITICAL: Encode categorical features BEFORE alignment
        categorical_cols = ['gender', 'activity_level', 'preferred_exercise_type', 'bmi_category']
        categorical_cols = [col for col in categorical_cols if col in X_df.columns]
        
        if categorical_cols:
            print(f"Encoding categorical columns: {categorical_cols}")
            X_df, _ = encode_categorical_features(X_df, categorical_cols)
        
        # Get expected feature names from the loaded model OR from ml_models
        predictor = ml_models['predictor']
        
        # Try to get feature names from multiple sources
        expected_features = None
        if hasattr(predictor, 'feature_names') and predictor.feature_names:
            expected_features = predictor.feature_names
        elif ml_models['feature_names']:
            expected_features = ml_models['feature_names']
        
        print(f"Current features: {len(X_df.columns)}")
        print(f"Expected features available: {expected_features is not None}")
        
        if expected_features and len(expected_features) > 0:
            # Align features with training
            print(f"Expected features: {len(expected_features)}, Got: {len(X_df.columns)}")
            print(f"Missing features will be filled with 0")
            
            # Add missing features with default values
            for feature in expected_features:
                if feature not in X_df.columns:
                    X_df[feature] = 0
            
            # Select only expected features in the correct order
            X_df = X_df[expected_features]
            print(f"After alignment: {len(X_df.columns)} features")
        else:
            print("No expected features found, using categorical encoding")
            # Encode categorical if present
            categorical_cols = ['gender', 'activity_level', 'preferred_exercise_type', 'bmi_category']
            categorical_cols = [col for col in categorical_cols if col in X_df.columns]
            
            if categorical_cols:
                X_encoded, _ = encode_categorical_features(X_df, categorical_cols)
                X_df = X_encoded
        
        # Fill NaN values
        X_df = X_df.fillna(0)
        X = X_df.values
        print(f"Final feature shape: {X.shape}")
        
        # Make predictions
        predictions = predictor.predict(X)
        
        return {
            "predictions": {
                "sleep_quality": {
                    "value": float(predictions['sleep_quality'][0]),
                    "unit": "1-10 scale",
                    "confidence": "high"
                },
                "mood_score": {
                    "value": float(predictions['mood_score'][0]),
                    "unit": "1-10 scale",
                    "confidence": "high"
                },
                "stress_level": {
                    "value": float(predictions['stress_level'][0]),
                    "unit": "1-10 scale",
                    "confidence": "high"
                },
                "exercise_probability": {
                    "value": float(predictions['exercise_binary'][0]),
                    "unit": "probability",
                    "confidence": "high"
                }
            },
            "prediction_date": (datetime.utcnow() + timedelta(days=1)).date().isoformat(),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/ai/recommendations")
async def get_recommendations(current_user: str = Depends(get_current_user)):
    """Get personalized health recommendations"""
    
    if not ml_models['recommendation_engine']:
        raise HTTPException(status_code=503, detail="Recommendation engine not available")
    
    try:
        user = users_collection.find_one({"_id": ObjectId(current_user)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        datasets = prepare_user_data_for_ml(current_user, days_lookback=28)
        rec_engine = ml_models['recommendation_engine']
        
        # Get predictions if possible
        if datasets and len(datasets['sleep']) >= 7 and ml_models['predictor']:
            try:
                features_df = calculate_user_features_for_inference(current_user, datasets)
                if features_df is not None:
                    exclude_cols = ['user_id', 'date', 'timestamp', 'bedtime', 'wake_time']
                    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
                    X_df = features_df[feature_cols].copy()
                    
                    categorical_cols = ['gender', 'activity_level', 'preferred_exercise_type', 'bmi_category']
                    categorical_cols = [col for col in categorical_cols if col in X_df.columns]
                    
                    if categorical_cols:
                        X_encoded, _ = encode_categorical_features(X_df, categorical_cols)
                        X = X_encoded.values
                    else:
                        X = X_df.fillna(0).values
                    
                    predictor = ml_models['predictor']
                    predictions = predictor.predict(X)
                else:
                    predictions = {
                        'sleep_quality': [7.0],
                        'mood_score': [7.0],
                        'stress_level': [5.0],
                        'exercise_binary': [0.5]
                    }
            except:
                predictions = {
                    'sleep_quality': [7.0],
                    'mood_score': [7.0],
                    'stress_level': [5.0],
                    'exercise_binary': [0.5]
                }
        else:
            predictions = {
                'sleep_quality': [7.0],
                'mood_score': [7.0],
                'stress_level': [5.0],
                'exercise_binary': [0.5]
            }
        
        # Generate recommendations
        user_features = {
            current_user: {
                'age': user['age'],
                'gender': user['gender'],
                'bmi': user['weight'] / ((user['height']/100) ** 2),
                'activity_level': user['activity_level'],
                'avg_daily_calories': datasets['meals']['calories'].sum() / 7 if datasets and len(datasets['meals']) > 0 else 2000
            }
        }
        
        recommendations = rec_engine.generate_personalized_recommendations(
            current_user,
            user_features,
            predictions,
            num_recommendations=5
        )
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Recommendation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@app.get("/ai/anomalies")
async def detect_anomalies(current_user: str = Depends(get_current_user)):
    """Detect health anomalies"""
    
    if not ml_models['anomaly_detector']:
        raise HTTPException(status_code=503, detail="Anomaly detector not available")
    
    try:
        datasets = prepare_user_data_for_ml(current_user, days_lookback=28)
        if not datasets or len(datasets['sleep']) < 14:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for anomaly detection (need at least 14 days)"
            )
        
        features_df = calculate_user_features_for_inference(current_user, datasets)
        if features_df is None:
            raise HTTPException(status_code=400, detail="Unable to calculate features")
        
        # Prepare features
        exclude_cols = ['user_id', 'date', 'timestamp', 'bedtime', 'wake_time']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X_df = features_df[feature_cols].copy()
        
        categorical_cols = ['gender', 'activity_level', 'preferred_exercise_type', 'bmi_category']
        categorical_cols = [col for col in categorical_cols if col in X_df.columns]
        
        if categorical_cols:
            X_encoded, _ = encode_categorical_features(X_df, categorical_cols)
            X = X_encoded.values
        else:
            X = X_df.fillna(0).values
        
        # Load and use anomaly detector
        anomaly_detector = ml_models['anomaly_detector']
        anomaly_detector.load_model(input_dim=X.shape[1])
        
        results = anomaly_detector.detect_anomalies(X)
        
        # Get explanations if anomalies found
        explanations = []
        if results['is_anomaly'][-1] == 1:
            if ml_models['feature_names']:
                exp = anomaly_detector.explain_anomaly(
                    X[-1:],
                    ml_models['feature_names'][:X.shape[1]],
                    top_n=5
                )
                if exp:
                    insights = anomaly_detector.get_health_insights(exp[0])
                    explanations = insights
        
        return {
            "is_anomaly": bool(results['is_anomaly'][-1]),
            "anomaly_score": float(results['anomaly_scores'][-1]),
            "severity": results['severity'][-1],
            "insights": explanations,
            "detected_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Anomaly detection error: {str(e)}")

@app.get("/health/logs")
async def get_health_logs(
    log_type: Optional[str] = None,
    days: int = 7,
    current_user: str = Depends(get_current_user)
):
    start_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
    
    query = {
        "user_id": current_user,
        "date": {"$gte": start_date}
    }
    
    if log_type:
        query["log_type"] = log_type
    
    logs = list(health_logs_collection.find(query).sort("timestamp", -1))
    
    for log in logs:
        log["_id"] = str(log["_id"])
    
    return logs

@app.get("/dashboard/stats")
async def get_dashboard_stats(days: int = 7, current_user: str = Depends(get_current_user)):
    start_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
    
    logs = list(health_logs_collection.find({
        "user_id": current_user,
        "date": {"$gte": start_date}
    }))
    
    stats = {
        "total_logs": len(logs),
        "meals_logged": len([l for l in logs if l["log_type"] == "meal"]),
        "exercises_logged": len([l for l in logs if l["log_type"] == "exercise"]),
        "sleep_logs": len([l for l in logs if l["log_type"] == "sleep"]),
        "mood_logs": len([l for l in logs if l["log_type"] == "mood"]),
        "avg_sleep_hours": 0,
        "avg_mood_score": 0,
        "total_calories_consumed": 0,
        "total_calories_burned": 0
    }
    
    sleep_logs = [l for l in logs if l["log_type"] == "sleep"]
    if sleep_logs:
        stats["avg_sleep_hours"] = sum(l["data"].get("sleep_duration_hours", 0) for l in sleep_logs) / len(sleep_logs)
    
    mood_logs = [l for l in logs if l["log_type"] == "mood"]
    if mood_logs:
        stats["avg_mood_score"] = sum(l["data"].get("mood_score", 0) for l in mood_logs) / len(mood_logs)
    
    meal_logs = [l for l in logs if l["log_type"] == "meal"]
    stats["total_calories_consumed"] = sum(l["data"].get("calories", 0) for l in meal_logs)
    
    exercise_logs = [l for l in logs if l["log_type"] == "exercise"]
    stats["total_calories_burned"] = sum(l["data"].get("calories_burned", 0) for l in exercise_logs)
    
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)