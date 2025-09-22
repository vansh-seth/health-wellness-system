# main.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
import jwt
import bcrypt
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Health & Wellness API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
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

# Pydantic Models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    age: int
    gender: str
    height: float  # cm
    weight: float  # kg
    activity_level: str  # sedentary, lightly_active, moderately_active, very_active
    health_goals: List[str]  # weight_loss, muscle_gain, maintain_health, etc.
    medical_conditions: Optional[List[str]] = []
    allergies: Optional[List[str]] = []

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class HealthLog(BaseModel):
    log_type: str  # meal, exercise, sleep, mood
    date: datetime
    data: dict

class MealLog(BaseModel):
    meal_name: str
    calories: int
    protein: Optional[float] = 0
    carbs: Optional[float] = 0
    fat: Optional[float] = 0
    meal_type: str  # breakfast, lunch, dinner, snack

class ExerciseLog(BaseModel):
    exercise_name: str
    duration_minutes: int
    calories_burned: Optional[int] = 0
    exercise_type: str  # cardio, strength, flexibility, sports

class SleepLog(BaseModel):
    bedtime: datetime
    wake_time: datetime
    sleep_quality: int  # 1-10 scale

class MoodLog(BaseModel):
    mood_score: int  # 1-10 scale
    stress_level: int  # 1-10 scale
    energy_level: int  # 1-10 scale
    notes: Optional[str] = ""

# Helper functions
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

# Routes
@app.get("/")
async def root():
    return {"message": "Health & Wellness API is running!"}

@app.post("/auth/register")
async def register_user(user: UserRegister):
    # Check if user exists
    existing_user = users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create user
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
    
    # Remove password from response
    user.pop("password", None)
    user["_id"] = str(user["_id"])
    return user

@app.put("/profile")
async def update_profile(updates: dict, current_user: str = Depends(get_current_user)):
    # Remove sensitive fields
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
    # Calculate sleep duration
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

@app.get("/health/logs")
async def get_health_logs(
    log_type: Optional[str] = None,
    days: int = 7,
    current_user: str = Depends(get_current_user)
):
    # Get logs from last N days
    start_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
    
    query = {
        "user_id": current_user,
        "date": {"$gte": start_date}
    }
    
    if log_type:
        query["log_type"] = log_type
    
    logs = list(health_logs_collection.find(query).sort("timestamp", -1))
    
    # Convert ObjectId to string
    for log in logs:
        log["_id"] = str(log["_id"])
    
    return logs

@app.get("/dashboard/stats")
async def get_dashboard_stats(days: int = 7, current_user: str = Depends(get_current_user)):
    start_date = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
    
    # Get all logs from last N days
    logs = list(health_logs_collection.find({
        "user_id": current_user,
        "date": {"$gte": start_date}
    }))
    
    # Calculate stats
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
    
    # Calculate averages
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