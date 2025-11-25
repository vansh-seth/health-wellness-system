# ml/data/synthetic_generator.py
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import json
from pathlib import Path

fake = Faker()

class HealthDataSynthesizer:
    def __init__(self, num_users: int = 1000, days_history: int = 90):
        self.num_users = num_users
        self.days_history = days_history
        self.users = []
        
    def generate_user_profiles(self) -> List[Dict]:
        """Generate synthetic user profiles with realistic health characteristics"""
        profiles = []
        
        for user_id in range(self.num_users):
            age = np.random.randint(18, 75)
            gender = random.choice(['male', 'female', 'other'])
            
            # Generate height and weight with realistic correlations
            if gender == 'male':
                height = np.random.normal(175, 8)  # cm
                base_weight = (height - 100) * 0.9  # rough estimate
            elif gender == 'female':
                height = np.random.normal(162, 7)  # cm
                base_weight = (height - 100) * 0.85
            else:
                height = np.random.normal(168, 10)
                base_weight = (height - 100) * 0.87
            
            # Add some variation to weight
            weight = np.random.normal(base_weight, base_weight * 0.15)
            weight = max(45, min(150, weight))  # reasonable bounds
            
            profile = {
                'user_id': f"user_{user_id:04d}",
                'age': age,
                'gender': gender,
                'height': round(height, 1),
                'weight': round(weight, 1),
                'activity_level': random.choice(['sedentary', 'lightly_active', 'moderately_active', 'very_active']),
                'health_goals': random.choices(
                    ['weight_loss', 'muscle_gain', 'maintain_health', 'improve_fitness', 'reduce_stress'],
                    k=random.randint(1, 3)
                ),
                'medical_conditions': random.choices(
                    ['diabetes', 'hypertension', 'anxiety', 'depression', 'arthritis', 'none'],
                    weights=[0.1, 0.15, 0.12, 0.08, 0.05, 0.5],
                    k=1
                ),
                'sleep_preference': np.random.normal(8, 1),  # preferred sleep hours
                'exercise_preference': random.choice(['cardio', 'strength', 'flexibility', 'mixed']),
                'stress_baseline': np.random.randint(3, 7)  # baseline stress level
            }
            profiles.append(profile)
            
        self.users = profiles
        return profiles
    
    def generate_meal_data(self, user_profile: Dict, date: datetime) -> List[Dict]:
        """Generate realistic meal data for a user on a specific date"""
        meals = []
        
        # Activity level affects calorie intake
        activity_multiplier = {
            'sedentary': 0.8,
            'lightly_active': 1.0,
            'moderately_active': 1.2,
            'very_active': 1.4
        }
        
        base_calories = 2000 if user_profile['gender'] == 'male' else 1800
        daily_calories = base_calories * activity_multiplier[user_profile['activity_level']]
        
        # Weekend effect - people tend to eat more on weekends
        if date.weekday() >= 5:
            daily_calories *= 1.15
        
        # Distribute calories across meals
        meal_distribution = {
            'breakfast': 0.25,
            'lunch': 0.35,
            'dinner': 0.30,
            'snack': 0.10
        }
        
        for meal_type, calorie_ratio in meal_distribution.items():
            # Skip some meals randomly
            if random.random() < 0.15:  # 15% chance to skip a meal
                continue
                
            calories = int(daily_calories * calorie_ratio * np.random.normal(1, 0.2))
            calories = max(50, min(1500, calories))
            
            # Macro distribution
            protein_ratio = np.random.normal(0.25, 0.05)
            fat_ratio = np.random.normal(0.30, 0.05)
            carb_ratio = 1 - protein_ratio - fat_ratio
            
            meals.append({
                'user_id': user_profile['user_id'],
                'date': date.strftime('%Y-%m-%d'),
                'meal_type': meal_type,
                'meal_name': self._generate_meal_name(meal_type),
                'calories': calories,
                'protein': round(calories * protein_ratio / 4, 1),  # 4 cal/g
                'carbs': round(calories * carb_ratio / 4, 1),
                'fat': round(calories * fat_ratio / 9, 1),  # 9 cal/g
                'timestamp': date.replace(
                    hour=self._get_meal_hour(meal_type),
                    minute=random.randint(0, 59)
                ).isoformat()
            })
            
        return meals
    
    def generate_exercise_data(self, user_profile: Dict, date: datetime, 
                              recent_sleep_quality: float = 7.0,
                              recent_stress: float = 5.0) -> List[Dict]:
        """Generate realistic exercise data with correlations"""
        exercises = []
        
        # Exercise frequency based on activity level
        exercise_prob = {
            'sedentary': 0.2,
            'lightly_active': 0.4,
            'moderately_active': 0.6,
            'very_active': 0.8
        }
        
        base_prob = exercise_prob[user_profile['activity_level']]
        
        # Poor sleep reduces exercise probability
        if recent_sleep_quality < 6:
            base_prob *= 0.7
        
        # High stress reduces exercise probability
        if recent_stress > 7:
            base_prob *= 0.8
        
        # Weekend effect - more likely to exercise
        if date.weekday() >= 5:
            base_prob *= 1.2
        
        if random.random() > base_prob:
            return exercises
        
        # Number of exercises per day
        num_exercises = np.random.choice([1, 2], p=[0.8, 0.2])
        
        for _ in range(num_exercises):
            exercise_type = random.choice(['cardio', 'strength', 'flexibility', 'sports'])
            duration = int(np.random.normal(45, 15))
            duration = max(10, min(180, duration))
            
            # Calories burned based on duration and intensity
            calories_per_min = {
                'cardio': np.random.normal(8, 2),
                'strength': np.random.normal(6, 1.5),
                'flexibility': np.random.normal(3, 0.5),
                'sports': np.random.normal(10, 3)
            }
            
            calories_burned = int(duration * calories_per_min[exercise_type])
            calories_burned = max(50, calories_burned)
            
            exercises.append({
                'user_id': user_profile['user_id'],
                'date': date.strftime('%Y-%m-%d'),
                'exercise_name': self._generate_exercise_name(exercise_type),
                'exercise_type': exercise_type,
                'duration_minutes': duration,
                'calories_burned': calories_burned,
                'timestamp': date.replace(
                    hour=random.randint(6, 22),
                    minute=random.randint(0, 59)
                ).isoformat()
            })
            
        return exercises
    
    def generate_sleep_data(self, user_profile: Dict, date: datetime,
                           recent_exercise: bool = False,
                           recent_stress: float = 5.0) -> Dict:
        """Generate realistic sleep data with correlations"""
        preferred_sleep = user_profile['sleep_preference']
        
        # Add some daily variation
        actual_sleep = np.random.normal(preferred_sleep, 0.8)
        actual_sleep = max(4, min(12, actual_sleep))
        
        # Weekend effect - slightly more sleep
        if date.weekday() >= 5:
            actual_sleep += 0.3
        
        # Bedtime
        preferred_bedtime = 23  # 11 PM
        bedtime_hour = int(np.random.normal(preferred_bedtime, 1.5))
        bedtime_hour = max(20, min(26, bedtime_hour)) % 24
        
        bedtime = date.replace(hour=bedtime_hour, minute=random.randint(0, 59))
        wake_time = bedtime + timedelta(hours=actual_sleep)
        
        # Sleep quality affected by various factors
        base_quality = 7
        
        # Age effect
        if user_profile['age'] > 65:
            base_quality -= 1
        
        # Medical conditions
        if 'anxiety' in user_profile['medical_conditions']:
            base_quality -= 1.5
        
        # Sleep duration impact
        if actual_sleep < 6:
            base_quality -= 2
        elif actual_sleep > 9:
            base_quality -= 0.5
        
        # Exercise improves sleep quality
        if recent_exercise:
            base_quality += 1.0
        
        # High stress reduces sleep quality
        if recent_stress > 7:
            base_quality -= 1.5
        
        # Weekend effect - slightly better sleep
        if date.weekday() >= 5:
            base_quality += 0.5
            
        quality = int(np.random.normal(base_quality, 1.5))
        quality = max(1, min(10, quality))
        
        return {
            'user_id': user_profile['user_id'],
            'date': date.strftime('%Y-%m-%d'),
            'bedtime': bedtime.isoformat(),
            'wake_time': wake_time.isoformat(),
            'sleep_duration_hours': round(actual_sleep, 1),
            'sleep_quality': quality,
            'timestamp': bedtime.isoformat()
        }
    
    def generate_mood_data(self, user_profile: Dict, date: datetime,
                          recent_sleep_quality: float = 7.0,
                          recent_exercise: bool = False) -> Dict:
        """Generate realistic mood data with correlations"""
        base_mood = 7
        base_stress = user_profile['stress_baseline']
        base_energy = 7
        
        # Medical conditions
        if 'depression' in user_profile['medical_conditions']:
            base_mood -= 2
        if 'anxiety' in user_profile['medical_conditions']:
            base_stress += 2
        
        # Weekly patterns (weekend vs weekday)
        if date.weekday() >= 5:  # Weekend
            base_mood += 1.0
            base_stress -= 1.5
        
        # Monday effect - higher stress
        if date.weekday() == 0:
            base_stress += 1.0
            base_mood -= 0.5
        
        # Sleep quality strongly affects mood and energy
        if recent_sleep_quality < 6:
            base_mood -= 2.0
            base_energy -= 2.5
            base_stress += 1.5
        elif recent_sleep_quality >= 8:
            base_mood += 1.5
            base_energy += 1.5
            base_stress -= 0.5
        
        # Exercise improves mood and energy, reduces stress
        if recent_exercise:
            base_mood += 1.5
            base_energy += 1.0
            base_stress -= 1.5
        
        # Add daily variation
        mood = int(np.random.normal(base_mood, 1.2))
        stress = int(np.random.normal(base_stress, 1.0))
        energy = int(np.random.normal(base_energy, 1.1))
        
        # Additional correlations
        if mood > 8:
            energy += 1
        if stress > 7:
            mood -= 1
            energy -= 1
            
        # Bounds
        mood = max(1, min(10, mood))
        stress = max(1, min(10, stress))
        energy = max(1, min(10, energy))
        
        notes = self._generate_mood_notes(mood, stress, energy)
        
        return {
            'user_id': user_profile['user_id'],
            'date': date.strftime('%Y-%m-%d'),
            'mood_score': mood,
            'stress_level': stress,
            'energy_level': energy,
            'notes': notes,
            'timestamp': date.replace(
                hour=random.randint(18, 23),  # Usually logged in evening
                minute=random.randint(0, 59)
            ).isoformat()
        }
    
    def generate_complete_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate complete synthetic dataset with realistic correlations"""
        print("Generating user profiles...")
        self.generate_user_profiles()
        
        all_meals = []
        all_exercises = []
        all_sleep = []
        all_mood = []
        
        start_date = datetime.now() - timedelta(days=self.days_history)
        
        print(f"Generating {self.days_history} days of data for {self.num_users} users...")
        
        for user_idx, user in enumerate(self.users):
            if user_idx % 100 == 0:
                print(f"  Processing user {user_idx}/{self.num_users}")
            
            # Track recent history for correlations
            user_sleep_history = []
            user_exercise_history = []
            user_stress_history = []
            
            for day in range(self.days_history):
                current_date = start_date + timedelta(days=day)
                
                # Get recent context (last 3 days)
                recent_sleep_quality = np.mean([s['sleep_quality'] for s in user_sleep_history[-3:]]) if user_sleep_history else 7.0
                recent_stress = np.mean([m['stress_level'] for m in user_stress_history[-3:]]) if user_stress_history else user['stress_baseline']
                recent_exercise = len(user_exercise_history[-3:]) > 0 if user_exercise_history else False
                
                # Generate correlated data
                meals = self.generate_meal_data(user, current_date)
                exercises = self.generate_exercise_data(user, current_date, recent_sleep_quality, recent_stress)
                sleep_data = self.generate_sleep_data(user, current_date, len(exercises) > 0, recent_stress)
                mood_data = self.generate_mood_data(user, current_date, recent_sleep_quality, len(exercises) > 0)
                
                all_meals.extend(meals)
                all_exercises.extend(exercises)
                all_sleep.append(sleep_data)
                all_mood.append(mood_data)
                
                # Update history
                user_sleep_history.append(sleep_data)
                user_exercise_history.extend(exercises)
                user_stress_history.append(mood_data)
        
        datasets = {
            'users': pd.DataFrame(self.users),
            'meals': pd.DataFrame(all_meals),
            'exercises': pd.DataFrame(all_exercises),
            'sleep': pd.DataFrame(all_sleep),
            'mood': pd.DataFrame(all_mood)
        }
        
        return datasets
    
    def _generate_meal_name(self, meal_type: str) -> str:
        meal_names = {
            'breakfast': ['Oatmeal with Berries', 'Scrambled Eggs', 'Greek Yogurt', 'Pancakes', 'Toast with Avocado'],
            'lunch': ['Chicken Salad', 'Pasta Bolognese', 'Sandwich', 'Soup and Bread', 'Rice Bowl'],
            'dinner': ['Grilled Salmon', 'Beef Stir Fry', 'Pizza', 'Vegetable Curry', 'Chicken Breast'],
            'snack': ['Apple', 'Nuts', 'Protein Bar', 'Yogurt', 'Crackers']
        }
        return random.choice(meal_names[meal_type])
    
    def _generate_exercise_name(self, exercise_type: str) -> str:
        exercise_names = {
            'cardio': ['Morning Run', 'Cycling', 'Swimming', 'Elliptical', 'Walking'],
            'strength': ['Weight Lifting', 'Push-ups', 'Squats', 'Deadlifts', 'Bench Press'],
            'flexibility': ['Yoga', 'Stretching', 'Pilates', 'Tai Chi', 'Foam Rolling'],
            'sports': ['Tennis', 'Basketball', 'Soccer', 'Volleyball', 'Badminton']
        }
        return random.choice(exercise_names[exercise_type])
    
    def _get_meal_hour(self, meal_type: str) -> int:
        hours = {
            'breakfast': random.randint(7, 10),
            'lunch': random.randint(12, 14),
            'dinner': random.randint(18, 21),
            'snack': random.randint(15, 17)
        }
        return hours[meal_type]
    
    def _generate_mood_notes(self, mood: int, stress: int, energy: int) -> str:
        if mood >= 8:
            return random.choice(["Feeling great today!", "Had a wonderful day", "Everything went well", ""])
        elif mood <= 3:
            return random.choice(["Tough day", "Feeling overwhelmed", "Not my best day", "Could be better"])
        elif stress >= 8:
            return random.choice(["Very stressful day", "Work pressure", "Too many deadlines", ""])
        else:
            return ""

def save_synthetic_data(output_dir: str = "ml/data/synthetic"):
    """Generate and save synthetic data"""
    generator = HealthDataSynthesizer(num_users=1000, days_history=90)
    datasets = generator.generate_complete_dataset()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    for name, df in datasets.items():
        filepath = Path(output_dir) / f"{name}.csv"
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} records to {filepath}")
    
    print(f"Synthetic data generation complete! Files saved in {output_dir}")
    return datasets

if __name__ == "__main__":
    datasets = save_synthetic_data()
    
    # Print summary
    print("\nDataset Summary:")
    for name, df in datasets.items():
        print(f"{name}: {len(df)} records")
        if len(df) > 0:
            print(f"  Columns: {list(df.columns)}")
        print()