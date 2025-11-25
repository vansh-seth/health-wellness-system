# ml/data/feature_engineering.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

class HealthFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """Create temporal features from date column"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic temporal features
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['month'] = df[date_col].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['day_of_year'] = df[date_col].dt.dayofyear
        
        # Cyclical encoding for temporal features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_user_aggregated_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create aggregated features per user from all data sources"""
        user_features = []
        
        users = datasets['users']['user_id'].unique()
        
        for user_id in users:
            user_data = {'user_id': user_id}
            
            # User profile features
            user_profile = datasets['users'][datasets['users']['user_id'] == user_id].iloc[0]
            user_data.update({
                'age': user_profile['age'],
                'gender': user_profile['gender'],
                'height': user_profile['height'],
                'weight': user_profile['weight'],
                'bmi': user_profile['weight'] / ((user_profile['height'] / 100) ** 2),
                'activity_level': user_profile['activity_level']
            })
            
            # Meal features
            user_meals = datasets['meals'][datasets['meals']['user_id'] == user_id]
            if len(user_meals) > 0:
                user_data.update({
                    'avg_daily_calories': user_meals['calories'].mean(),
                    'avg_protein': user_meals['protein'].mean(),
                    'avg_carbs': user_meals['carbs'].mean(),
                    'avg_fat': user_meals['fat'].mean(),
                    'meals_per_day': len(user_meals) / len(user_meals['date'].unique()) if len(user_meals['date'].unique()) > 0 else 0,
                    'calories_std': user_meals['calories'].std() or 0,
                    'meal_regularity': self._calculate_meal_regularity(user_meals)
                })
            else:
                user_data.update({
                    'avg_daily_calories': 0, 'avg_protein': 0, 'avg_carbs': 0, 'avg_fat': 0,
                    'meals_per_day': 0, 'calories_std': 0, 'meal_regularity': 0
                })
            
            # Exercise features
            user_exercises = datasets['exercises'][datasets['exercises']['user_id'] == user_id]
            if len(user_exercises) > 0:
                user_data.update({
                    'avg_exercise_duration': user_exercises['duration_minutes'].mean(),
                    'avg_calories_burned': user_exercises['calories_burned'].mean(),
                    'exercise_frequency': len(user_exercises) / 90,  # exercises per day over 90 days
                    'exercise_consistency': self._calculate_exercise_consistency(user_exercises),
                    'preferred_exercise_type': user_exercises['exercise_type'].mode().iloc[0] if len(user_exercises) > 0 else 'none'
                })
            else:
                user_data.update({
                    'avg_exercise_duration': 0, 'avg_calories_burned': 0, 'exercise_frequency': 0,
                    'exercise_consistency': 0, 'preferred_exercise_type': 'none'
                })
            
            # Sleep features
            user_sleep = datasets['sleep'][datasets['sleep']['user_id'] == user_id]
            if len(user_sleep) > 0:
                user_data.update({
                    'avg_sleep_duration': user_sleep['sleep_duration_hours'].mean(),
                    'avg_sleep_quality': user_sleep['sleep_quality'].mean(),
                    'sleep_regularity': 1 - (user_sleep['sleep_duration_hours'].std() / user_sleep['sleep_duration_hours'].mean()) if user_sleep['sleep_duration_hours'].mean() > 0 else 0,
                    'sleep_quality_trend': self._calculate_trend(user_sleep['sleep_quality'].values)
                })
            else:
                user_data.update({
                    'avg_sleep_duration': 0, 'avg_sleep_quality': 0, 'sleep_regularity': 0, 'sleep_quality_trend': 0
                })
            
            # Mood features
            user_mood = datasets['mood'][datasets['mood']['user_id'] == user_id]
            if len(user_mood) > 0:
                user_data.update({
                    'avg_mood_score': user_mood['mood_score'].mean(),
                    'avg_stress_level': user_mood['stress_level'].mean(),
                    'avg_energy_level': user_mood['energy_level'].mean(),
                    'mood_variability': user_mood['mood_score'].std() or 0,
                    'mood_trend': self._calculate_trend(user_mood['mood_score'].values),
                    'stress_trend': self._calculate_trend(user_mood['stress_level'].values)
                })
            else:
                user_data.update({
                    'avg_mood_score': 0, 'avg_stress_level': 0, 'avg_energy_level': 0,
                    'mood_variability': 0, 'mood_trend': 0, 'stress_trend': 0
                })
            
            user_features.append(user_data)
        
        return pd.DataFrame(user_features)
    
    def create_time_series_features(self, datasets: Dict[str, pd.DataFrame], lookback_days: int = 7) -> pd.DataFrame:
        """Create time-series features with lookback windows"""
        time_series_features = []
        
        users = datasets['users']['user_id'].unique()
        
        for user_id in users:
            # Get user data
            user_meals = datasets['meals'][datasets['meals']['user_id'] == user_id].copy()
            user_exercises = datasets['exercises'][datasets['exercises']['user_id'] == user_id].copy()
            user_sleep = datasets['sleep'][datasets['sleep']['user_id'] == user_id].copy()
            user_mood = datasets['mood'][datasets['mood']['user_id'] == user_id].copy()
            
            # Convert dates
            for df in [user_meals, user_exercises, user_sleep, user_mood]:
                if len(df) > 0:
                    df['date'] = pd.to_datetime(df['date'])
            
            # Get unique dates
            all_dates = set()
            for df in [user_meals, user_exercises, user_sleep, user_mood]:
                if len(df) > 0:
                    all_dates.update(df['date'].unique())
            
            all_dates = sorted(list(all_dates))
            
            # Create features for each date
            for date in all_dates:
                if date < min(all_dates) + timedelta(days=lookback_days):
                    continue  # Not enough history
                
                lookback_start = date - timedelta(days=lookback_days)
                
                feature_row = {
                    'user_id': user_id,
                    'date': date,
                    'target_date': date + timedelta(days=1)  # Predicting next day
                }
                
                # Lookback features for meals
                meals_lookback = user_meals[
                    (user_meals['date'] >= lookback_start) & 
                    (user_meals['date'] < date)
                ]
                
                if len(meals_lookback) > 0:
                    feature_row.update({
                        'calories_7d_avg': meals_lookback['calories'].mean(),
                        'calories_7d_std': meals_lookback['calories'].std() or 0,
                        'calories_7d_trend': self._calculate_trend(meals_lookback.groupby('date')['calories'].sum().values),
                        'protein_7d_avg': meals_lookback['protein'].mean(),
                        'carbs_7d_avg': meals_lookback['carbs'].mean(),
                        'fat_7d_avg': meals_lookback['fat'].mean(),
                        'meal_frequency_7d': len(meals_lookback) / lookback_days
                    })
                else:
                    feature_row.update({
                        'calories_7d_avg': 0, 'calories_7d_std': 0, 'calories_7d_trend': 0,
                        'protein_7d_avg': 0, 'carbs_7d_avg': 0, 'fat_7d_avg': 0,
                        'meal_frequency_7d': 0
                    })
                
                # Lookback features for exercises
                exercises_lookback = user_exercises[
                    (user_exercises['date'] >= lookback_start) & 
                    (user_exercises['date'] < date)
                ]
                
                if len(exercises_lookback) > 0:
                    feature_row.update({
                        'exercise_duration_7d_avg': exercises_lookback['duration_minutes'].mean(),
                        'exercise_calories_7d_avg': exercises_lookback['calories_burned'].mean(),
                        'exercise_frequency_7d': len(exercises_lookback) / lookback_days,
                        'exercise_consistency_7d': len(exercises_lookback['date'].unique()) / lookback_days
                    })
                else:
                    feature_row.update({
                        'exercise_duration_7d_avg': 0, 'exercise_calories_7d_avg': 0,
                        'exercise_frequency_7d': 0, 'exercise_consistency_7d': 0
                    })
                
                # Lookback features for sleep
                sleep_lookback = user_sleep[
                    (user_sleep['date'] >= lookback_start) & 
                    (user_sleep['date'] < date)
                ]
                
                if len(sleep_lookback) > 0:
                    feature_row.update({
                        'sleep_duration_7d_avg': sleep_lookback['sleep_duration_hours'].mean(),
                        'sleep_quality_7d_avg': sleep_lookback['sleep_quality'].mean(),
                        'sleep_duration_7d_std': sleep_lookback['sleep_duration_hours'].std() or 0,
                        'sleep_quality_trend_7d': self._calculate_trend(sleep_lookback['sleep_quality'].values)
                    })
                else:
                    feature_row.update({
                        'sleep_duration_7d_avg': 0, 'sleep_quality_7d_avg': 0,
                        'sleep_duration_7d_std': 0, 'sleep_quality_trend_7d': 0
                    })
                
                # Lookback features for mood
                mood_lookback = user_mood[
                    (user_mood['date'] >= lookback_start) & 
                    (user_mood['date'] < date)
                ]
                
                if len(mood_lookback) > 0:
                    feature_row.update({
                        'mood_7d_avg': mood_lookback['mood_score'].mean(),
                        'stress_7d_avg': mood_lookback['stress_level'].mean(),
                        'energy_7d_avg': mood_lookback['energy_level'].mean(),
                        'mood_7d_std': mood_lookback['mood_score'].std() or 0,
                        'mood_trend_7d': self._calculate_trend(mood_lookback['mood_score'].values),
                        'stress_trend_7d': self._calculate_trend(mood_lookback['stress_level'].values)
                    })
                else:
                    feature_row.update({
                        'mood_7d_avg': 0, 'stress_7d_avg': 0, 'energy_7d_avg': 0,
                        'mood_7d_std': 0, 'mood_trend_7d': 0, 'stress_trend_7d': 0
                    })
                
                # Get target variables (next day values)
                target_date = date + timedelta(days=1)
                
                # Sleep quality target
                target_sleep = user_sleep[user_sleep['date'] == target_date]
                feature_row['target_sleep_quality'] = target_sleep['sleep_quality'].iloc[0] if len(target_sleep) > 0 else None
                
                # Mood target
                target_mood = user_mood[user_mood['date'] == target_date]
                feature_row['target_mood_score'] = target_mood['mood_score'].iloc[0] if len(target_mood) > 0 else None
                feature_row['target_stress_level'] = target_mood['stress_level'].iloc[0] if len(target_mood) > 0 else None
                
                # Exercise target (binary: will exercise or not)
                target_exercise = user_exercises[user_exercises['date'] == target_date]
                feature_row['target_will_exercise'] = 1 if len(target_exercise) > 0 else 0
                
                time_series_features.append(feature_row)
        
        df = pd.DataFrame(time_series_features)
        # Drop rows with missing targets
        df = df.dropna(subset=['target_sleep_quality', 'target_mood_score'])
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different health metrics"""
        df = df.copy()
        
        # BMI categories
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, float('inf')], 
                                   labels=['underweight', 'normal', 'overweight', 'obese'])
        
        # Sleep-Exercise interaction
        df['sleep_exercise_balance'] = df['sleep_duration_7d_avg'] * df['exercise_frequency_7d']
        
        # Nutrition balance
        df['macro_balance'] = (df['protein_7d_avg'] + df['fat_7d_avg']) / (df['carbs_7d_avg'] + 1e-6)
        
        # Energy balance (calories in vs out)
        df['energy_balance'] = df['calories_7d_avg'] - df['exercise_calories_7d_avg']
        
        # Stress-sleep interaction
        df['stress_sleep_impact'] = df['stress_7d_avg'] / (df['sleep_quality_7d_avg'] + 1e-6)
        
        # Activity consistency
        df['overall_consistency'] = (df['meal_frequency_7d'] * df['exercise_consistency_7d'] * 
                                    (1 - df['sleep_duration_7d_std'] / (df['sleep_duration_7d_avg'] + 1e-6)))
        
        # Weekend effect
        df['weekend_mood_diff'] = df.groupby('user_id')['mood_7d_avg'].transform(
            lambda x: x - x.mean()
        ) * df['is_weekend']
        
        return df
    
    def prepare_features_for_training(self, df: pd.DataFrame, target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features for ML training"""
        df_processed = df.copy()
        
        # Handle categorical features
        categorical_cols = ['gender', 'activity_level', 'preferred_exercise_type', 'bmi_category']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_processed[col] = self.encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    df_processed[col] = self.encoders[col].transform(df_processed[col].astype(str))
        
        # Select feature columns (exclude target columns and metadata)
        feature_cols = [col for col in df_processed.columns 
                       if col not in target_cols + ['user_id', 'date', 'target_date']]
        
        X = df_processed[feature_cols].values
        y = df_processed[target_cols].values
        
        # Scale features
        if 'features' not in self.scalers:
            self.scalers['features'] = StandardScaler()
            X_scaled = self.scalers['features'].fit_transform(X)
        else:
            X_scaled = self.scalers['features'].transform(X)
        
        return X_scaled, y, feature_cols
    
    def _calculate_meal_regularity(self, meals_df: pd.DataFrame) -> float:
        """Calculate how regular meal timing is"""
        if len(meals_df) == 0:
            return 0
        
        # Group by meal type and calculate time consistency
        regularity_scores = []
        
        for meal_type in meals_df['meal_type'].unique():
            meal_times = meals_df[meals_df['meal_type'] == meal_type]['timestamp']
            if len(meal_times) < 2:
                continue
                
            # Extract hour from timestamp
            hours = pd.to_datetime(meal_times).dt.hour
            regularity = 1 - (hours.std() / 24)  # Lower std = higher regularity
            regularity_scores.append(max(0, regularity))
        
        return np.mean(regularity_scores) if regularity_scores else 0
    
    def _calculate_exercise_consistency(self, exercises_df: pd.DataFrame) -> float:
        """Calculate exercise consistency over time"""
        if len(exercises_df) == 0:
            return 0
        
        # Count days with exercise
        exercise_days = exercises_df['date'].nunique()
        total_days = 90  # Assuming 90-day period
        
        return exercise_days / total_days
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend direction (-1 to 1, where 1 is improving)"""
        if len(values) < 2:
            return 0
        
        # Simple linear trend
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            # Normalize slope to [-1, 1] range
            return np.tanh(slope)
        except:
            return 0

def create_ml_features(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Main function to create all ML features"""
    engineer = HealthFeatureEngineer()
    
    print("Creating aggregated user features...")
    user_features = engineer.create_user_aggregated_features(datasets)
    
    print("Creating time-series features...")
    time_series_features = engineer.create_time_series_features(datasets)
    
    print("Adding temporal features...")
    time_series_features = engineer.create_temporal_features(time_series_features)
    
    # IMPORTANT: Merge user features BEFORE creating interaction features
    # because interaction features need the 'bmi' column from user_features
    print("Merging features...")
    final_features = time_series_features.merge(user_features, on='user_id', how='left')
    
    print("Creating interaction features...")
    final_features = engineer.create_interaction_features(final_features)
    
    # Prepare training data
    target_cols = ['target_sleep_quality', 'target_mood_score', 'target_stress_level', 'target_will_exercise']
    X, y, feature_names = engineer.prepare_features_for_training(final_features, target_cols)
    
    return {
        'user_features': user_features,
        'time_series_features': time_series_features,
        'final_features': final_features,
        'X_train': X,
        'y_train': y,
        'feature_names': feature_names,
        'engineer': engineer
    }

if __name__ == "__main__":
    # Example usage
    from .synthetic_generator import save_synthetic_data
    
    # Generate synthetic data
    datasets = save_synthetic_data()
    
    # Create ML features
    ml_features = create_ml_features(datasets)
    
    print(f"Final feature matrix shape: {ml_features['X_train'].shape}")
    print(f"Target matrix shape: {ml_features['y_train'].shape}")
    print(f"Number of features: {len(ml_features['feature_names'])}")
    print(f"Sample features: {ml_features['feature_names'][:10]}")