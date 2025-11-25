# ml/data/feature_engineering_v2.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler

class ImprovedHealthFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """Enhanced temporal features with more granularity"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic temporal features
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['month'] = df[date_col].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        
        # Cyclical encoding for temporal features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lookback_data: pd.DataFrame, 
                           target_cols: List[str]) -> pd.DataFrame:
        """Create lag features for recent history"""
        df = df.copy()
        
        for lag in [1, 2, 3, 7]:
            for col in target_cols:
                lag_col_name = f'{col}_lag_{lag}'
                df[lag_col_name] = 0  # Default value
                
                for idx, row in df.iterrows():
                    user_id = row['user_id']
                    date = pd.to_datetime(row['date'])
                    lag_date = date - timedelta(days=lag)
                    
                    lag_value = lookback_data[
                        (lookback_data['user_id'] == user_id) & 
                        (pd.to_datetime(lookback_data['date']) == lag_date)
                    ]
                    
                    if len(lag_value) > 0 and col in lag_value.columns:
                        df.at[idx, lag_col_name] = lag_value[col].iloc[0]
        
        return df
    
    def create_user_aggregated_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create comprehensive aggregated features per user"""
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
                daily_calories = user_meals.groupby('date')['calories'].sum()
                user_data.update({
                    'avg_daily_calories': daily_calories.mean(),
                    'median_daily_calories': daily_calories.median(),
                    'calories_percentile_25': daily_calories.quantile(0.25),
                    'calories_percentile_75': daily_calories.quantile(0.75),
                    'avg_protein': user_meals['protein'].mean(),
                    'avg_carbs': user_meals['carbs'].mean(),
                    'avg_fat': user_meals['fat'].mean(),
                    'meals_per_day': len(user_meals) / max(len(user_meals['date'].unique()), 1),
                    'calories_std': daily_calories.std() if len(daily_calories) > 1 else 0,
                    'meal_regularity': self._calculate_meal_regularity(user_meals),
                    'protein_ratio': user_meals['protein'].mean() / (user_meals['calories'].mean() + 1e-6),
                    'carbs_ratio': user_meals['carbs'].mean() / (user_meals['calories'].mean() + 1e-6),
                    'fat_ratio': user_meals['fat'].mean() / (user_meals['calories'].mean() + 1e-6),
                })
            else:
                user_data.update({
                    'avg_daily_calories': 0, 'median_daily_calories': 0,
                    'calories_percentile_25': 0, 'calories_percentile_75': 0,
                    'avg_protein': 0, 'avg_carbs': 0, 'avg_fat': 0,
                    'meals_per_day': 0, 'calories_std': 0, 'meal_regularity': 0,
                    'protein_ratio': 0, 'carbs_ratio': 0, 'fat_ratio': 0
                })
            
            # Exercise features
            user_exercises = datasets['exercises'][datasets['exercises']['user_id'] == user_id]
            if len(user_exercises) > 0:
                user_data.update({
                    'avg_exercise_duration': user_exercises['duration_minutes'].mean(),
                    'median_exercise_duration': user_exercises['duration_minutes'].median(),
                    'avg_calories_burned': user_exercises['calories_burned'].mean(),
                    'exercise_frequency': len(user_exercises) / 90,
                    'exercise_consistency': self._calculate_exercise_consistency(user_exercises),
                    'preferred_exercise_type': user_exercises['exercise_type'].mode().iloc[0] if len(user_exercises) > 0 else 'none',
                    'exercise_intensity_var': user_exercises['calories_burned'].std() if len(user_exercises) > 1 else 0,
                    'total_exercise_days': len(user_exercises['date'].unique()),
                })
            else:
                user_data.update({
                    'avg_exercise_duration': 0, 'median_exercise_duration': 0,
                    'avg_calories_burned': 0, 'exercise_frequency': 0,
                    'exercise_consistency': 0, 'preferred_exercise_type': 'none',
                    'exercise_intensity_var': 0, 'total_exercise_days': 0
                })
            
            # Sleep features
            user_sleep = datasets['sleep'][datasets['sleep']['user_id'] == user_id]
            if len(user_sleep) > 0:
                user_data.update({
                    'avg_sleep_duration': user_sleep['sleep_duration_hours'].mean(),
                    'median_sleep_duration': user_sleep['sleep_duration_hours'].median(),
                    'avg_sleep_quality': user_sleep['sleep_quality'].mean(),
                    'median_sleep_quality': user_sleep['sleep_quality'].median(),
                    'sleep_duration_std': user_sleep['sleep_duration_hours'].std() if len(user_sleep) > 1 else 0,
                    'sleep_regularity': 1 - min(user_sleep['sleep_duration_hours'].std() / (user_sleep['sleep_duration_hours'].mean() + 1e-6), 1),
                    'sleep_quality_trend': self._calculate_trend(user_sleep['sleep_quality'].values),
                    'sleep_quality_std': user_sleep['sleep_quality'].std() if len(user_sleep) > 1 else 0,
                    'poor_sleep_days': len(user_sleep[user_sleep['sleep_quality'] < 5]),
                })
            else:
                user_data.update({
                    'avg_sleep_duration': 0, 'median_sleep_duration': 0,
                    'avg_sleep_quality': 0, 'median_sleep_quality': 0,
                    'sleep_duration_std': 0, 'sleep_regularity': 0,
                    'sleep_quality_trend': 0, 'sleep_quality_std': 0,
                    'poor_sleep_days': 0
                })
            
            # Mood features
            user_mood = datasets['mood'][datasets['mood']['user_id'] == user_id]
            if len(user_mood) > 0:
                user_data.update({
                    'avg_mood_score': user_mood['mood_score'].mean(),
                    'median_mood_score': user_mood['mood_score'].median(),
                    'avg_stress_level': user_mood['stress_level'].mean(),
                    'median_stress_level': user_mood['stress_level'].median(),
                    'avg_energy_level': user_mood['energy_level'].mean(),
                    'mood_variability': user_mood['mood_score'].std() if len(user_mood) > 1 else 0,
                    'mood_trend': self._calculate_trend(user_mood['mood_score'].values),
                    'stress_trend': self._calculate_trend(user_mood['stress_level'].values),
                    'energy_variability': user_mood['energy_level'].std() if len(user_mood) > 1 else 0,
                    'high_stress_days': len(user_mood[user_mood['stress_level'] > 7]),
                })
            else:
                user_data.update({
                    'avg_mood_score': 0, 'median_mood_score': 0,
                    'avg_stress_level': 0, 'median_stress_level': 0,
                    'avg_energy_level': 0, 'mood_variability': 0,
                    'mood_trend': 0, 'stress_trend': 0, 'energy_variability': 0,
                    'high_stress_days': 0
                })
            
            user_features.append(user_data)
        
        return pd.DataFrame(user_features)
    
    def create_time_series_features(self, datasets: Dict[str, pd.DataFrame], 
                                   lookback_days: int = 28,
                                   sample_rate: float = 0.5) -> pd.DataFrame:
        """Create enhanced time-series features with better sampling"""
        time_series_features = []
        users = datasets['users']['user_id'].unique()
        
        # Sample users to reduce computation
        if sample_rate < 1.0:
            num_sample_users = int(len(users) * sample_rate)
            users = np.random.choice(users, size=num_sample_users, replace=False)
        
        print(f"Processing {len(users)} users for time series features...")
        
        for idx, user_id in enumerate(users):
            if idx % 50 == 0:
                print(f"Processing user {idx}/{len(users)}")
            
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
            
            # Sample dates - keep more recent data
            if len(all_dates) > 35:
                # Keep all dates in first lookback_days, then every 2nd date
                sampled_dates = [d for i, d in enumerate(all_dates) 
                               if i < lookback_days or i % 2 == 0]
            else:
                sampled_dates = all_dates
            
            # Create features for each date
            for date in sampled_dates:
                if date < min(all_dates) + timedelta(days=lookback_days):
                    continue
                
                lookback_start = date - timedelta(days=lookback_days)
                
                feature_row = {
                    'user_id': user_id,
                    'date': date,
                    'target_date': date + timedelta(days=1)
                }
                
                # Enhanced meal features with multiple windows
                meals_lookback = user_meals[
                    (user_meals['date'] >= lookback_start) & 
                    (user_meals['date'] < date)
                ]
                
                if len(meals_lookback) > 0:
                    daily_calories = meals_lookback.groupby('date')['calories'].sum()
                    feature_row.update({
                        'calories_28d_avg': meals_lookback['calories'].mean(),
                        'calories_28d_std': meals_lookback['calories'].std() if len(meals_lookback) > 1 else 0,
                        'calories_28d_min': daily_calories.min() if len(daily_calories) > 0 else 0,
                        'calories_28d_max': daily_calories.max() if len(daily_calories) > 0 else 0,
                        'calories_14d_avg': meals_lookback[meals_lookback['date'] >= date - timedelta(days=14)]['calories'].mean(),
                        'calories_7d_avg': meals_lookback[meals_lookback['date'] >= date - timedelta(days=7)]['calories'].mean(),
                        'calories_3d_avg': meals_lookback[meals_lookback['date'] >= date - timedelta(days=3)]['calories'].mean(),
                        'calories_trend': self._calculate_trend(daily_calories.values) if len(daily_calories) > 1 else 0,
                        'protein_28d_avg': meals_lookback['protein'].mean(),
                        'carbs_28d_avg': meals_lookback['carbs'].mean(),
                        'fat_28d_avg': meals_lookback['fat'].mean(),
                        'meal_frequency_28d': len(meals_lookback) / lookback_days,
                        'macro_balance_28d': self._calculate_macro_balance(meals_lookback),
                    })
                else:
                    feature_row.update({
                        'calories_28d_avg': 0, 'calories_28d_std': 0,
                        'calories_28d_min': 0, 'calories_28d_max': 0,
                        'calories_14d_avg': 0, 'calories_7d_avg': 0, 'calories_3d_avg': 0,
                        'calories_trend': 0, 'protein_28d_avg': 0,
                        'carbs_28d_avg': 0, 'fat_28d_avg': 0,
                        'meal_frequency_28d': 0, 'macro_balance_28d': 0
                    })
                
                # Enhanced exercise features
                exercises_lookback = user_exercises[
                    (user_exercises['date'] >= lookback_start) & 
                    (user_exercises['date'] < date)
                ]
                
                if len(exercises_lookback) > 0:
                    feature_row.update({
                        'exercise_duration_28d_avg': exercises_lookback['duration_minutes'].mean(),
                        'exercise_duration_14d_avg': exercises_lookback[exercises_lookback['date'] >= date - timedelta(days=14)]['duration_minutes'].mean(),
                        'exercise_duration_7d_avg': exercises_lookback[exercises_lookback['date'] >= date - timedelta(days=7)]['duration_minutes'].mean(),
                        'exercise_calories_28d_avg': exercises_lookback['calories_burned'].mean(),
                        'exercise_frequency_28d': len(exercises_lookback) / lookback_days,
                        'exercise_frequency_7d': len(exercises_lookback[exercises_lookback['date'] >= date - timedelta(days=7)]) / 7,
                        'exercise_consistency_28d': len(exercises_lookback['date'].unique()) / lookback_days,
                        'days_since_exercise': (date - exercises_lookback['date'].max()).days if len(exercises_lookback) > 0 else lookback_days,
                        'exercise_intensity_avg': exercises_lookback['calories_burned'].mean() / (exercises_lookback['duration_minutes'].mean() + 1e-6),
                    })
                else:
                    feature_row.update({
                        'exercise_duration_28d_avg': 0, 'exercise_duration_14d_avg': 0,
                        'exercise_duration_7d_avg': 0, 'exercise_calories_28d_avg': 0,
                        'exercise_frequency_28d': 0, 'exercise_frequency_7d': 0,
                        'exercise_consistency_28d': 0, 'days_since_exercise': lookback_days,
                        'exercise_intensity_avg': 0
                    })
                
                # Enhanced sleep features with multiple windows
                sleep_lookback = user_sleep[
                    (user_sleep['date'] >= lookback_start) & 
                    (user_sleep['date'] < date)
                ]
                
                if len(sleep_lookback) > 0:
                    feature_row.update({
                        'sleep_duration_28d_avg': sleep_lookback['sleep_duration_hours'].mean(),
                        'sleep_duration_14d_avg': sleep_lookback[sleep_lookback['date'] >= date - timedelta(days=14)]['sleep_duration_hours'].mean(),
                        'sleep_duration_7d_avg': sleep_lookback[sleep_lookback['date'] >= date - timedelta(days=7)]['sleep_duration_hours'].mean(),
                        'sleep_duration_3d_avg': sleep_lookback[sleep_lookback['date'] >= date - timedelta(days=3)]['sleep_duration_hours'].mean(),
                        'sleep_quality_28d_avg': sleep_lookback['sleep_quality'].mean(),
                        'sleep_quality_14d_avg': sleep_lookback[sleep_lookback['date'] >= date - timedelta(days=14)]['sleep_quality'].mean(),
                        'sleep_quality_7d_avg': sleep_lookback[sleep_lookback['date'] >= date - timedelta(days=7)]['sleep_quality'].mean(),
                        'sleep_quality_3d_avg': sleep_lookback[sleep_lookback['date'] >= date - timedelta(days=3)]['sleep_quality'].mean(),
                        'sleep_duration_28d_std': sleep_lookback['sleep_duration_hours'].std() if len(sleep_lookback) > 1 else 0,
                        'sleep_quality_trend_28d': self._calculate_trend(sleep_lookback['sleep_quality'].values) if len(sleep_lookback) > 1 else 0,
                        'sleep_debt': max(0, 8 - sleep_lookback['sleep_duration_hours'].mean()),
                        'poor_sleep_streak': self._calculate_streak(sleep_lookback['sleep_quality'].values, threshold=5, below=True),
                    })
                else:
                    feature_row.update({
                        'sleep_duration_28d_avg': 0, 'sleep_duration_14d_avg': 0,
                        'sleep_duration_7d_avg': 0, 'sleep_duration_3d_avg': 0,
                        'sleep_quality_28d_avg': 0, 'sleep_quality_14d_avg': 0,
                        'sleep_quality_7d_avg': 0, 'sleep_quality_3d_avg': 0,
                        'sleep_duration_28d_std': 0, 'sleep_quality_trend_28d': 0,
                        'sleep_debt': 0, 'poor_sleep_streak': 0
                    })
                
                # Enhanced mood features
                mood_lookback = user_mood[
                    (user_mood['date'] >= lookback_start) & 
                    (user_mood['date'] < date)
                ]
                
                if len(mood_lookback) > 0:
                    feature_row.update({
                        'mood_28d_avg': mood_lookback['mood_score'].mean(),
                        'mood_14d_avg': mood_lookback[mood_lookback['date'] >= date - timedelta(days=14)]['mood_score'].mean(),
                        'mood_7d_avg': mood_lookback[mood_lookback['date'] >= date - timedelta(days=7)]['mood_score'].mean(),
                        'mood_3d_avg': mood_lookback[mood_lookback['date'] >= date - timedelta(days=3)]['mood_score'].mean(),
                        'stress_28d_avg': mood_lookback['stress_level'].mean(),
                        'stress_14d_avg': mood_lookback[mood_lookback['date'] >= date - timedelta(days=14)]['stress_level'].mean(),
                        'stress_7d_avg': mood_lookback[mood_lookback['date'] >= date - timedelta(days=7)]['stress_level'].mean(),
                        'stress_3d_avg': mood_lookback[mood_lookback['date'] >= date - timedelta(days=3)]['stress_level'].mean(),
                        'energy_28d_avg': mood_lookback['energy_level'].mean(),
                        'energy_7d_avg': mood_lookback[mood_lookback['date'] >= date - timedelta(days=7)]['energy_level'].mean(),
                        'mood_28d_std': mood_lookback['mood_score'].std() if len(mood_lookback) > 1 else 0,
                        'mood_trend_28d': self._calculate_trend(mood_lookback['mood_score'].values) if len(mood_lookback) > 1 else 0,
                        'stress_trend_28d': self._calculate_trend(mood_lookback['stress_level'].values) if len(mood_lookback) > 1 else 0,
                        'mood_energy_correlation': self._calculate_correlation(
                            mood_lookback['mood_score'].values,
                            mood_lookback['energy_level'].values
                        ),
                    })
                else:
                    feature_row.update({
                        'mood_28d_avg': 0, 'mood_14d_avg': 0, 'mood_7d_avg': 0, 'mood_3d_avg': 0,
                        'stress_28d_avg': 0, 'stress_14d_avg': 0, 'stress_7d_avg': 0, 'stress_3d_avg': 0,
                        'energy_28d_avg': 0, 'energy_7d_avg': 0,
                        'mood_28d_std': 0, 'mood_trend_28d': 0, 'stress_trend_28d': 0,
                        'mood_energy_correlation': 0
                    })
                
                # Get target variables
                target_date = date + timedelta(days=1)
                
                target_sleep = user_sleep[user_sleep['date'] == target_date]
                feature_row['target_sleep_quality'] = target_sleep['sleep_quality'].iloc[0] if len(target_sleep) > 0 else None
                
                target_mood = user_mood[user_mood['date'] == target_date]
                feature_row['target_mood_score'] = target_mood['mood_score'].iloc[0] if len(target_mood) > 0 else None
                feature_row['target_stress_level'] = target_mood['stress_level'].iloc[0] if len(target_mood) > 0 else None
                
                target_exercise = user_exercises[user_exercises['date'] == target_date]
                feature_row['target_will_exercise'] = 1 if len(target_exercise) > 0 else 0
                
                time_series_features.append(feature_row)
        
        df = pd.DataFrame(time_series_features)
        df = df.dropna(subset=['target_sleep_quality', 'target_mood_score'])
        
        print(f"Created {len(df)} time series samples")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced interaction features based on domain knowledge"""
        df = df.copy()
        
        # BMI categories
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, float('inf')], 
                                   labels=['underweight', 'normal', 'overweight', 'obese'])
        
        # Sleep-Exercise interactions
        df['sleep_exercise_balance'] = df['sleep_duration_28d_avg'] * df['exercise_frequency_28d']
        df['sleep_recovery_score'] = df['sleep_quality_28d_avg'] * (df['sleep_duration_28d_avg'] / 8)
        
        # Nutrition balance
        df['macro_balance'] = (df['protein_28d_avg'] + df['fat_28d_avg']) / (df['carbs_28d_avg'] + 1e-6)
        df['calorie_consistency'] = 1 / (1 + df['calories_28d_std'] / (df['calories_28d_avg'] + 1e-6))
        
        # Energy balance
        df['energy_balance'] = df['calories_28d_avg'] - df['exercise_calories_28d_avg']
        df['energy_surplus'] = (df['energy_balance'] > 0).astype(int)
        
        # Stress-sleep interactions (KEY for improvement)
        df['stress_sleep_impact'] = df['stress_28d_avg'] * (1 / (df['sleep_quality_28d_avg'] + 1))
        df['recovery_capacity'] = df['sleep_quality_28d_avg'] * df['energy_28d_avg'] / (df['stress_28d_avg'] + 1)
        
        # Activity and mood interactions
        df['active_happy_score'] = df['exercise_frequency_28d'] * df['mood_28d_avg']
        df['stress_exercise_relief'] = df['exercise_frequency_28d'] / (df['stress_28d_avg'] + 1)
        
        # Overall health score
        df['overall_health_score'] = (
            df['sleep_quality_28d_avg'] * 0.3 +
            df['mood_28d_avg'] * 0.3 +
            (10 - df['stress_28d_avg']) * 0.2 +
            (df['exercise_frequency_28d'] * 10) * 0.2
        )
        
        # Trend interactions
        df['positive_trend_score'] = (
            (df['mood_trend_28d'] > 0).astype(int) +
            (df['sleep_quality_trend_28d'] > 0).astype(int) +
            (df['stress_trend_28d'] < 0).astype(int)
        )
        
        # Recent vs long-term patterns
        df['sleep_recent_vs_longterm'] = df['sleep_quality_7d_avg'] / (df['sleep_quality_28d_avg'] + 1e-6)
        df['mood_recent_vs_longterm'] = df['mood_7d_avg'] / (df['mood_28d_avg'] + 1e-6)
        df['stress_recent_vs_longterm'] = df['stress_7d_avg'] / (df['stress_28d_avg'] + 1e-6)
        df['calories_recent_vs_longterm'] = df['calories_7d_avg'] / (df['calories_28d_avg'] + 1e-6)
        
        # Day-specific interactions
        df['monday_stress'] = df['is_monday'] * df['stress_28d_avg']
        df['weekend_sleep'] = df['is_weekend'] * df['sleep_duration_28d_avg']
        df['friday_mood'] = df['is_friday'] * df['mood_28d_avg']
        
        return df
    
    def prepare_features_for_training(self, df: pd.DataFrame, target_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features with improved scaling"""
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
        
        # Select feature columns
        feature_cols = [col for col in df_processed.columns 
                       if col not in target_cols + ['user_id', 'date', 'target_date']]
        
        X = df_processed[feature_cols].values
        y = df_processed[target_cols].values
        
        # Replace any remaining NaNs
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        # Use RobustScaler for better handling of outliers
        if 'features' not in self.scalers:
            self.scalers['features'] = RobustScaler()
            X_scaled = self.scalers['features'].fit_transform(X)
        else:
            X_scaled = self.scalers['features'].transform(X)
        
        return X_scaled, y, feature_cols
    
    def _calculate_meal_regularity(self, meals_df: pd.DataFrame) -> float:
        """Calculate meal timing regularity"""
        if len(meals_df) == 0:
            return 0
        
        regularity_scores = []
        for meal_type in meals_df['meal_type'].unique():
            meal_times = meals_df[meals_df['meal_type'] == meal_type]['timestamp']
            if len(meal_times) < 2:
                continue
            
            hours = pd.to_datetime(meal_times).dt.hour
            regularity = 1 - min(hours.std() / 24, 1)
            regularity_scores.append(max(0, regularity))
        
        return np.mean(regularity_scores) if regularity_scores else 0
    
    def _calculate_exercise_consistency(self, exercises_df: pd.DataFrame) -> float:
        """Calculate exercise consistency"""
        if len(exercises_df) == 0:
            return 0
        
        exercise_days = exercises_df['date'].nunique()
        total_days = 90
        return exercise_days / total_days
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend with better handling"""
        if len(values) < 2:
            return 0
        
        try:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return np.tanh(slope)
        except:
            return 0
    
    def _calculate_streak(self, values: np.ndarray, threshold: float = 5, below: bool = True) -> int:
        """Calculate current streak of values above/below threshold"""
        if len(values) == 0:
            return 0
        
        streak = 0
        for val in reversed(values):
            if below and val < threshold:
                streak += 1
            elif not below and val >= threshold:
                streak += 1
            else:
                break
        
        return streak
    
    def _calculate_macro_balance(self, meals_df: pd.DataFrame) -> float:
        """Calculate macronutrient balance score"""
        if len(meals_df) == 0:
            return 0
        
        protein = meals_df['protein'].mean()
        carbs = meals_df['carbs'].mean()
        fat = meals_df['fat'].mean()
        
        total = protein + carbs + fat
        if total == 0:
            return 0
        
        # Ideal ratios: protein 30%, carbs 40%, fat 30%
        protein_ratio = (protein / total) * 100
        carbs_ratio = (carbs / total) * 100
        fat_ratio = (fat / total) * 100
        
        score = 1 - (
            abs(protein_ratio - 30) / 30 +
            abs(carbs_ratio - 40) / 40 +
            abs(fat_ratio - 30) / 30
        ) / 3
        
        return max(0, score)
    
    def _calculate_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate correlation between two arrays"""
        if len(x) < 2 or len(y) < 2:
            return 0
        
        try:
            return np.corrcoef(x, y)[0, 1]
        except:
            return 0


def create_ml_features(datasets: Dict[str, pd.DataFrame], 
                      sample_rate: float = 0.5,
                      lookback_days: int = 28) -> Dict:
    """Main function to create all ML features with improvements"""
    engineer = ImprovedHealthFeatureEngineer()
    
    print("Creating aggregated user features...")
    user_features = engineer.create_user_aggregated_features(datasets)
    
    print(f"Creating time-series features (sampling {sample_rate*100}% of data)...")
    print(f"Using {lookback_days}-day lookback window...")
    time_series_features = engineer.create_time_series_features(
        datasets, 
        lookback_days=lookback_days,
        sample_rate=sample_rate
    )
    
    print("Adding temporal features...")
    time_series_features = engineer.create_temporal_features(time_series_features)
    
    print("Merging with user features...")
    final_features = time_series_features.merge(user_features, on='user_id', how='left')
    
    print("Creating interaction features...")
    final_features = engineer.create_interaction_features(final_features)
    
    # Prepare training data
    target_cols = ['target_sleep_quality', 'target_mood_score', 'target_stress_level', 'target_will_exercise']
    X, y, feature_names = engineer.prepare_features_for_training(final_features, target_cols)
    
    print(f"\nFinal dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    return {
        'user_features': user_features,
        'time_series_features': time_series_features,
        'final_features': final_features,
        'X_train': X,
        'y_train': y,
        'feature_names': feature_names,
        'engineer': engineer
    }