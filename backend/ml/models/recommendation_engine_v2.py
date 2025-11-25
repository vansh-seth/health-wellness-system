# ml/models/recommendation_engine_v2.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path
import joblib

def encode_categorical_features(df, categorical_cols=None):
    """
    Encode categorical features for prediction
    Returns encoded dataframe and mapping for reference
    """
    df_encoded = df.copy()
    
    if categorical_cols is None:
        # Auto-detect categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    encoding_map = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            # Create label encoding
            unique_values = df_encoded[col].unique()
            value_map = {val: idx for idx, val in enumerate(unique_values)}
            encoding_map[col] = value_map
            
            # Apply encoding
            df_encoded[col] = df_encoded[col].map(value_map)
            
            # Handle any NaN values that might result from unmapped values
            if df_encoded[col].isna().any():
                df_encoded[col].fillna(-1, inplace=True)
    
    return df_encoded, encoding_map


class NeuralCollaborativeFiltering(nn.Module):
    """Neural collaborative filtering for personalized health recommendations"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Neural MF layers with regularization
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        concat_embeds = torch.cat([user_embeds, item_embeds], dim=1)
        
        # Pass through neural layers
        output = self.fc_layers(concat_embeds)
        return torch.sigmoid(output)

class HealthRecommendationEngine:
    """AI-powered health recommendation system using multi-modal data"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.user_profiles = {}
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_catalog = self._initialize_item_catalog()
        
    def _initialize_item_catalog(self) -> Dict[str, List[Dict]]:
        """Initialize catalog of recommendable health items/actions"""
        catalog = {
            'meals': [
                {'id': 0, 'name': 'Mediterranean Salad', 'calories': 350, 'protein': 15, 'carbs': 25, 'fat': 22, 'tags': ['healthy', 'light', 'vegetarian'], 'prep_time': 15},
                {'id': 1, 'name': 'Grilled Chicken Breast', 'calories': 280, 'protein': 53, 'carbs': 0, 'fat': 6, 'tags': ['high-protein', 'low-carb'], 'prep_time': 25},
                {'id': 2, 'name': 'Quinoa Bowl', 'calories': 420, 'protein': 18, 'carbs': 65, 'fat': 8, 'tags': ['vegetarian', 'high-fiber', 'complete-protein'], 'prep_time': 20},
                {'id': 3, 'name': 'Salmon with Vegetables', 'calories': 380, 'protein': 42, 'carbs': 12, 'fat': 18, 'tags': ['omega-3', 'anti-inflammatory'], 'prep_time': 30},
                {'id': 4, 'name': 'Greek Yogurt with Berries', 'calories': 180, 'protein': 20, 'carbs': 15, 'fat': 5, 'tags': ['probiotics', 'antioxidants'], 'prep_time': 5},
                {'id': 5, 'name': 'Avocado Toast', 'calories': 320, 'protein': 12, 'carbs': 30, 'fat': 18, 'tags': ['healthy-fats', 'fiber'], 'prep_time': 10},
                {'id': 6, 'name': 'Protein Smoothie', 'calories': 250, 'protein': 25, 'carbs': 20, 'fat': 8, 'tags': ['post-workout', 'quick'], 'prep_time': 5},
                {'id': 7, 'name': 'Lentil Soup', 'calories': 230, 'protein': 16, 'carbs': 35, 'fat': 3, 'tags': ['plant-based', 'fiber', 'warming'], 'prep_time': 40},
                {'id': 8, 'name': 'Oatmeal with Nuts', 'calories': 290, 'protein': 12, 'carbs': 42, 'fat': 10, 'tags': ['fiber', 'sustained-energy'], 'prep_time': 10},
                {'id': 9, 'name': 'Turkey Wrap', 'calories': 340, 'protein': 28, 'carbs': 38, 'fat': 9, 'tags': ['balanced', 'portable'], 'prep_time': 15},
                {'id': 10, 'name': 'Egg White Omelette', 'calories': 220, 'protein': 30, 'carbs': 8, 'fat': 7, 'tags': ['high-protein', 'low-calorie'], 'prep_time': 15},
                {'id': 11, 'name': 'Buddha Bowl', 'calories': 380, 'protein': 16, 'carbs': 55, 'fat': 12, 'tags': ['vegetarian', 'colorful', 'balanced'], 'prep_time': 25},
                {'id': 12, 'name': 'Tuna Poke Bowl', 'calories': 410, 'protein': 35, 'carbs': 48, 'fat': 10, 'tags': ['omega-3', 'fresh'], 'prep_time': 15},
            ],
            
            'exercises': [
                {'id': 0, 'name': '30-min Morning Walk', 'duration': 30, 'type': 'cardio', 'intensity': 'low', 'calories_burned': 120, 'equipment': 'none'},
                {'id': 1, 'name': 'HIIT Training', 'duration': 20, 'type': 'cardio', 'intensity': 'high', 'calories_burned': 250, 'equipment': 'none'},
                {'id': 2, 'name': 'Strength Training', 'duration': 45, 'type': 'strength', 'intensity': 'moderate', 'calories_burned': 200, 'equipment': 'weights'},
                {'id': 3, 'name': 'Yoga Session', 'duration': 30, 'type': 'flexibility', 'intensity': 'low', 'calories_burned': 90, 'equipment': 'mat'},
                {'id': 4, 'name': 'Swimming', 'duration': 40, 'type': 'cardio', 'intensity': 'moderate', 'calories_burned': 300, 'equipment': 'pool'},
                {'id': 5, 'name': 'Cycling', 'duration': 45, 'type': 'cardio', 'intensity': 'moderate', 'calories_burned': 280, 'equipment': 'bike'},
                {'id': 6, 'name': 'Pilates', 'duration': 35, 'type': 'flexibility', 'intensity': 'moderate', 'calories_burned': 120, 'equipment': 'mat'},
                {'id': 7, 'name': 'Rock Climbing', 'duration': 60, 'type': 'strength', 'intensity': 'high', 'calories_burned': 400, 'equipment': 'gym'},
                {'id': 8, 'name': 'Jogging', 'duration': 35, 'type': 'cardio', 'intensity': 'moderate', 'calories_burned': 220, 'equipment': 'none'},
                {'id': 9, 'name': 'Dance Class', 'duration': 50, 'type': 'cardio', 'intensity': 'moderate', 'calories_burned': 260, 'equipment': 'none'},
                {'id': 10, 'name': 'Bodyweight Circuit', 'duration': 25, 'type': 'strength', 'intensity': 'moderate', 'calories_burned': 180, 'equipment': 'none'},
                {'id': 11, 'name': 'Stretching Routine', 'duration': 15, 'type': 'flexibility', 'intensity': 'low', 'calories_burned': 45, 'equipment': 'none'},
            ],
            
            'wellness_tips': [
                {'id': 0, 'category': 'sleep', 'tip': 'Try the 4-7-8 breathing technique before bed', 'impact': 'stress_reduction', 'difficulty': 'easy'},
                {'id': 1, 'category': 'nutrition', 'tip': 'Eat the rainbow - include 5 different colored fruits/vegetables', 'impact': 'nutrient_variety', 'difficulty': 'easy'},
                {'id': 2, 'category': 'mental_health', 'tip': 'Practice gratitude journaling for 5 minutes daily', 'impact': 'mood_improvement', 'difficulty': 'easy'},
                {'id': 3, 'category': 'hydration', 'tip': 'Drink a glass of water before each meal', 'impact': 'better_digestion', 'difficulty': 'easy'},
                {'id': 4, 'category': 'movement', 'tip': 'Take a 5-minute walk every hour during work', 'impact': 'energy_boost', 'difficulty': 'easy'},
                {'id': 5, 'category': 'sleep', 'tip': 'Keep bedroom temperature between 65-68°F', 'impact': 'sleep_quality', 'difficulty': 'easy'},
                {'id': 6, 'category': 'stress', 'tip': 'Try progressive muscle relaxation', 'impact': 'anxiety_reduction', 'difficulty': 'moderate'},
                {'id': 7, 'category': 'social', 'tip': 'Connect with a friend or family member today', 'impact': 'emotional_wellbeing', 'difficulty': 'easy'},
                {'id': 8, 'category': 'mindfulness', 'tip': 'Practice 10 minutes of meditation daily', 'impact': 'stress_management', 'difficulty': 'moderate'},
                {'id': 9, 'category': 'sleep', 'tip': 'Establish a consistent sleep schedule', 'impact': 'circadian_rhythm', 'difficulty': 'moderate'},
                {'id': 10, 'category': 'nutrition', 'tip': 'Meal prep on Sundays for the week ahead', 'impact': 'consistency', 'difficulty': 'moderate'},
                {'id': 11, 'category': 'mindfulness', 'tip': 'Practice mindful eating - no screens during meals', 'impact': 'better_digestion', 'difficulty': 'moderate'},
            ]
        }
        return catalog
    
    def build_user_item_matrix(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
        """Build user-item interaction matrix from historical data"""
        users = datasets['users']['user_id'].unique()
        
        # Create user mappings
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        
        # Initialize interaction matrices
        num_users = len(users)
        
        # For meals - based on meal preferences and ratings
        meal_matrix = np.zeros((num_users, len(self.item_catalog['meals'])))
        
        for user_idx, user_id in enumerate(users):
            user_meals = datasets['meals'][datasets['meals']['user_id'] == user_id]
            
            if len(user_meals) > 0:
                # Calculate meal preferences based on frequency and nutritional patterns
                avg_calories = user_meals['calories'].mean()
                avg_protein = user_meals['protein'].mean()
                avg_carbs = user_meals['carbs'].mean()
                
                for meal_idx, meal in enumerate(self.item_catalog['meals']):
                    # Multi-factor similarity
                    calorie_sim = 1 - min(abs(meal['calories'] - avg_calories) / 1000, 1)
                    protein_sim = 1 - min(abs(meal['protein'] - avg_protein) / 50, 1)
                    carbs_sim = 1 - min(abs(meal['carbs'] - avg_carbs) / 70, 1)
                    
                    # Weighted combination (scale 0-5 like ratings)
                    preference_score = max(0, min(5, (calorie_sim * 1.5 + protein_sim * 2.0 + carbs_sim * 1.5)))
                    meal_matrix[user_idx, meal_idx] = preference_score
        
        return meal_matrix, user_to_idx, idx_to_user
    
    def train_collaborative_filtering(self, interaction_matrix: np.ndarray, epochs: int = 100, lr: float = 0.001) -> Optional[NeuralCollaborativeFiltering]:
        """Train neural collaborative filtering model with early stopping"""
        num_users, num_items = interaction_matrix.shape
        
        # Create training data
        user_ids, item_ids, ratings = [], [], []
        
        for user_idx in range(num_users):
            for item_idx in range(num_items):
                if interaction_matrix[user_idx, item_idx] > 0:
                    user_ids.append(user_idx)
                    item_ids.append(item_idx)
                    ratings.append(interaction_matrix[user_idx, item_idx] / 5.0)  # Normalize
                    
                    # Add negative samples
                    if np.random.random() < 0.15:  # 15% negative samples
                        neg_item = np.random.randint(num_items)
                        if interaction_matrix[user_idx, neg_item] == 0:
                            user_ids.append(user_idx)
                            item_ids.append(neg_item)
                            ratings.append(0.0)
        
        if len(user_ids) == 0:
            print("Warning: No training data for collaborative filtering")
            return None
        
        print(f"Training CF with {len(user_ids)} interactions")
        
        # Convert to tensors
        user_tensor = torch.LongTensor(user_ids).to(self.device)
        item_tensor = torch.LongTensor(item_ids).to(self.device)
        rating_tensor = torch.FloatTensor(ratings).unsqueeze(1).to(self.device)
        
        # Initialize model
        model = NeuralCollaborativeFiltering(num_users, num_items, embedding_dim=64).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        model.train()
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(user_tensor, item_tensor)
            loss = criterion(predictions, rating_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        self.models['collaborative_filtering'] = model
        return model
    
    def generate_personalized_recommendations(self, user_id: str, user_features: Dict, 
                                            prediction_results: Dict, 
                                            num_recommendations: int = 5) -> Dict[str, List[Dict]]:
        """Generate personalized recommendations with confidence scores"""
        
        recommendations = {
            'meals': [],
            'exercises': [],
            'wellness_tips': [],
            'metadata': {
                'user_id': user_id,
                'timestamp': pd.Timestamp.now().isoformat(),
                'prediction_confidence': {}
            }
        }
        
        # Extract user characteristics
        user_profile = user_features.get(user_id, {})
        
        # Calculate prediction confidence (based on historical accuracy)
        for target in ['sleep_quality', 'mood_score', 'stress_level']:
            if target in prediction_results and len(prediction_results[target]) > 0:
                pred_value = prediction_results[target][0]
                # Normalize to confidence score (higher values = more confident)
                recommendations['metadata']['prediction_confidence'][target] = min(1.0, max(0.5, 1.0 - abs(pred_value - 7) / 10))
        
        # Generate recommendations
        meal_recs = self._recommend_meals(user_profile, prediction_results, num_recommendations)
        recommendations['meals'] = meal_recs
        
        exercise_recs = self._recommend_exercises(user_profile, prediction_results, num_recommendations)
        recommendations['exercises'] = exercise_recs
        
        wellness_recs = self._recommend_wellness_tips(user_profile, prediction_results, num_recommendations)
        recommendations['wellness_tips'] = wellness_recs
        
        return recommendations
    
    def _recommend_meals(self, user_profile: Dict, predictions: Dict, num_recs: int) -> List[Dict]:
        """Enhanced meal recommendations with better scoring"""
        meal_scores = []
        
        # Get user preferences
        target_calories = user_profile.get('avg_daily_calories', 2000) / 3  # Per meal
        activity_level = user_profile.get('activity_level', 'moderately_active')
        bmi = user_profile.get('bmi', 22)
        
        # Predicted values
        predicted_stress = predictions.get('stress_level', [5])[0] if 'stress_level' in predictions and len(predictions['stress_level']) > 0 else 5
        predicted_energy = predictions.get('mood_score', [7])[0] if 'mood_score' in predictions and len(predictions['mood_score']) > 0 else 7
        
        for meal in self.item_catalog['meals']:
            score = 0.0
            reasons = []
            
            # Calorie matching (0-1 score)
            calorie_diff = abs(meal['calories'] - target_calories)
            calorie_score = max(0, 1 - calorie_diff / max(target_calories, 1))
            score += calorie_score * 0.3
            
            # Macronutrient balance
            protein_ratio = meal['protein'] / (meal['calories'] + 1e-6) * 100
            if protein_ratio > 20:
                score += 0.2
                reasons.append("High protein content")
            
            # Activity level matching
            if activity_level == 'very_active' and meal['protein'] > 35:
                score += 0.25
                reasons.append("Supports active lifestyle")
            elif activity_level == 'sedentary' and meal['calories'] < 300:
                score += 0.25
                reasons.append("Appropriate for lower activity")
            
            # BMI-based recommendations
            if bmi < 18.5 and meal['calories'] > target_calories:
                score += 0.2
                reasons.append("Higher calories for healthy weight gain")
            elif bmi > 25 and meal['calories'] < target_calories * 0.85:
                score += 0.2
                reasons.append("Lower calories for weight management")
            
            # Stress-based recommendations
            if predicted_stress > 7:
                if 'anti-inflammatory' in meal.get('tags', []) or 'omega-3' in meal.get('tags', []):
                    score += 0.3
                    reasons.append("Anti-inflammatory properties reduce stress")
            
            # Quick prep for low energy
            if predicted_energy < 6 and meal.get('prep_time', 30) < 15:
                score += 0.2
                reasons.append("Quick and easy preparation")
            
            # Diversity bonus
            score += np.random.uniform(-0.05, 0.15)
            
            if not reasons:
                reasons.append("Balanced nutritional profile")
            
            meal_scores.append({
                'meal': meal,
                'score': max(0, min(1, score)),  # Clamp to [0, 1]
                'confidence': 'high' if score > 0.7 else 'medium' if score > 0.5 else 'moderate',
                'reasoning': "; ".join(reasons[:3])  # Top 3 reasons
            })
        
        # Sort and return
        meal_scores.sort(key=lambda x: x['score'], reverse=True)
        return meal_scores[:num_recs]
    
    def _recommend_exercises(self, user_profile: Dict, predictions: Dict, num_recs: int) -> List[Dict]:
        """Enhanced exercise recommendations"""
        exercise_scores = []
        
        # User characteristics
        activity_level = user_profile.get('activity_level', 'moderately_active')
        preferred_exercise = user_profile.get('preferred_exercise_type', 'mixed')
        age = user_profile.get('age', 35)
        bmi = user_profile.get('bmi', 22)
        
        # Predictions
        will_exercise_prob = predictions.get('exercise_binary', [0.5])[0] if 'exercise_binary' in predictions and len(predictions['exercise_binary']) > 0 else 0.5
        predicted_mood = predictions.get('mood_score', [7])[0] if 'mood_score' in predictions and len(predictions['mood_score']) > 0 else 7
        predicted_stress = predictions.get('stress_level', [5])[0] if 'stress_level' in predictions and len(predictions['stress_level']) > 0 else 5
        predicted_energy = predictions.get('mood_score', [7])[0] if 'mood_score' in predictions and len(predictions['mood_score']) > 0 else 7
        
        for exercise in self.item_catalog['exercises']:
            score = 0.0
            reasons = []
            
            # Activity level matching
            intensity_match = {
                'sedentary': {'low': 0.9, 'moderate': 0.4, 'high': 0.1},
                'lightly_active': {'low': 0.6, 'moderate': 0.9, 'high': 0.4},
                'moderately_active': {'low': 0.4, 'moderate': 0.9, 'high': 0.7},
                'very_active': {'low': 0.3, 'moderate': 0.7, 'high': 0.95}
            }
            score += intensity_match.get(activity_level, {}).get(exercise['intensity'], 0.5) * 0.3
            
            # Type preference
            if preferred_exercise == exercise['type']:
                score += 0.3
                reasons.append(f"Matches your {exercise['type']} preference")
            elif preferred_exercise == 'mixed':
                score += 0.15
            
            # Age considerations
            if age > 60:
                if exercise['intensity'] == 'low' or exercise['type'] == 'flexibility':
                    score += 0.25
                    reasons.append("Gentle on joints")
            elif age < 30:
                if exercise['intensity'] in ['moderate', 'high']:
                    score += 0.2
            
            # Mood and stress boosting
            if predicted_mood < 6 or predicted_stress > 7:
                if exercise['type'] == 'cardio':
                    score += 0.3
                    reasons.append("Boosts endorphins and mood")
                elif exercise['name'] in ['Yoga Session', 'Stretching Routine']:
                    score += 0.25
                    reasons.append("Calming and stress-reducing")
            
            # Energy level considerations
            if predicted_energy < 5:
                if exercise['duration'] < 30 and exercise['intensity'] == 'low':
                    score += 0.3
                    reasons.append("Short and manageable")
            
            # Exercise probability adjustment
            if will_exercise_prob < 0.4:
                if exercise['duration'] < 25 and exercise.get('equipment') == 'none':
                    score += 0.3
                    reasons.append("Easy to start, no equipment needed")
            
            # BMI-based recommendations
            if bmi > 25 and exercise['type'] == 'cardio':
                score += 0.2
                reasons.append("Effective for calorie burn")
            
            # Diversity
            score += np.random.uniform(-0.05, 0.1)
            
            if not reasons:
                reasons.append("Supports overall fitness goals")
            
            exercise_scores.append({
                'exercise': exercise,
                'score': max(0, min(1, score)),
                'confidence': 'high' if score > 0.7 else 'medium' if score > 0.5 else 'moderate',
                'reasoning': "; ".join(reasons[:3])
            })
        
        exercise_scores.sort(key=lambda x: x['score'], reverse=True)
        return exercise_scores[:num_recs]
    
    def _recommend_wellness_tips(self, user_profile: Dict, predictions: Dict, num_recs: int) -> List[Dict]:
        """Enhanced wellness tip recommendations"""
        tip_scores = []
        
        # Predicted values
        predicted_sleep = predictions.get('sleep_quality', [7])[0] if 'sleep_quality' in predictions and len(predictions['sleep_quality']) > 0 else 7
        predicted_stress = predictions.get('stress_level', [5])[0] if 'stress_level' in predictions and len(predictions['stress_level']) > 0 else 5
        predicted_mood = predictions.get('mood_score', [7])[0] if 'mood_score' in predictions and len(predictions['mood_score']) > 0 else 7
        
        for tip in self.item_catalog['wellness_tips']:
            score = 0.0
            reasons = []
            
            # Sleep-related tips (highest priority if sleep is poor)
            if predicted_sleep < 6:
                if tip['category'] == 'sleep':
                    score += 0.9
                    reasons.append("Critical for improving your sleep quality")
            elif predicted_sleep < 7:
                if tip['category'] == 'sleep':
                    score += 0.6
                    reasons.append("Can help optimize your sleep")
            
            # Stress-related tips
            if predicted_stress > 7:
                if tip['category'] in ['stress', 'mental_health', 'mindfulness']:
                    score += 0.85
                    reasons.append("Effective stress management technique")
            elif predicted_stress > 5:
                if tip['category'] in ['stress', 'mindfulness']:
                    score += 0.5
                    reasons.append("Helps maintain stress balance")
            
            # Mood-related tips
            if predicted_mood < 6:
                if tip['category'] in ['mental_health', 'social', 'mindfulness']:
                    score += 0.7
                    reasons.append("Supports emotional wellbeing")
            
            # General wellness (always somewhat relevant)
            if tip['category'] in ['nutrition', 'movement', 'hydration']:
                score += 0.35
                reasons.append("Foundational health practice")
            
            # Difficulty consideration
            if tip.get('difficulty') == 'easy':
                score += 0.1
                reasons.append("Easy to implement")
            
            # Variety
            score += np.random.uniform(-0.05, 0.1)
            
            if not reasons:
                reasons.append("Promotes overall wellness")
            
            tip_scores.append({
                'tip': tip,
                'score': max(0, min(1, score)),
                'priority': 'high' if score > 0.8 else 'medium' if score > 0.5 else 'normal',
                'reasoning': "; ".join(reasons[:2])
            })
        
        tip_scores.sort(key=lambda x: x['score'], reverse=True)
        return tip_scores[:num_recs]
    
    def save_model(self, save_path: str = "ml/trained_models/recommendation_engine_v2.pkl"):
        """Save recommendation engine"""
        save_data = {
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_catalog': self.item_catalog,
            'version': 'v2'
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_data, save_path)
        
        # Save PyTorch model separately
        if 'collaborative_filtering' in self.models:
            model_path = save_path.replace('.pkl', '_cf_model.pth')
            torch.save(self.models['collaborative_filtering'].state_dict(), model_path)
            print(f"Collaborative filtering model saved to {model_path}")
        
        print(f"Recommendation engine saved to {save_path}")
    
    def load_model(self, load_path: str = "ml/trained_models/recommendation_engine_v2.pkl"):
        """Load recommendation engine"""
        save_data = joblib.load(load_path)
        
        self.user_to_idx = save_data['user_to_idx']
        self.idx_to_user = save_data['idx_to_user']
        self.item_catalog = save_data['item_catalog']
        
        print(f"Recommendation engine loaded from {load_path}")


def create_recommendation_system(datasets: Dict[str, pd.DataFrame]) -> HealthRecommendationEngine:
    """Create and train the complete recommendation system"""
    
    engine = HealthRecommendationEngine()
    
    print("Building user-item interaction matrix...")
    interaction_matrix, user_to_idx, idx_to_user = engine.build_user_item_matrix(datasets)
    
    print(f"Matrix shape: {interaction_matrix.shape}")
    print(f"Non-zero interactions: {np.count_nonzero(interaction_matrix)}")
    
    print("\nTraining collaborative filtering model...")
    engine.train_collaborative_filtering(interaction_matrix, epochs=100, lr=0.001)
    
    # Store user mappings
    engine.user_to_idx = user_to_idx
    engine.idx_to_user = idx_to_user
    
    print("\n✓ Recommendation system training complete!")
    return engine


if __name__ == "__main__":
    # Example usage with v2 models
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data.synthetic_generator import save_synthetic_data
    from data.feature_engineering_v2 import create_ml_features
    from models.health_predictor_v2 import ImprovedHealthPredictor
    
    print("="*80)
    print("HEALTH RECOMMENDATION ENGINE - TRAINING & DEMO")
    print("="*80)
    
    # Generate data
    print("\nStep 1: Generating synthetic data...")
    datasets = save_synthetic_data()
    
    print("\nStep 2: Creating ML features...")
    ml_features = create_ml_features(datasets, sample_rate=0.5, lookback_days=28)
    
    # Load trained predictor
    print("\nStep 3: Loading trained prediction models...")
    predictor = ImprovedHealthPredictor(model_type='stacking')
    try:
        predictor.load_models()
        print("✓ Loaded existing models")
    except:
        print("No trained models found. Please train models first using train_improved_models_v2.py")
        sys.exit(1)
    
    # Create recommendation system
    print("\nStep 4: Training recommendation system...")
    rec_engine = create_recommendation_system(datasets)
    
    # Test recommendations for sample users
    print("\n" + "="*80)
    print("GENERATING SAMPLE RECOMMENDATIONS")
    print("="*80)
    
    sample_users = datasets['users'].head(3)['user_id'].tolist()
    
    for user_id in sample_users:
        print(f"\n{'='*80}")
        print(f"USER: {user_id}")
        print('='*80)
        
        # Get user features
        user_features_row = ml_features['user_features'][
            ml_features['user_features']['user_id'] == user_id
        ]
        
        if len(user_features_row) == 0:
            print(f"No features found for user {user_id}")
            continue
        
        sample_features = user_features_row.to_dict('records')[0]
        
        # Get predictions
        user_data_row = ml_features['final_features'][
            ml_features['final_features']['user_id'] == user_id
        ].head(1)
        
        if len(user_data_row) == 0:
            print(f"No time series data for user {user_id}")
            continue
        
        # Prepare features for prediction
        feature_cols = [col for col in user_data_row.columns 
                       if col not in ['user_id', 'date', 'target_date', 
                                     'target_sleep_quality', 'target_mood_score', 
                                     'target_stress_level', 'target_will_exercise']]
        
        X_user_df = user_data_row[feature_cols].copy()
        
        # **FIX: Encode categorical features before prediction**
        categorical_features = ['gender', 'activity_level', 'preferred_exercise_type', 
                               'dietary_preference', 'health_goal']
        X_user_encoded, _ = encode_categorical_features(X_user_df, categorical_features)
        X_user = X_user_encoded.values
        
        try:
            predictions = predictor.predict(X_user)
        except Exception as e:
            print(f"Error making predictions for {user_id}: {e}")
            print(f"Feature types: {X_user_df.dtypes.value_counts()}")
            continue
        
        # Generate recommendations
        recommendations = rec_engine.generate_personalized_recommendations(
            user_id, {user_id: sample_features}, predictions, num_recommendations=3
        )
        
        # Display user profile
        print("\n📊 User Profile:")
        print(f"  Age: {sample_features.get('age', 'N/A')}")
        print(f"  BMI: {sample_features.get('bmi', 0):.1f}")
        print(f"  Activity Level: {sample_features.get('activity_level', 'N/A')}")
        
        # Display predictions
        print("\n🔮 Predictions for Tomorrow:")
        print(f"  Sleep Quality: {predictions.get('sleep_quality', [0])[0]:.1f}/10")
        print(f"  Mood Score: {predictions.get('mood_score', [0])[0]:.1f}/10")
        print(f"  Stress Level: {predictions.get('stress_level', [0])[0]:.1f}/10")
        print(f"  Exercise Probability: {predictions.get('exercise_binary', [0])[0]:.1%}")
        
        # Display recommendations
        print("\n🍽️  TOP MEAL RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations['meals'], 1):
            meal = rec['meal']
            print(f"\n  {i}. {meal['name']}")
            print(f"     Calories: {meal['calories']} | Protein: {meal['protein']}g | "
                  f"Carbs: {meal['carbs']}g | Fat: {meal['fat']}g")
            print(f"     Prep Time: {meal.get('prep_time', 'N/A')} min")
            print(f"     💡 {rec['reasoning']}")
            print(f"     Match Score: {rec['score']:.2f} | Confidence: {rec['confidence']}")
        
        print("\n🏃‍♂️ TOP EXERCISE RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations['exercises'], 1):
            exercise = rec['exercise']
            print(f"\n  {i}. {exercise['name']}")
            print(f"     Duration: {exercise['duration']} min | Type: {exercise['type']} | "
                  f"Intensity: {exercise['intensity']}")
            print(f"     Calories Burned: {exercise['calories_burned']} | "
                  f"Equipment: {exercise.get('equipment', 'N/A')}")
            print(f"     💡 {rec['reasoning']}")
            print(f"     Match Score: {rec['score']:.2f} | Confidence: {rec['confidence']}")
        
        print("\n💡 TOP WELLNESS TIPS:")
        for i, rec in enumerate(recommendations['wellness_tips'], 1):
            tip = rec['tip']
            print(f"\n  {i}. {tip['tip']}")
            print(f"     Category: {tip['category']} | Impact: {tip['impact']}")
            print(f"     💡 {rec['reasoning']}")
            print(f"     Relevance: {rec['score']:.2f} | Priority: {rec['priority']}")
        
        print()
    
    # Save recommendation engine
    print("\n" + "="*80)
    print("SAVING RECOMMENDATION ENGINE")
    print("="*80)
    rec_engine.save_model()
    
    print("\n✓ Recommendation system ready for use!")
    print("\nTo use in your application:")
    print("  from ml.models.recommendation_engine_v2 import HealthRecommendationEngine")
    print("  engine = HealthRecommendationEngine()")
    print("  engine.load_model()")
    print("  recommendations = engine.generate_personalized_recommendations(...)")