# ml/models/recommendation_engine.py
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

class NeuralCollaborativeFiltering(nn.Module):
    """Neural collaborative filtering for personalized health recommendations"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Neural MF layers
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
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
                {'id': 0, 'name': 'Mediterranean Salad', 'calories': 350, 'protein': 15, 'carbs': 25, 'fat': 22, 'tags': ['healthy', 'light', 'vegetarian']},
                {'id': 1, 'name': 'Grilled Chicken Breast', 'calories': 280, 'protein': 53, 'carbs': 0, 'fat': 6, 'tags': ['high-protein', 'low-carb']},
                {'id': 2, 'name': 'Quinoa Bowl', 'calories': 420, 'protein': 18, 'carbs': 65, 'fat': 8, 'tags': ['vegetarian', 'high-fiber', 'complete-protein']},
                {'id': 3, 'name': 'Salmon with Vegetables', 'calories': 380, 'protein': 42, 'carbs': 12, 'fat': 18, 'tags': ['omega-3', 'anti-inflammatory']},
                {'id': 4, 'name': 'Greek Yogurt with Berries', 'calories': 180, 'protein': 20, 'carbs': 15, 'fat': 5, 'tags': ['probiotics', 'antioxidants']},
                {'id': 5, 'name': 'Avocado Toast', 'calories': 320, 'protein': 12, 'carbs': 30, 'fat': 18, 'tags': ['healthy-fats', 'fiber']},
                {'id': 6, 'name': 'Protein Smoothie', 'calories': 250, 'protein': 25, 'carbs': 20, 'fat': 8, 'tags': ['post-workout', 'quick']},
                {'id': 7, 'name': 'Lentil Soup', 'calories': 230, 'protein': 16, 'carbs': 35, 'fat': 3, 'tags': ['plant-based', 'fiber', 'warming']},
                {'id': 8, 'name': 'Oatmeal with Nuts', 'calories': 290, 'protein': 12, 'carbs': 42, 'fat': 10, 'tags': ['fiber', 'sustained-energy']},
                {'id': 9, 'name': 'Turkey Wrap', 'calories': 340, 'protein': 28, 'carbs': 38, 'fat': 9, 'tags': ['balanced', 'portable']},
            ],
            
            'exercises': [
                {'id': 0, 'name': '30-min Morning Walk', 'duration': 30, 'type': 'cardio', 'intensity': 'low', 'calories_burned': 120},
                {'id': 1, 'name': 'HIIT Training', 'duration': 20, 'type': 'cardio', 'intensity': 'high', 'calories_burned': 250},
                {'id': 2, 'name': 'Strength Training', 'duration': 45, 'type': 'strength', 'intensity': 'moderate', 'calories_burned': 200},
                {'id': 3, 'name': 'Yoga Session', 'duration': 30, 'type': 'flexibility', 'intensity': 'low', 'calories_burned': 90},
                {'id': 4, 'name': 'Swimming', 'duration': 40, 'type': 'cardio', 'intensity': 'moderate', 'calories_burned': 300},
                {'id': 5, 'name': 'Cycling', 'duration': 45, 'type': 'cardio', 'intensity': 'moderate', 'calories_burned': 280},
                {'id': 6, 'name': 'Pilates', 'duration': 35, 'type': 'flexibility', 'intensity': 'moderate', 'calories_burned': 120},
                {'id': 7, 'name': 'Rock Climbing', 'duration': 60, 'type': 'strength', 'intensity': 'high', 'calories_burned': 400},
                {'id': 8, 'name': 'Jogging', 'duration': 35, 'type': 'cardio', 'intensity': 'moderate', 'calories_burned': 220},
                {'id': 9, 'name': 'Dance Class', 'duration': 50, 'type': 'cardio', 'intensity': 'moderate', 'calories_burned': 260},
            ],
            
            'wellness_tips': [
                {'id': 0, 'category': 'sleep', 'tip': 'Try the 4-7-8 breathing technique before bed', 'impact': 'stress_reduction'},
                {'id': 1, 'category': 'nutrition', 'tip': 'Eat the rainbow - include 5 different colored fruits/vegetables', 'impact': 'nutrient_variety'},
                {'id': 2, 'category': 'mental_health', 'tip': 'Practice gratitude journaling for 5 minutes daily', 'impact': 'mood_improvement'},
                {'id': 3, 'category': 'hydration', 'tip': 'Drink a glass of water before each meal', 'impact': 'better_digestion'},
                {'id': 4, 'category': 'movement', 'tip': 'Take a 5-minute walk every hour during work', 'impact': 'energy_boost'},
                {'id': 5, 'category': 'sleep', 'tip': 'Keep bedroom temperature between 65-68°F', 'impact': 'sleep_quality'},
                {'id': 6, 'category': 'stress', 'tip': 'Try progressive muscle relaxation', 'impact': 'anxiety_reduction'},
                {'id': 7, 'category': 'social', 'tip': 'Connect with a friend or family member today', 'impact': 'emotional_wellbeing'},
                {'id': 8, 'category': 'mindfulness', 'tip': 'Practice 10 minutes of meditation daily', 'impact': 'stress_management'},
                {'id': 9, 'category': 'sleep', 'tip': 'Establish a consistent sleep schedule', 'impact': 'circadian_rhythm'},
            ]
        }
        return catalog
    
    def build_user_item_matrix(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
        """Build user-item interaction matrix from historical data"""
        users = datasets['users']['user_id'].unique()
        
        # Create user mappings
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        
        # Initialize interaction matrices for different item types
        num_users = len(users)
        
        # For meals - based on meal preferences and ratings
        meal_matrix = np.zeros((num_users, len(self.item_catalog['meals'])))
        
        for user_idx, user_id in enumerate(users):
            user_meals = datasets['meals'][datasets['meals']['user_id'] == user_id]
            
            if len(user_meals) > 0:
                # Calculate meal preferences based on frequency and nutritional patterns
                avg_calories = user_meals['calories'].mean()
                avg_protein = user_meals['protein'].mean()
                
                for meal_idx, meal in enumerate(self.item_catalog['meals']):
                    # Similarity based on nutritional preferences
                    calorie_sim = 1 - min(abs(meal['calories'] - avg_calories) / 1000, 1)
                    protein_sim = 1 - min(abs(meal['protein'] - avg_protein) / 50, 1)
                    
                    # Combine similarities (scale 0-5 like ratings)
                    preference_score = max(0, min(5, (calorie_sim + protein_sim) * 2.5))
                    meal_matrix[user_idx, meal_idx] = preference_score
        
        return meal_matrix, user_to_idx, idx_to_user
    
    def train_collaborative_filtering(self, interaction_matrix: np.ndarray, epochs: int = 100) -> NeuralCollaborativeFiltering:
        """Train neural collaborative filtering model"""
        num_users, num_items = interaction_matrix.shape
        
        # Create training data from interaction matrix
        user_ids, item_ids, ratings = [], [], []
        
        for user_idx in range(num_users):
            for item_idx in range(num_items):
                if interaction_matrix[user_idx, item_idx] > 0:
                    user_ids.append(user_idx)
                    item_ids.append(item_idx)
                    ratings.append(interaction_matrix[user_idx, item_idx] / 5.0)  # Normalize to 0-1
                    
                    # Add some negative samples
                    if np.random.random() < 0.1:  # 10% negative samples
                        neg_item = np.random.randint(num_items)
                        if interaction_matrix[user_idx, neg_item] == 0:
                            user_ids.append(user_idx)
                            item_ids.append(neg_item)
                            ratings.append(0.0)
        
        if len(user_ids) == 0:
            print("Warning: No training data for collaborative filtering")
            return None
        
        # Convert to tensors
        user_tensor = torch.LongTensor(user_ids).to(self.device)
        item_tensor = torch.LongTensor(item_ids).to(self.device)
        rating_tensor = torch.FloatTensor(ratings).unsqueeze(1).to(self.device)
        
        # Initialize model
        model = NeuralCollaborativeFiltering(num_users, num_items).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(user_tensor, item_tensor)
            loss = criterion(predictions, rating_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        self.models['collaborative_filtering'] = model
        return model
    
    def generate_personalized_recommendations(self, user_id: str, user_features: Dict, prediction_results: Dict, 
                                           num_recommendations: int = 5) -> Dict[str, List[Dict]]:
        """Generate personalized recommendations based on user profile and ML predictions"""
        
        recommendations = {
            'meals': [],
            'exercises': [],
            'wellness_tips': []
        }
        
        # Extract user characteristics
        user_profile = user_features.get(user_id, {})
        
        # Meal recommendations
        meal_recs = self._recommend_meals(user_profile, prediction_results, num_recommendations)
        recommendations['meals'] = meal_recs
        
        # Exercise recommendations
        exercise_recs = self._recommend_exercises(user_profile, prediction_results, num_recommendations)
        recommendations['exercises'] = exercise_recs
        
        # Wellness tip recommendations
        wellness_recs = self._recommend_wellness_tips(user_profile, prediction_results, num_recommendations)
        recommendations['wellness_tips'] = wellness_recs
        
        return recommendations
    
    def _recommend_meals(self, user_profile: Dict, predictions: Dict, num_recs: int) -> List[Dict]:
        """Recommend meals based on user profile and predictions"""
        meal_scores = []
        
        # Get user preferences
        target_calories = user_profile.get('avg_daily_calories', 2000) / 3  # Per meal
        activity_level = user_profile.get('activity_level', 'moderately_active')
        health_goals = user_profile.get('health_goals', [])
        
        # Predicted stress level affects food recommendations
        predicted_stress = predictions.get('stress_level', [5])[0] if 'stress_level' in predictions and len(predictions['stress_level']) > 0 else 5
        
        for meal in self.item_catalog['meals']:
            score = 0
            
            # Calorie matching
            calorie_diff = abs(meal['calories'] - target_calories)
            score += max(0, 1 - calorie_diff / max(target_calories, 1))
            
            # Activity level matching
            if activity_level == 'very_active' and meal['protein'] > 40:
                score += 0.3
            elif activity_level == 'sedentary' and meal['calories'] < 300:
                score += 0.3
            
            # Health goals
            if 'weight_loss' in health_goals and meal['calories'] < target_calories:
                score += 0.4
            if 'muscle_gain' in health_goals and meal['protein'] > 25:
                score += 0.4
            
            # Stress-based recommendations
            if predicted_stress > 7:
                if 'anti-inflammatory' in meal.get('tags', []) or 'omega-3' in meal.get('tags', []):
                    score += 0.3
            
            # Add some randomness for diversity
            score += np.random.normal(0, 0.1)
            
            meal_scores.append({
                'meal': meal,
                'score': max(0, score),  # Ensure non-negative
                'reasoning': self._generate_meal_reasoning(meal, user_profile, predicted_stress)
            })
        
        # Sort and return top recommendations
        meal_scores.sort(key=lambda x: x['score'], reverse=True)
        return meal_scores[:num_recs]
    
    def _recommend_exercises(self, user_profile: Dict, predictions: Dict, num_recs: int) -> List[Dict]:
        """Recommend exercises based on user profile and predictions"""
        exercise_scores = []
        
        # Get user preferences
        activity_level = user_profile.get('activity_level', 'moderately_active')
        preferred_exercise = user_profile.get('preferred_exercise_type', 'mixed')
        age = user_profile.get('age', 35)
        
        # Predicted exercise likelihood
        will_exercise_prob = predictions.get('exercise_binary', [0.5])[0] if 'exercise_binary' in predictions and len(predictions['exercise_binary']) > 0 else 0.5
        predicted_mood = predictions.get('mood_score', [7])[0] if 'mood_score' in predictions and len(predictions['mood_score']) > 0 else 7
        
        for exercise in self.item_catalog['exercises']:
            score = 0
            
            # Activity level matching
            intensity_match = {
                'sedentary': {'low': 0.8, 'moderate': 0.3, 'high': 0.1},
                'lightly_active': {'low': 0.5, 'moderate': 0.8, 'high': 0.3},
                'moderately_active': {'low': 0.3, 'moderate': 0.8, 'high': 0.6},
                'very_active': {'low': 0.2, 'moderate': 0.6, 'high': 0.9}
            }
            score += intensity_match.get(activity_level, {}).get(exercise['intensity'], 0.5)
            
            # Exercise type preference
            if preferred_exercise == exercise['type'] or preferred_exercise == 'mixed':
                score += 0.3
            
            # Age considerations
            if age > 60 and exercise['intensity'] == 'low':
                score += 0.2
            elif age < 30 and exercise['intensity'] == 'high':
                score += 0.2
            
            # Mood-based recommendations
            if predicted_mood < 6:
                if exercise['type'] == 'cardio' or exercise['name'] == 'Yoga Session':
                    score += 0.3  # Endorphins and mindfulness
            
            # Exercise probability adjustment
            if will_exercise_prob < 0.3:
                # Recommend easier, shorter exercises
                if exercise['duration'] < 30 and exercise['intensity'] == 'low':
                    score += 0.4
            
            # Add diversity
            score += np.random.normal(0, 0.1)
            
            exercise_scores.append({
                'exercise': exercise,
                'score': max(0, score),  # Ensure non-negative
                'reasoning': self._generate_exercise_reasoning(exercise, user_profile, predicted_mood)
            })
        
        exercise_scores.sort(key=lambda x: x['score'], reverse=True)
        return exercise_scores[:num_recs]
    
    def _recommend_wellness_tips(self, user_profile: Dict, predictions: Dict, num_recs: int) -> List[Dict]:
        """Recommend wellness tips based on predictions"""
        tip_scores = []
        
        predicted_sleep = predictions.get('sleep_quality', [7])[0] if 'sleep_quality' in predictions and len(predictions['sleep_quality']) > 0 else 7
        predicted_stress = predictions.get('stress_level', [5])[0] if 'stress_level' in predictions and len(predictions['stress_level']) > 0 else 5
        predicted_mood = predictions.get('mood_score', [7])[0] if 'mood_score' in predictions and len(predictions['mood_score']) > 0 else 7
        
        for tip in self.item_catalog['wellness_tips']:
            score = 0
            
            # Sleep-related tips
            if predicted_sleep < 6 and tip['category'] == 'sleep':
                score += 0.8
            
            # Stress-related tips
            if predicted_stress > 7 and tip['category'] in ['stress', 'mental_health', 'mindfulness']:
                score += 0.8
            
            # Mood-related tips
            if predicted_mood < 6 and tip['category'] in ['mental_health', 'social', 'mindfulness']:
                score += 0.7
            
            # General wellness
            if tip['category'] in ['nutrition', 'movement', 'hydration']:
                score += 0.3
            
            # Add randomness for variety
            score += np.random.normal(0, 0.1)
            
            tip_scores.append({
                'tip': tip,
                'score': max(0, score),  # Ensure non-negative
                'reasoning': self._generate_tip_reasoning(tip, predicted_sleep, predicted_stress, predicted_mood)
            })
        
        tip_scores.sort(key=lambda x: x['score'], reverse=True)
        return tip_scores[:num_recs]
    
    def _generate_meal_reasoning(self, meal: Dict, user_profile: Dict, predicted_stress: float) -> str:
        """Generate reasoning for meal recommendation"""
        reasons = []
        
        if predicted_stress > 7 and 'anti-inflammatory' in meal.get('tags', []):
            reasons.append("Anti-inflammatory properties may help reduce stress")
        
        if 'weight_loss' in user_profile.get('health_goals', []) and meal['calories'] < 300:
            reasons.append("Lower calorie option supports your weight loss goal")
        
        if 'muscle_gain' in user_profile.get('health_goals', []) and meal['protein'] > 25:
            reasons.append("High protein content supports muscle building")
        
        if meal['protein'] > 20:
            reasons.append("Good protein source for sustained energy")
        
        if not reasons:
            reasons.append("Balanced nutritional profile")
        
        return "; ".join(reasons)
    
    def _generate_exercise_reasoning(self, exercise: Dict, user_profile: Dict, predicted_mood: float) -> str:
        """Generate reasoning for exercise recommendation"""
        reasons = []
        
        if predicted_mood < 6 and exercise['type'] == 'cardio':
            reasons.append("Cardio exercise can boost mood through endorphin release")
        
        if exercise['intensity'] == 'low' and user_profile.get('activity_level') == 'sedentary':
            reasons.append("Gentle introduction to increase activity level")
        
        if exercise['duration'] < 30:
            reasons.append("Short duration makes it easy to fit into your schedule")
        
        if exercise['type'] == 'flexibility':
            reasons.append("Helps with stress relief and mobility")
        
        if not reasons:
            reasons.append("Good for overall fitness")
        
        return "; ".join(reasons)
    
    def _generate_tip_reasoning(self, tip: Dict, sleep: float, stress: float, mood: float) -> str:
        """Generate reasoning for wellness tip"""
        if sleep < 6 and tip['category'] == 'sleep':
            return "Your predicted sleep quality suggests this could help improve rest"
        elif stress > 7 and tip['category'] in ['stress', 'mental_health', 'mindfulness']:
            return "May help manage elevated stress levels"
        elif mood < 6 and tip['category'] in ['mental_health', 'social', 'mindfulness']:
            return "Could help boost your mood and emotional wellbeing"
        else:
            return "Supports overall health and wellness"
    
    def save_model(self, save_path: str = "ml/trained_models/recommendation_engine.pkl"):
        """Save recommendation engine"""
        save_data = {
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_catalog': self.item_catalog
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_data, save_path)
        
        # Save PyTorch model separately
        if 'collaborative_filtering' in self.models:
            model_path = save_path.replace('.pkl', '_cf_model.pth')
            torch.save(self.models['collaborative_filtering'].state_dict(), model_path)
        
        print(f"Recommendation engine saved to {save_path}")
    
    def load_model(self, load_path: str = "ml/trained_models/recommendation_engine.pkl"):
        """Load recommendation engine"""
        save_data = joblib.load(load_path)
        
        self.user_to_idx = save_data['user_to_idx']
        self.idx_to_user = save_data['idx_to_user']
        self.item_catalog = save_data['item_catalog']
        
        print(f"Recommendation engine loaded from {load_path}")

def create_recommendation_system(datasets: Dict[str, pd.DataFrame], ml_features: Dict) -> HealthRecommendationEngine:
    """Create and train the complete recommendation system"""
    
    engine = HealthRecommendationEngine()
    
    # Build user-item matrix and train collaborative filtering
    print("Building user-item interaction matrix...")
    interaction_matrix, user_to_idx, idx_to_user = engine.build_user_item_matrix(datasets)
    
    print("Training collaborative filtering model...")
    engine.train_collaborative_filtering(interaction_matrix)
    
    # Store user mappings
    engine.user_to_idx = user_to_idx
    engine.idx_to_user = idx_to_user
    
    print("Recommendation system training complete!")
    return engine

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data.synthetic_generator import save_synthetic_data
    from data.feature_engineering_v2 import create_ml_features
    from models.fusion_model import HealthDataFusion
    
    # Generate data and features
    print("Generating synthetic data...")
    datasets = save_synthetic_data()
    
    print("\nCreating ML features...")
    ml_features = create_ml_features(datasets)
    
    # Train fusion model for predictions
    print("\nTraining fusion model...")
    fusion = HealthDataFusion()
    fusion.train_fusion_model(ml_features['final_features'], epochs=20)
    
    # Create recommendation system
    print("\nTraining recommendation system...")
    rec_engine = create_recommendation_system(datasets, ml_features)
    
    # Test recommendations for a sample user
    sample_user = datasets['users'].iloc[0]['user_id']
    sample_features = ml_features['user_features'][ml_features['user_features']['user_id'] == sample_user].to_dict('records')[0]
    
    # Get predictions for the user
    user_data = ml_features['final_features'][ml_features['final_features']['user_id'] == sample_user].head(1)
    predictions = fusion.predict(user_data)
    
    # Generate recommendations
    print(f"\nGenerating recommendations for user {sample_user}...")
    recommendations = rec_engine.generate_personalized_recommendations(
        sample_user, {sample_user: sample_features}, predictions
    )
    
    print(f"\n{'='*60}")
    print(f"Recommendations for user {sample_user}:")
    print('='*60)
    
    print("\n🍽️  Meal Recommendations:")
    for i, rec in enumerate(recommendations['meals'][:3], 1):
        print(f"\n{i}. {rec['meal']['name']}")
        print(f"   Calories: {rec['meal']['calories']} | Protein: {rec['meal']['protein']}g")
        print(f"   Reason: {rec['reasoning']}")
        print(f"   Match Score: {rec['score']:.2f}")
    
    print("\n\n🏃‍♂️ Exercise Recommendations:")
    for i, rec in enumerate(recommendations['exercises'][:3], 1):
        print(f"\n{i}. {rec['exercise']['name']}")
        print(f"   Duration: {rec['exercise']['duration']} min | Type: {rec['exercise']['type']}")
        print(f"   Reason: {rec['reasoning']}")
        print(f"   Match Score: {rec['score']:.2f}")
    
    print("\n\n💡 Wellness Tips:")
    for i, rec in enumerate(recommendations['wellness_tips'][:3], 1):
        print(f"\n{i}. {rec['tip']['tip']}")
        print(f"   Category: {rec['tip']['category']}")
        print(f"   Reason: {rec['reasoning']}")
        print(f"   Relevance Score: {rec['score']:.2f}")
    
    print(f"\n{'='*60}")
    
    # Save recommendation engine
    rec_engine.save_model()
    print("\nRecommendation system saved successfully!")