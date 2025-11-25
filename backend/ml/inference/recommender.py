from typing import Dict, List, Any

class RecommendationInferenceEngine:
    """Real-time recommendations"""
    
    def __init__(self, models_dir: str = "ml/trained_models"):
        self.models_dir = Path(models_dir)
        self.rec_engine = None
        self.loaded = False
    
    def load_models(self):
        """Load recommendation models"""
        
        try:
            from ml.models.recommendation_engine import HealthRecommendationEngine
            self.rec_engine = HealthRecommendationEngine()
            # Load saved models
            self.loaded = True
            print("Recommendation models loaded!")
            
        except Exception as e:
            print(f"Error loading recommendation models: {e}")
            self.loaded = False
    
    def get_recommendations(self, user_id: str, user_profile: Dict, 
                          predictions: Dict, num_recs: int = 5) -> Dict[str, List[Dict]]:
        """Get real-time recommendations"""
        
        if not self.loaded:
            raise ValueError("Models not loaded.")
        
        recommendations = self.rec_engine.generate_personalized_recommendations(
            user_id, {user_id: user_profile}, predictions, num_recs
        )
        
        return recommendations

