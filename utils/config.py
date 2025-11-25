from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MLConfig:
    """ML system configuration"""
    
    # Data settings
    synthetic_users: int = 1000
    days_history: int = 90
    train_test_split: float = 0.2
    validation_split: float = 0.1
    
    # Model settings
    predictor_type: str = 'ensemble'  # 'ensemble', 'deep_learning', 'traditional'
    fusion_hidden_dim: int = 256
    fusion_epochs: int = 100
    predictor_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Feature engineering
    lookback_days: int = 7
    aggregation_windows: List[int] = None
    
    # Anomaly detection
    anomaly_method: str = 'ensemble'
    anomaly_threshold: float = 0.95
    
    # Recommendation
    num_recommendations: int = 5
    recommendation_diversity: float = 0.3
    
    # Paths
    data_dir: str = "ml/data/synthetic"
    models_dir: str = "ml/trained_models"
    logs_dir: str = "ml/logs"
    
    def __post_init__(self):
        if self.aggregation_windows is None:
            self.aggregation_windows = [3, 7, 14, 30]

# Global config instance
ml_config = MLConfig()