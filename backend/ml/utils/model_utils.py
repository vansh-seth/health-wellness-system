import torch
import numpy as np
from typing import Dict, Any
import json

class ModelUtils:
    """Utility functions for ML models"""
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        """Count trainable parameters in PyTorch model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def save_model_metadata(metadata: Dict[str, Any], filepath: str):
        """Save model metadata as JSON"""
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load_model_metadata(filepath: str) -> Dict[str, Any]:
        """Load model metadata from JSON"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def set_seed(seed: int = 42):
        """Set random seeds for reproducibility"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def get_device():
        """Get available device (CUDA or CPU)"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')