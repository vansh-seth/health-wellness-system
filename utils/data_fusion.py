import numpy as np
import torch
from typing import Dict, List

class DataFusionUtils:
    """Utilities for multi-modal data fusion"""
    
    @staticmethod
    def early_fusion(modalities: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate all modality features"""
        return np.concatenate(list(modalities.values()), axis=1)
    
    @staticmethod
    def late_fusion(predictions: Dict[str, np.ndarray], weights: Dict[str, float] = None) -> np.ndarray:
        """Weighted average of predictions from different modalities"""
        
        if weights is None:
            weights = {k: 1.0 / len(predictions) for k in predictions.keys()}
        
        fused = np.zeros_like(list(predictions.values())[0])
        for modality, preds in predictions.items():
            fused += weights.get(modality, 1.0) * preds
        
        return fused
    
    @staticmethod
    def attention_weights_to_importance(attention_scores: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Convert attention scores to importance percentages"""
        
        importance = {}
        total = sum(scores.mean().item() for scores in attention_scores.values())
        
        for modality, scores in attention_scores.items():
            importance[modality] = (scores.mean().item() / total) * 100
        
        return importance