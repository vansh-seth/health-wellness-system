import optuna
from typing import Dict, Any, Callable
import numpy as np
from sklearn.model_selection import cross_val_score

class HyperparameterTuner:
    """AutoML hyperparameter tuning using Optuna"""
    
    def __init__(self, n_trials: int = 50):
        self.n_trials = n_trials
        self.best_params = {}
        self.studies = {}
    
    def tune_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray, 
                              target_name: str) -> Dict[str, Any]:
        """Tune GradientBoostingRegressor hyperparameters"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(**params, random_state=42)
            
            # Cross-validation score
            scores = cross_val_score(model, X_train, y_train, cv=5, 
                                    scoring='neg_mean_squared_error')
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params[target_name] = study.best_params
        self.studies[target_name] = study
        
        print(f"\nBest params for {target_name}:")
        print(study.best_params)
        print(f"Best score: {-study.best_value:.4f}")
        
        return study.best_params