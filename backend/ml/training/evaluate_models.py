# ml/training/evaluate_models.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class ModelEvaluator:
    """Complete model evaluation and visualization system"""
    
    def __init__(self, output_dir: str = "ml/evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        self.plots = []
        
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           model_name: str, target_name: str) -> Dict[str, float]:
        """Evaluate regression model with comprehensive metrics"""
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        
        # Explained variance
        explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        # Max error
        max_error = np.max(np.abs(y_true - y_pred))
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape),
            'explained_variance': float(explained_var),
            'max_error': float(max_error),
            'mean_prediction': float(np.mean(y_pred)),
            'std_prediction': float(np.std(y_pred)),
            'mean_true': float(np.mean(y_true)),
            'std_true': float(np.std(y_true))
        }
        
        key = f"{model_name}_{target_name}"
        self.metrics[key] = metrics
        
        print(f"\n{model_name} - {target_name} Metrics:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_pred_proba: Optional[np.ndarray], 
                               model_name: str, target_name: str) -> Dict[str, float]:
        """Evaluate classification model"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        # AUC-ROC if probabilities available
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            try:
                auc_roc = roc_auc_score(y_true, y_pred_proba)
                metrics['auc_roc'] = float(auc_roc)
            except:
                metrics['auc_roc'] = 0.0
        
        key = f"{model_name}_{target_name}"
        self.metrics[key] = metrics
        
        print(f"\n{model_name} - {target_name} Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        if 'auc_roc' in metrics:
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return metrics
    
    def plot_regression_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str, target_name: str):
        """Create comprehensive regression plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0, 0].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'{model_name} - {target_name}\nActual vs Predicted')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', 
                       transform=axes[0, 0].transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Residuals Plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals Distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add statistics
        axes[1, 0].text(0.05, 0.95, 
                       f'Mean: {np.mean(residuals):.4f}\nStd: {np.std(residuals):.4f}',
                       transform=axes[1, 0].transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Error Distribution
        errors = np.abs(residuals)
        axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Absolute Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add MAE to plot
        mae = np.mean(errors)
        axes[1, 1].axvline(x=mae, color='r', linestyle='--', lw=2, label=f'MAE = {mae:.4f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f'{model_name}_{target_name}_regression.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved plot: {save_path}")
        plt.close()
        
        self.plots.append(str(save_path))
    
    def plot_classification_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_pred_proba: Optional[np.ndarray],
                                    model_name: str, target_name: str):
        """Create comprehensive classification plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   cbar_kws={'label': 'Count'})
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_title(f'{model_name} - {target_name}\nConfusion Matrix')
        
        # 2. Class Distribution
        unique, counts = np.unique(y_true, return_counts=True)
        axes[0, 1].bar(['Class 0', 'Class 1'], counts, color=['#1f77b4', '#ff7f0e'])
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('True Class Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add counts on bars
        for i, count in enumerate(counts):
            axes[0, 1].text(i, count, str(count), ha='center', va='bottom')
        
        # 3. ROC Curve (if probabilities available)
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                auc_score = roc_auc_score(y_true, y_pred_proba)
                
                axes[1, 0].plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
                axes[1, 0].plot([0, 1], [0, 1], 'r--', lw=2, label='Random Classifier')
                axes[1, 0].set_xlabel('False Positive Rate')
                axes[1, 0].set_ylabel('True Positive Rate')
                axes[1, 0].set_title('ROC Curve')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            except:
                axes[1, 0].text(0.5, 0.5, 'ROC Curve\nNot Available',
                               ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, 'ROC Curve\nNot Available',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. Precision-Recall Curve
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                
                axes[1, 1].plot(recall, precision, lw=2, label='PR Curve')
                axes[1, 1].set_xlabel('Recall')
                axes[1, 1].set_ylabel('Precision')
                axes[1, 1].set_title('Precision-Recall Curve')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            except:
                axes[1, 1].text(0.5, 0.5, 'PR Curve\nNot Available',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'PR Curve\nNot Available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f'{model_name}_{target_name}_classification.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved plot: {save_path}")
        plt.close()
        
        self.plots.append(str(save_path))
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importances: np.ndarray,
                               model_name: str, top_n: int = 20):
        """Plot feature importance"""
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title(f'{model_name} - Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f'{model_name}_feature_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved plot: {save_path}")
        plt.close()
        
        self.plots.append(str(save_path))
    
    def plot_learning_curves(self, train_scores: List[float], 
                            val_scores: List[float],
                            model_name: str):
        """Plot learning curves"""
        
        epochs = range(1, len(train_scores) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_scores, 'b-', label='Training Score', linewidth=2)
        plt.plot(epochs, val_scores, 'r-', label='Validation Score', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title(f'{model_name} - Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f'{model_name}_learning_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved plot: {save_path}")
        plt.close()
        
        self.plots.append(str(save_path))
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        
        if not self.metrics:
            print("No metrics available for dashboard")
            return
        
        # Prepare data
        models = []
        targets = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        for key, metrics in self.metrics.items():
            model, target = key.rsplit('_', 1)
            models.append(model)
            targets.append(target)
            r2_scores.append(metrics.get('r2_score', 0))
            rmse_scores.append(metrics.get('rmse', 0))
            mae_scores.append(metrics.get('mae', 0))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² Score Comparison', 'RMSE Comparison',
                          'MAE Comparison', 'Model Performance Overview'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # R² Score
        fig.add_trace(
            go.Bar(x=targets, y=r2_scores, name='R² Score', 
                  marker_color='lightblue'),
            row=1, col=1
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(x=targets, y=rmse_scores, name='RMSE',
                  marker_color='lightcoral'),
            row=1, col=2
        )
        
        # MAE
        fig.add_trace(
            go.Bar(x=targets, y=mae_scores, name='MAE',
                  marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Scatter: R² vs RMSE
        fig.add_trace(
            go.Scatter(x=r2_scores, y=rmse_scores, mode='markers+text',
                      text=targets, textposition='top center',
                      marker=dict(size=12, color='purple'),
                      name='Performance'),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Target", row=1, col=1)
        fig.update_xaxes(title_text="Target", row=1, col=2)
        fig.update_xaxes(title_text="Target", row=2, col=1)
        fig.update_xaxes(title_text="R² Score", row=2, col=2)
        
        fig.update_yaxes(title_text="R² Score", row=1, col=1)
        fig.update_yaxes(title_text="RMSE", row=1, col=2)
        fig.update_yaxes(title_text="MAE", row=2, col=1)
        fig.update_yaxes(title_text="RMSE", row=2, col=2)
        
        fig.update_layout(
            title_text="Health Prediction Models - Performance Dashboard",
            showlegend=False,
            height=800,
            width=1400
        )
        
        # Save interactive dashboard
        save_path = self.output_dir / 'interactive_dashboard.html'
        fig.write_html(str(save_path))
        print(f"\n  Saved interactive dashboard: {save_path}")
        self.plots.append(str(save_path))
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        
        report_lines = [
            "="*80,
            "HEALTH PREDICTION MODELS - EVALUATION REPORT",
            "="*80,
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Models Evaluated: {len(self.metrics)}",
            "\n" + "="*80,
            "\nDETAILED METRICS",
            "="*80
        ]
        
        for key, metrics in self.metrics.items():
            model, target = key.rsplit('_', 1)
            report_lines.append(f"\n{model} - {target}:")
            report_lines.append("-" * 60)
            
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric_name:.<40} {value:.4f}")
                else:
                    report_lines.append(f"  {metric_name:.<40} {value}")
        
        report_lines.extend([
            "\n" + "="*80,
            "SUMMARY STATISTICS",
            "="*80
        ])
        
        # Calculate average metrics
        if self.metrics:
            avg_r2 = np.mean([m.get('r2_score', 0) for m in self.metrics.values()])
            avg_rmse = np.mean([m.get('rmse', 0) for m in self.metrics.values()])
            avg_mae = np.mean([m.get('mae', 0) for m in self.metrics.values()])
            
            report_lines.extend([
                f"\nAverage R² Score: {avg_r2:.4f}",
                f"Average RMSE: {avg_rmse:.4f}",
                f"Average MAE: {avg_mae:.4f}"
            ])
        
        report_lines.extend([
            "\n" + "="*80,
            f"GENERATED PLOTS ({len(self.plots)} files)",
            "="*80
        ])
        
        for plot_path in self.plots:
            report_lines.append(f"  - {plot_path}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        # Print report
        print("\n" + report_text)
        
        # Save report
        report_path = self.output_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Save metrics as JSON
        json_path = self.output_dir / 'metrics.json'
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"\n  Report saved: {report_path}")
        print(f"  Metrics JSON saved: {json_path}")
        
        return report_text


def evaluate_all_models():
    """Complete evaluation pipeline for all trained models"""
    
    print("\n" + "="*80)
    print("EVALUATING ALL HEALTH PREDICTION MODELS")
    print("="*80)
    
    # Import required modules
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from ml.data.synthetic_generator import save_synthetic_data
    from ml.data.feature_engineering import create_ml_features
    from ml.models.health_predictor import HealthPredictor
    from sklearn.model_selection import train_test_split
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load or generate data
    print("\n[1/5] Loading data...")
    data_dir = Path("ml/data/synthetic")
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        print("  Generating synthetic data...")
        datasets = save_synthetic_data()
    else:
        print("  Loading existing data...")
        datasets = {
            'users': pd.read_csv(data_dir / 'users.csv'),
            'meals': pd.read_csv(data_dir / 'meals.csv'),
            'exercises': pd.read_csv(data_dir / 'exercises.csv'),
            'sleep': pd.read_csv(data_dir / 'sleep.csv'),
            'mood': pd.read_csv(data_dir / 'mood.csv')
        }
    
    # Create features
    print("\n[2/5] Creating features...")
    ml_features = create_ml_features(datasets)
    
    X = ml_features['X_train']
    y_dict = {
        'sleep_quality': ml_features['y_train'][:, 0],
        'mood_score': ml_features['y_train'][:, 1],
        'stress_level': ml_features['y_train'][:, 2],
        'exercise_binary': ml_features['y_train'][:, 3]
    }
    feature_names = ml_features['feature_names']
    
    # Split data
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    y_train = {k: v[:len(X_train)] for k, v in y_dict.items()}
    y_test = {k: v[len(X_train):] for k, v in y_dict.items()}
    
    # Load or train predictor
    print("\n[3/5] Loading/Training predictor...")
    predictor = HealthPredictor(model_type='ensemble')
    
    models_dir = Path("ml/trained_models")
    if models_dir.exists() and (models_dir / "sleep_quality_predictor.pkl").exists():
        print("  Loading existing models...")
        predictor.load_models()
    else:
        print("  Training new models...")
        predictor.build_models(X.shape[1])
        predictor.train(X_train, y_train, X_test, y_test)
        predictor.save_models()
    
    # Make predictions
    print("\n[4/5] Making predictions and evaluating...")
    predictions = predictor.predict(X_test)
    
    # Evaluate regression models
    print("\n  Evaluating Regression Models:")
    for target in ['sleep_quality', 'mood_score', 'stress_level']:
        metrics = evaluator.evaluate_regression(
            y_test[target], predictions[target],
            'HealthPredictor', target
        )
        
        # Create plots
        evaluator.plot_regression_results(
            y_test[target], predictions[target],
            'HealthPredictor', target
        )
    
    # Evaluate classification model
    print("\n  Evaluating Classification Model:")
    y_pred_binary = (predictions['exercise_binary'] > 0.5).astype(int)
    metrics = evaluator.evaluate_classification(
        y_test['exercise_binary'].astype(int), 
        y_pred_binary,
        predictions['exercise_binary'],
        'HealthPredictor', 'exercise'
    )
    
    evaluator.plot_classification_results(
        y_test['exercise_binary'].astype(int),
        y_pred_binary,
        predictions['exercise_binary'],
        'HealthPredictor', 'exercise'
    )
    
    # Feature importance
    print("\n  Plotting feature importance...")
    importance_dict = predictor.get_feature_importance(feature_names)
    for target, importances in importance_dict.items():
        importance_array = np.array(list(importances.values()))
        evaluator.plot_feature_importance(
            list(importances.keys()), importance_array,
            f'HealthPredictor_{target}', top_n=20
        )
    
    # Create interactive dashboard
    print("\n[5/5] Creating interactive dashboard...")
    evaluator.create_interactive_dashboard()
    
    # Generate report
    print("\n  Generating evaluation report...")
    evaluator.generate_report()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {evaluator.output_dir}")
    print(f"Total plots generated: {len(evaluator.plots)}")
    print("\nOpen 'interactive_dashboard.html' in your browser for interactive visualization!")
    
    return evaluator


if __name__ == "__main__":
    evaluator = evaluate_all_models()