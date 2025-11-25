# ml/evaluate_all_models.py
"""
Comprehensive Model Evaluation Script
Evaluates all trained models and generates accuracy reports
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import joblib
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from backend.ml.data.synthetic_generator import save_synthetic_data
from backend.ml.data.feature_engineering_v2 import create_ml_features
from backend.ml.models.health_predictor_v2 import ImprovedHealthPredictor
from backend.ml.models.anomaly_detector_v2 import HealthAnomalyDetector
from backend.ml.models.recommendation_engine_v2 import (
    HealthRecommendationEngine, 
    encode_categorical_features
)


def evaluate_regression_models(predictor, X_test, y_test, feature_names):
    """Evaluate regression models (sleep, mood, stress)"""
    
    print("\n" + "="*80)
    print("REGRESSION MODELS EVALUATION")
    print("="*80)
    
    results = {}
    predictions = predictor.predict(X_test)
    
    targets = ['sleep_quality', 'mood_score', 'stress_level']
    
    for i, target in enumerate(targets):
        print(f"\n{'─'*80}")
        print(f"📊 {target.upper().replace('_', ' ')}")
        print(f"{'─'*80}")
        
        y_true = y_test[:, i]
        y_pred = predictions[target]
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Accuracy within threshold (±1 point for 1-10 scale)
        accuracy_1pt = np.mean(np.abs(y_true - y_pred) <= 1) * 100
        accuracy_2pt = np.mean(np.abs(y_true - y_pred) <= 2) * 100
        
        results[target] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'max_error': max_error,
            'accuracy_within_1pt': accuracy_1pt,
            'accuracy_within_2pt': accuracy_2pt
        }
        
        print(f"  MSE:                    {mse:.4f}")
        print(f"  RMSE:                   {rmse:.4f}")
        print(f"  MAE:                    {mae:.4f}")
        print(f"  R² Score:               {r2:.4f}")
        print(f"  MAPE:                   {mape:.2f}%")
        print(f"  Max Error:              {max_error:.4f}")
        print(f"  Accuracy (±1 point):    {accuracy_1pt:.2f}%")
        print(f"  Accuracy (±2 points):   {accuracy_2pt:.2f}%")
        
        # Performance assessment
        if r2 >= 0.30:
            status = "✅ EXCELLENT"
        elif r2 >= 0.20:
            status = "✓ GOOD"
        elif r2 >= 0.10:
            status = "~ FAIR"
        else:
            status = "⚠️ NEEDS IMPROVEMENT"
        
        print(f"\n  Overall Performance:    {status}")
    
    return results


def evaluate_classification_model(predictor, X_test, y_test):
    """Evaluate exercise binary classification model"""
    
    print("\n" + "="*80)
    print("CLASSIFICATION MODEL EVALUATION")
    print("="*80)
    
    print(f"\n{'─'*80}")
    print("🏃 EXERCISE PROBABILITY PREDICTION")
    print(f"{'─'*80}")
    
    predictions = predictor.predict(X_test)
    y_true = y_test[:, 3]  # Exercise binary is 4th column
    y_pred_proba = predictions['exercise_binary']
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'specificity': specificity,
        'confusion_matrix': cm.tolist()
    }
    
    print(f"  Accuracy:               {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision:              {precision:.4f}")
    print(f"  Recall (Sensitivity):   {recall:.4f}")
    print(f"  Specificity:            {specificity:.4f}")
    print(f"  F1 Score:               {f1:.4f}")
    print(f"  AUC-ROC:                {auc_roc:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    No    Yes")
    print(f"  Actual    No    {tn:5d} {fp:5d}")
    print(f"            Yes   {fn:5d} {tp:5d}")
    
    # Performance assessment
    if accuracy >= 0.75:
        status = "✅ EXCELLENT"
    elif accuracy >= 0.65:
        status = "✓ GOOD"
    elif accuracy >= 0.55:
        status = "~ FAIR"
    else:
        status = "⚠️ NEEDS IMPROVEMENT"
    
    print(f"\n  Overall Performance:    {status}")
    
    return results


def evaluate_anomaly_detector(X_test, feature_names):
    """Evaluate anomaly detection model"""
    
    print("\n" + "="*80)
    print("ANOMALY DETECTION MODEL EVALUATION")
    print("="*80)
    
    detector = HealthAnomalyDetector(method='ensemble')
    
    try:
        detector.load_model(
            load_path="ml/trained_models/anomaly_detector_v2.pkl",
            input_dim=X_test.shape[1]
        )
        
        print(f"\n{'─'*80}")
        print("🔍 ANOMALY DETECTION")
        print(f"{'─'*80}")
        
        # Detect anomalies
        results = detector.detect_anomalies(X_test, return_details=True)
        
        num_anomalies = results['is_anomaly'].sum()
        anomaly_rate = (num_anomalies / len(X_test)) * 100
        
        print(f"  Test Samples:           {len(X_test):,}")
        print(f"  Anomalies Detected:     {num_anomalies:,} ({anomaly_rate:.2f}%)")
        print(f"  Average Anomaly Score:  {results['anomaly_scores'].mean():.4f}")
        
        # Severity distribution
        severity_counts = pd.Series(results['severity']).value_counts()
        print(f"\n  Severity Distribution:")
        for severity in ['normal', 'mild', 'moderate', 'high', 'critical']:
            count = severity_counts.get(severity, 0)
            pct = (count / len(X_test)) * 100
            print(f"    {severity.capitalize():10s}: {count:5d} ({pct:5.1f}%)")
        
        # Get explanations for top anomalies
        if num_anomalies > 0:
            top_indices = np.argsort(results['anomaly_scores'])[-min(3, num_anomalies):][::-1]
            explanations = detector.explain_anomaly(
                X_test[top_indices],
                feature_names,
                top_n=5
            )
            
            print(f"\n  Top Anomaly Contributors:")
            for exp in explanations[:1]:  # Show first example
                for feat_info in exp['top_features'][:3]:
                    print(f"    - {feat_info['feature']}: {feat_info['contribution_pct']:.1f}% contribution")
        
        evaluation = {
            'anomaly_rate': anomaly_rate,
            'avg_anomaly_score': float(results['anomaly_scores'].mean()),
            'severity_distribution': severity_counts.to_dict(),
            'status': '✅ OPERATIONAL'
        }
        
        return evaluation
        
    except Exception as e:
        print(f"  ⚠️ Error loading anomaly detector: {e}")
        return {'status': '❌ NOT LOADED', 'error': str(e)}


def evaluate_recommendation_engine(datasets):
    """Evaluate recommendation engine"""
    
    print("\n" + "="*80)
    print("RECOMMENDATION ENGINE EVALUATION")
    print("="*80)
    
    try:
        rec_engine = HealthRecommendationEngine()
        rec_engine.load_model()
        
        print(f"\n{'─'*80}")
        print("🎯 RECOMMENDATION SYSTEM")
        print(f"{'─'*80}")
        
        # Test recommendations for sample users
        sample_users = datasets['users'].sample(min(5, len(datasets['users'])))['user_id'].tolist()
        
        successful_recs = 0
        total_meal_recs = 0
        total_exercise_recs = 0
        total_wellness_recs = 0
        
        for user_id in sample_users:
            try:
                # Mock predictions for testing
                mock_predictions = {
                    'sleep_quality': [7.0],
                    'mood_score': [7.0],
                    'stress_level': [5.0],
                    'exercise_binary': [0.6]
                }
                
                # Mock user features
                mock_user_features = {
                    user_id: {
                        'age': 30,
                        'bmi': 22,
                        'activity_level': 'moderately_active',
                        'avg_daily_calories': 2000
                    }
                }
                
                recommendations = rec_engine.generate_personalized_recommendations(
                    user_id,
                    mock_user_features,
                    mock_predictions,
                    num_recommendations=3
                )
                
                successful_recs += 1
                total_meal_recs += len(recommendations['meals'])
                total_exercise_recs += len(recommendations['exercises'])
                total_wellness_recs += len(recommendations['wellness_tips'])
                
            except Exception as e:
                print(f"    Error for user {user_id}: {e}")
        
        success_rate = (successful_recs / len(sample_users)) * 100
        
        print(f"  Users Tested:           {len(sample_users)}")
        print(f"  Successful Recs:        {successful_recs} ({success_rate:.1f}%)")
        print(f"  Avg Meal Recs:          {total_meal_recs/successful_recs:.1f}" if successful_recs > 0 else "  Avg Meal Recs:          N/A")
        print(f"  Avg Exercise Recs:      {total_exercise_recs/successful_recs:.1f}" if successful_recs > 0 else "  Avg Exercise Recs:      N/A")
        print(f"  Avg Wellness Tips:      {total_wellness_recs/successful_recs:.1f}" if successful_recs > 0 else "  Avg Wellness Tips:      N/A")
        
        evaluation = {
            'success_rate': success_rate,
            'avg_recommendations': {
                'meals': total_meal_recs/successful_recs if successful_recs > 0 else 0,
                'exercises': total_exercise_recs/successful_recs if successful_recs > 0 else 0,
                'wellness_tips': total_wellness_recs/successful_recs if successful_recs > 0 else 0
            },
            'status': '✅ OPERATIONAL' if success_rate > 80 else '⚠️ PARTIAL'
        }
        
        return evaluation
        
    except Exception as e:
        print(f"  ⚠️ Error loading recommendation engine: {e}")
        return {'status': '❌ NOT LOADED', 'error': str(e)}


def generate_evaluation_report(all_results, save_path):
    """Generate comprehensive evaluation report"""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("HEALTH & WELLNESS AI SYSTEM - MODEL EVALUATION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n" + "="*80)
    report_lines.append("1. PREDICTION MODELS")
    report_lines.append("="*80)
    
    # Regression models
    for target, metrics in all_results['regression'].items():
        report_lines.append(f"\n{target.upper().replace('_', ' ')}:")
        report_lines.append(f"  R² Score:               {metrics['r2_score']:.4f}")
        report_lines.append(f"  MAE:                    {metrics['mae']:.4f}")
        report_lines.append(f"  RMSE:                   {metrics['rmse']:.4f}")
        report_lines.append(f"  Accuracy (±1 pt):       {metrics['accuracy_within_1pt']:.2f}%")
        report_lines.append(f"  Accuracy (±2 pts):      {metrics['accuracy_within_2pt']:.2f}%")
    
    # Classification model
    report_lines.append("\n" + "="*80)
    report_lines.append("2. EXERCISE PREDICTION MODEL")
    report_lines.append("="*80)
    cls_metrics = all_results['classification']
    report_lines.append(f"\n  Accuracy:               {cls_metrics['accuracy']:.4f} ({cls_metrics['accuracy']*100:.2f}%)")
    report_lines.append(f"  Precision:              {cls_metrics['precision']:.4f}")
    report_lines.append(f"  Recall:                 {cls_metrics['recall']:.4f}")
    report_lines.append(f"  F1 Score:               {cls_metrics['f1_score']:.4f}")
    report_lines.append(f"  AUC-ROC:                {cls_metrics['auc_roc']:.4f}")
    
    # Anomaly detection
    report_lines.append("\n" + "="*80)
    report_lines.append("3. ANOMALY DETECTION SYSTEM")
    report_lines.append("="*80)
    if 'error' not in all_results['anomaly']:
        anom_metrics = all_results['anomaly']
        report_lines.append(f"\n  Anomaly Rate:           {anom_metrics['anomaly_rate']:.2f}%")
        report_lines.append(f"  Avg Anomaly Score:      {anom_metrics['avg_anomaly_score']:.4f}")
        report_lines.append(f"  Status:                 {anom_metrics['status']}")
    else:
        report_lines.append(f"\n  Status:                 {all_results['anomaly']['status']}")
    
    # Recommendation engine
    report_lines.append("\n" + "="*80)
    report_lines.append("4. RECOMMENDATION ENGINE")
    report_lines.append("="*80)
    if 'error' not in all_results['recommendations']:
        rec_metrics = all_results['recommendations']
        report_lines.append(f"\n  Success Rate:           {rec_metrics['success_rate']:.1f}%")
        report_lines.append(f"  Avg Meal Recs:          {rec_metrics['avg_recommendations']['meals']:.1f}")
        report_lines.append(f"  Avg Exercise Recs:      {rec_metrics['avg_recommendations']['exercises']:.1f}")
        report_lines.append(f"  Avg Wellness Tips:      {rec_metrics['avg_recommendations']['wellness_tips']:.1f}")
        report_lines.append(f"  Status:                 {rec_metrics['status']}")
    else:
        report_lines.append(f"\n  Status:                 {all_results['recommendations']['status']}")
    
    # Overall assessment
    report_lines.append("\n" + "="*80)
    report_lines.append("OVERALL SYSTEM ASSESSMENT")
    report_lines.append("="*80)
    
    avg_r2 = np.mean([m['r2_score'] for m in all_results['regression'].values()])
    exercise_acc = all_results['classification']['accuracy']
    
    report_lines.append(f"\n  Average R² Score:       {avg_r2:.4f}")
    report_lines.append(f"  Exercise Accuracy:      {exercise_acc:.4f}")
    
    if avg_r2 >= 0.30 and exercise_acc >= 0.70:
        overall_status = "✅ PRODUCTION READY"
    elif avg_r2 >= 0.20 and exercise_acc >= 0.60:
        overall_status = "✓ GOOD - MINOR IMPROVEMENTS NEEDED"
    else:
        overall_status = "⚠️ NEEDS IMPROVEMENT"
    
    report_lines.append(f"\n  Overall Status:         {overall_status}")
    
    report_lines.append("\n" + "="*80)
    
    # Save report
    report_text = "\n".join(report_lines)
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n📄 Report saved to: {save_path}")


def main():
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    # Step 1: Load data
    print("\n📊 Step 1: Generating test data...")
    datasets = save_synthetic_data()
    
    # Step 2: Create features
    print("\n🔧 Step 2: Creating features...")
    ml_features = create_ml_features(datasets, sample_rate=0.5, lookback_days=28)
    
    X = ml_features['X_train']
    y = ml_features['y_train']
    feature_names = ml_features['feature_names']
    
    print(f"   Dataset shape: {X.shape}")
    
    # Step 3: Load and evaluate prediction models
    print("\n🤖 Step 3: Loading prediction models...")
    predictor = ImprovedHealthPredictor(model_type='stacking')
    predictor.load_models()
    
    # Use last 20% as test set
    test_size = int(len(X) * 0.2)
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    # Step 4: Evaluate all models
    all_results = {}
    
    all_results['regression'] = evaluate_regression_models(
        predictor, X_test, y_test, feature_names
    )
    
    all_results['classification'] = evaluate_classification_model(
        predictor, X_test, y_test
    )
    
    all_results['anomaly'] = evaluate_anomaly_detector(X_test, feature_names)
    
    all_results['recommendations'] = evaluate_recommendation_engine(datasets)
    
    # Step 5: Generate report
    print("\n📝 Step 5: Generating evaluation report...")
    report_dir = Path("ml/evaluation_results_v3")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f"model_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    generate_evaluation_report(all_results, report_path)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()