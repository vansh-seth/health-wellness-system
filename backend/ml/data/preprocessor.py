# ml/data/preprocessor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import joblib

class HealthDataPreprocessor:
    """Preprocess health data for ML models"""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_stats = {}
        self.is_fitted = False
        
    def fit(self, datasets: Dict[str, pd.DataFrame]):
        """
        Fit preprocessing transformers on training data
        
        Args:
            datasets: Dictionary of dataframes for each data type
        """
        
        print("Fitting preprocessors on training data...")
        
        for data_type, df in datasets.items():
            if len(df) == 0:
                continue
            
            # Store feature statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                self.feature_stats[data_type] = {
                    'numeric_cols': numeric_cols,
                    'means': df[numeric_cols].mean().to_dict(),
                    'stds': df[numeric_cols].std().to_dict(),
                    'mins': df[numeric_cols].min().to_dict(),
                    'maxs': df[numeric_cols].max().to_dict()
                }
            
            # Fit scalers for numeric features
            if numeric_cols:
                self.scalers[data_type] = StandardScaler()
                self.scalers[data_type].fit(df[numeric_cols])
            
            # Fit imputers
            self.imputers[data_type] = KNNImputer(n_neighbors=5)
            if numeric_cols:
                self.imputers[data_type].fit(df[numeric_cols])
            
            # Fit label encoders for categorical features
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols:
                if col not in ['user_id', 'date', 'timestamp']:
                    self.encoders[f"{data_type}_{col}"] = LabelEncoder()
                    self.encoders[f"{data_type}_{col}"].fit(df[col].astype(str))
        
        self.is_fitted = True
        print("Preprocessing fit complete!")
    
    def transform(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Transform data using fitted preprocessors
        
        Args:
            datasets: Dictionary of dataframes
            
        Returns:
            Transformed datasets
        """
        
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        transformed_datasets = {}
        
        for data_type, df in datasets.items():
            if len(df) == 0:
                transformed_datasets[data_type] = df
                continue
            
            df_transformed = df.copy()
            
            # Handle missing values
            numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols and data_type in self.imputers:
                df_transformed[numeric_cols] = self.imputers[data_type].transform(df_transformed[numeric_cols])
            
            # Encode categorical variables
            categorical_cols = df_transformed.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols:
                if col not in ['user_id', 'date', 'timestamp']:
                    encoder_key = f"{data_type}_{col}"
                    if encoder_key in self.encoders:
                        df_transformed[col] = self.encoders[encoder_key].transform(df_transformed[col].astype(str))
            
            # Scale numeric features
            if numeric_cols and data_type in self.scalers:
                df_transformed[numeric_cols] = self.scalers[data_type].transform(df_transformed[numeric_cols])
            
            transformed_datasets[data_type] = df_transformed
        
        return transformed_datasets
    
    def fit_transform(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Fit and transform in one step"""
        self.fit(datasets)
        return self.transform(datasets)
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in numeric columns
        
        Args:
            df: DataFrame
            columns: Columns to check for outliers
            method: 'iqr', 'zscore', or 'clip'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        
        df_clean = df.copy()
        
        for col in columns:
            if col not in df_clean.columns:
                continue
            
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Clip outliers
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            
            elif method == 'zscore':
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            
            elif method == 'clip':
                # Clip based on percentiles
                lower_percentile = df_clean[col].quantile(0.01)
                upper_percentile = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(lower_percentile, upper_percentile)
        
        return df_clean
    
    def validate_data_quality(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Validate data quality and return quality metrics
        
        Args:
            datasets: Dictionary of dataframes
            
        Returns:
            Quality metrics for each dataset
        """
        
        quality_metrics = {}
        
        for data_type, df in datasets.items():
            if len(df) == 0:
                quality_metrics[data_type] = {'status': 'empty'}
                continue
            
            metrics = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns)
            }
            
            # Check for columns with too many missing values
            high_missing_cols = [
                col for col, pct in metrics['missing_percentage'].items() if pct > 50
            ]
            metrics['high_missing_columns'] = high_missing_cols
            
            # Calculate completeness score
            avg_missing_pct = sum(metrics['missing_percentage'].values()) / len(metrics['missing_percentage'])
            metrics['completeness_score'] = 100 - avg_missing_pct
            
            quality_metrics[data_type] = metrics
        
        return quality_metrics
    
    def normalize_timestamps(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Normalize timestamps to a standard format"""
        
        df_normalized = df.copy()
        
        if timestamp_col in df_normalized.columns:
            df_normalized[timestamp_col] = pd.to_datetime(df_normalized[timestamp_col])
            
            # Extract useful temporal features
            df_normalized[f'{timestamp_col}_hour'] = df_normalized[timestamp_col].dt.hour
            df_normalized[f'{timestamp_col}_day_of_week'] = df_normalized[timestamp_col].dt.dayofweek
            df_normalized[f'{timestamp_col}_is_weekend'] = (df_normalized[timestamp_col].dt.dayofweek >= 5).astype(int)
        
        return df_normalized
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray, method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance dataset for classification tasks
        
        Args:
            X: Features
            y: Labels
            method: 'oversample', 'undersample', or 'smote'
            
        Returns:
            Balanced X and y
        """
        
        try:
            from imblearn.over_sampling import SMOTE, RandomOverSampler
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError:
            print("imbalanced-learn not installed. Skipping balancing.")
            return X, y
        
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'oversample':
            sampler = RandomOverSampler(random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        else:
            return X, y
        
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        print(f"Dataset balanced using {method}")
        print(f"Original shape: {X.shape}, Balanced shape: {X_balanced.shape}")
        
        return X_balanced, y_balanced
    
    def save(self, filepath: str = "ml/trained_models/preprocessor.pkl"):
        """Save preprocessor"""
        
        preprocessor_data = {
            'scalers': self.scalers,
            'imputers': self.imputers,
            'encoders': self.encoders,
            'feature_stats': self.feature_stats,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath: str = "ml/trained_models/preprocessor.pkl"):
        """Load preprocessor"""
        
        preprocessor_data = joblib.load(filepath)
        
        self.scalers = preprocessor_data['scalers']
        self.imputers = preprocessor_data['imputers']
        self.encoders = preprocessor_data['encoders']
        self.feature_stats = preprocessor_data['feature_stats']
        self.is_fitted = preprocessor_data['is_fitted']
        
        print(f"Preprocessor loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from ml.data.synthetic_generator import save_synthetic_data
    
    # Generate synthetic data
    print("Generating synthetic data...")
    datasets = save_synthetic_data()
    
    # Initialize preprocessor
    preprocessor = HealthDataPreprocessor()
    
    # Validate data quality
    print("\nValidating data quality...")
    quality_metrics = preprocessor.validate_data_quality(datasets)
    
    for data_type, metrics in quality_metrics.items():
        if 'completeness_score' in metrics:
            print(f"\n{data_type}:")
            print(f"  Total rows: {metrics['total_rows']}")
            print(f"  Completeness: {metrics['completeness_score']:.2f}%")
            print(f"  Duplicate rows: {metrics['duplicate_rows']}")
    
    # Fit and transform
    print("\nFitting and transforming data...")
    transformed_datasets = preprocessor.fit_transform(datasets)
    
    print("\nTransformed dataset shapes:")
    for data_type, df in transformed_datasets.items():
        print(f"{data_type}: {df.shape}")
    
    # Save preprocessor
    preprocessor.save()