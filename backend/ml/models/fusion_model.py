# ml/models/fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import pandas as pd

class AttentionFusion(nn.Module):
    """Attention-based fusion of multiple health data modalities"""
    
    def __init__(self, modality_dims: Dict[str, int], hidden_dim: int = 128):
        super(AttentionFusion, self).__init__()
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        
        # Individual modality encoders
        self.encoders = nn.ModuleDict()
        self.attention_weights = nn.ModuleDict()
        
        for modality, dim in modality_dims.items():
            # Encoder for each modality
            self.encoders[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # Attention mechanism for each modality
            self.attention_weights[modality] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Final fusion layer
        total_dim = len(modality_dims) * hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        encoded_modalities = {}
        attention_scores = {}
        
        # Encode each modality
        for modality, input_tensor in modality_inputs.items():
            if modality in self.encoders:
                encoded = self.encoders[modality](input_tensor)
                attention_score = self.attention_weights[modality](encoded)
                
                # Apply attention
                attended = encoded * torch.sigmoid(attention_score)
                
                encoded_modalities[modality] = attended
                attention_scores[modality] = attention_score
        
        # Concatenate all modalities
        if encoded_modalities:
            fused = torch.cat(list(encoded_modalities.values()), dim=1)
            output = self.fusion_layer(fused)
            return output, attention_scores
        else:
            raise ValueError("No valid modalities provided")

class HealthLSTM(nn.Module):
    """LSTM model for temporal health data"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, output_dim: int = 4):
        super(HealthLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Self-attention mechanism
        lstm_out_transposed = lstm_out.transpose(0, 1)  # (seq_len, batch, hidden_dim)
        attn_out, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, hidden_dim)
        
        # Use the last output or mean pooling
        if seq_lengths is not None:
            # Variable length sequences
            batch_indices = torch.arange(batch_size)
            last_outputs = attn_out[batch_indices, seq_lengths - 1]
        else:
            # Fixed length sequences - use last output
            last_outputs = attn_out[:, -1, :]
        
        output = self.output_layer(last_outputs)
        return output

class MultiModalHealthPredictor(nn.Module):
    """Complete multi-modal health prediction model"""
    
    def __init__(self, modality_config: Dict[str, Dict], target_dims: Dict[str, int]):
        super(MultiModalHealthPredictor, self).__init__()
        self.modality_config = modality_config
        self.target_dims = target_dims
        
        # Attention fusion layer
        fusion_dims = {mod: config['dim'] for mod, config in modality_config.items()}
        self.fusion = AttentionFusion(fusion_dims, hidden_dim=256)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        fusion_output_dim = 128  # From fusion layer
        
        for task, dim in target_dims.items():
            self.task_heads[task] = nn.Sequential(
                nn.Linear(fusion_output_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, dim)
            )
    
    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Fuse modalities
        fused_features, attention_scores = self.fusion(modality_inputs)
        
        # Generate predictions for each task
        predictions = {}
        for task in self.target_dims:
            predictions[task] = self.task_heads[task](fused_features)
        
        return predictions, attention_scores

class HealthDataFusion:
    """Main class for multi-modal health data fusion and prediction"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        
    def prepare_modality_data(self, features_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare data for different modalities"""
        # First, ensure all data is numeric
        df_numeric = features_df.copy()
        
        # Identify and encode categorical columns
        categorical_cols = df_numeric.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in ['user_id', 'date', 'target_date']:
                # Convert categorical to numeric codes
                if df_numeric[col].dtype == 'object' or df_numeric[col].dtype.name == 'category':
                    df_numeric[col] = pd.Categorical(df_numeric[col]).codes
        
        # Drop non-feature columns
        columns_to_drop = ['user_id', 'date', 'target_date', 
                        'target_sleep_quality', 'target_mood_score', 
                        'target_stress_level', 'target_will_exercise']
        df_numeric = df_numeric.drop(columns=[col for col in columns_to_drop if col in df_numeric.columns])
        
        # Ensure all remaining columns are numeric
        df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
        
        # Fill any NaN values with 0
        df_numeric = df_numeric.fillna(0)
        
        modalities = {}
        
        # Nutrition modality
        nutrition_cols = [col for col in df_numeric.columns if any(
            nutrient in col.lower() for nutrient in ['calories', 'protein', 'carbs', 'fat', 'meal']
        ) and 'exercise' not in col.lower()]
        if nutrition_cols:
            modalities['nutrition'] = df_numeric[nutrition_cols].values.astype(np.float32)
        
        # Exercise modality
        exercise_cols = [col for col in df_numeric.columns if any(
            term in col.lower() for term in ['exercise', 'duration', 'calories_burned']
        )]
        if exercise_cols:
            modalities['exercise'] = df_numeric[exercise_cols].values.astype(np.float32)
        
        # Sleep modality
        sleep_cols = [col for col in df_numeric.columns if any(
            term in col.lower() for term in ['sleep', 'quality']
        ) and 'duration' in col.lower()]
        if sleep_cols:
            modalities['sleep'] = df_numeric[sleep_cols].values.astype(np.float32)
        
        # Mood/Mental health modality
        mood_cols = [col for col in df_numeric.columns if any(
            term in col.lower() for term in ['mood', 'stress', 'energy']
        )]
        if mood_cols:
            modalities['mood'] = df_numeric[mood_cols].values.astype(np.float32)
        
        # Demographics modality
        demo_cols = [col for col in df_numeric.columns if any(
            term in col.lower() for term in ['age', 'gender', 'bmi', 'height', 'weight', 'activity']
        )]
        if demo_cols:
            modalities['demographics'] = df_numeric[demo_cols].values.astype(np.float32)
        
        # Validate that all modalities have data
        modalities = {k: v for k, v in modalities.items() if v.shape[1] > 0}
        
        return modalities
        
    def create_temporal_sequences(self, features_df: pd.DataFrame, sequence_length: int = 7) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Create temporal sequences for LSTM training"""
        sequences = {modality: [] for modality in ['nutrition', 'exercise', 'sleep', 'mood', 'demographics']}
        targets = []
        
        users = features_df['user_id'].unique()
        
        for user_id in users:
            user_data = features_df[features_df['user_id'] == user_id].sort_values('date')
            
            if len(user_data) < sequence_length + 1:
                continue
            
            modality_data = self.prepare_modality_data(user_data)
            targets_data = user_data[['target_sleep_quality', 'target_mood_score', 'target_stress_level', 'target_will_exercise']].values
            
            # Create sequences
            for i in range(len(user_data) - sequence_length):
                for modality, data in modality_data.items():
                    if len(data) > i + sequence_length:
                        seq = data[i:i + sequence_length]
                        sequences[modality].append(seq)
                
                if len(targets_data) > i + sequence_length:
                    targets.append(targets_data[i + sequence_length])
        
        # Convert to tensors
        sequence_tensors = {}
        min_length = min(len(seq_list) for seq_list in sequences.values() if seq_list)
        
        for modality, seq_list in sequences.items():
            if seq_list and len(seq_list) >= min_length:
                sequence_tensors[modality] = torch.FloatTensor(np.array(seq_list[:min_length]))
        
        targets_tensor = torch.FloatTensor(np.array(targets[:min_length]))
        
        return sequence_tensors, targets_tensor
    
    def train_fusion_model(self, features_df: pd.DataFrame, epochs: int = 100, batch_size: int = 32):
        """Train the multi-modal fusion model with memory optimization"""
        
        # Prepare modality data
        modality_data = self.prepare_modality_data(features_df)
        
        # Define modality configuration
        modality_config = {}
        for modality, data in modality_data.items():
            modality_config[modality] = {'dim': data.shape[1]}
        
        # Target dimensions
        target_dims = {
            'sleep_quality': 1,
            'mood_score': 1,
            'stress_level': 1,
            'exercise_binary': 1
        }
        
        # Create model
        model = MultiModalHealthPredictor(modality_config, target_dims).to(self.device)
        
        # Prepare targets - keep on CPU
        targets = features_df[['target_sleep_quality', 'target_mood_score', 'target_stress_level', 'target_will_exercise']].values.astype(np.float32)
        
        # Split data first
        indices = np.arange(len(targets))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        # Optimizer and loss functions
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle training indices
            np.random.shuffle(train_idx)
            
            # Mini-batch training - only load batch to GPU
            for i in range(0, len(train_idx), batch_size):
                batch_indices = train_idx[i:i+batch_size]
                
                # Prepare batch data - move to GPU only when needed
                batch_modalities = {}
                batch_targets = {}
                
                for mod, data in modality_data.items():
                    batch_modalities[mod] = torch.FloatTensor(data[batch_indices]).to(self.device)
                
                batch_targets['sleep_quality'] = torch.FloatTensor(targets[batch_indices, 0]).unsqueeze(1).to(self.device)
                batch_targets['mood_score'] = torch.FloatTensor(targets[batch_indices, 1]).unsqueeze(1).to(self.device)
                batch_targets['stress_level'] = torch.FloatTensor(targets[batch_indices, 2]).unsqueeze(1).to(self.device)
                batch_targets['exercise_binary'] = torch.FloatTensor(targets[batch_indices, 3]).unsqueeze(1).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                predictions, attention_scores = model(batch_modalities)
                
                # Calculate losses
                total_loss = 0
                
                # Regression tasks
                for task in ['sleep_quality', 'mood_score', 'stress_level']:
                    loss = mse_loss(predictions[task], batch_targets[task])
                    total_loss += loss
                
                # Binary classification task
                loss = bce_loss(predictions['exercise_binary'], batch_targets['exercise_binary'])
                total_loss += loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Clear GPU cache periodically
                if num_batches % 10 == 0:
                    del batch_modalities, batch_targets, predictions, attention_scores
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Validation - also in batches to save memory
            if epoch % 10 == 0:
                model.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for i in range(0, len(val_idx), batch_size):
                        val_batch_idx = val_idx[i:i+batch_size]
                        
                        val_modalities = {}
                        val_targets = {}
                        
                        for mod, data in modality_data.items():
                            val_modalities[mod] = torch.FloatTensor(data[val_batch_idx]).to(self.device)
                        
                        val_targets['sleep_quality'] = torch.FloatTensor(targets[val_batch_idx, 0]).unsqueeze(1).to(self.device)
                        val_targets['mood_score'] = torch.FloatTensor(targets[val_batch_idx, 1]).unsqueeze(1).to(self.device)
                        val_targets['stress_level'] = torch.FloatTensor(targets[val_batch_idx, 2]).unsqueeze(1).to(self.device)
                        val_targets['exercise_binary'] = torch.FloatTensor(targets[val_batch_idx, 3]).unsqueeze(1).to(self.device)
                        
                        val_preds, _ = model(val_modalities)
                        
                        batch_loss = 0
                        for task in ['sleep_quality', 'mood_score', 'stress_level']:
                            batch_loss += mse_loss(val_preds[task], val_targets[task]).item()
                        batch_loss += bce_loss(val_preds['exercise_binary'], val_targets['exercise_binary']).item()
                        
                        val_loss += batch_loss
                        val_batches += 1
                        
                        # Clear memory
                        del val_modalities, val_targets, val_preds
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
                avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
                
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                scheduler.step(avg_val_loss)
                model.train()
                
                # Clear cache after validation
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.models['fusion'] = model
        return model
    
    def predict(self, features_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions using the trained fusion model"""
        if 'fusion' not in self.models:
            raise ValueError("Model not trained yet. Call train_fusion_model first.")
        
        model = self.models['fusion']
        model.eval()
        
        # Prepare data
        modality_data = self.prepare_modality_data(features_df)
        modality_tensors = {mod: torch.FloatTensor(data).to(self.device) for mod, data in modality_data.items()}
        
        with torch.no_grad():
            predictions, attention_scores = model(modality_tensors)
        
        # Convert predictions to numpy
        results = {}
        for task, pred in predictions.items():
            if task == 'exercise_binary':
                results[task] = torch.sigmoid(pred).cpu().numpy()
            else:
                results[task] = pred.cpu().numpy()
        
        results['attention_scores'] = {mod: scores.cpu().numpy() for mod, scores in attention_scores.items()}
        
        return results

# Example usage and testing
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data.synthetic_generator import save_synthetic_data
    from data.feature_engineering import create_ml_features
    
    # Generate synthetic data
    print("Generating synthetic data...")
    datasets = save_synthetic_data()
    
    # Create ML features
    print("Creating ML features...")
    ml_features = create_ml_features(datasets)
    
    # Initialize fusion model
    print("Training fusion model...")
    fusion = HealthDataFusion()
    
    # Train model
    model = fusion.train_fusion_model(ml_features['final_features'], epochs=25)
    
    # Make predictions
    print("Making predictions...")
    predictions = fusion.predict(ml_features['final_features'].head(100))
    
    print("Prediction shapes:")
    for task, preds in predictions.items():
        if task != 'attention_scores':
            print(f"{task}: {preds.shape}")
    
    print("Sample attention scores:")
    for modality, scores in predictions['attention_scores'].items():
        print(f"{modality}: mean attention = {scores.mean():.4f}")
        
    print("Multi-modal fusion training complete!")