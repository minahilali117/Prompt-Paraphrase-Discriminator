import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sentence_transformers import SentenceTransformer
import argparse
import pickle
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class PromptPairDataset(Dataset):
    """Dataset class for prompt pairs."""
    
    def __init__(self, df, embedder, max_length=512):
        self.df = df
        self.embedder = embedder
        self.max_length = max_length
        
        # Pre-compute embeddings for efficiency
        print("Computing embeddings...")
        self.embeddings1 = self._get_embeddings(df['prompt1'].tolist())
        self.embeddings2 = self._get_embeddings(df['prompt2'].tolist())
        self.labels = torch.tensor(df['label'].values, dtype=torch.float32)
        
    def _get_embeddings(self, texts):
        """Get embeddings for a list of texts."""
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        emb1 = self.embeddings1[idx]
        emb2 = self.embeddings2[idx]
        label = self.labels[idx]
        
        # Create feature vector
        features = self._create_features(emb1, emb2)
        
        return features, label
    
    def _create_features(self, emb1, emb2):
        """Create feature vector from two embeddings."""
        # Cosine similarity
        cos_sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        
        # Element-wise operations
        abs_diff = torch.abs(emb1 - emb2)
        element_mult = emb1 * emb2
        
        # Concatenate all features
        features = torch.cat([
            emb1,
            emb2,
            abs_diff,
            element_mult,
            cos_sim
        ])
        
        return features

class PromptClassifierNN(nn.Module):
    """Neural network for prompt pair classification."""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super(PromptClassifierNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()

class PromptClassifier:
    """Main classifier class."""
    
    def __init__(self, model_type='labse', hidden_dims=[512, 256, 128], dropout=0.3):
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Initialize embedder
        # In your PromptClassifier.__init__ method, replace:
        if model_type == 'labse':
            self.embedder = SentenceTransformer('sentence-transformers/LaBSE')
        elif model_type == 'mt5':
            self.embedder = SentenceTransformer('sentence-transformers/mt5-base-encode')

        # With:
        if model_type == 'labse':
            # Try multiple alternatives
            models_to_try = [
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'all-MiniLM-L6-v2',
                'paraphrase-MiniLM-L6-v2'
            ]
        elif model_type == 'mt5':
            models_to_try = [
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'all-MiniLM-L6-v2'
            ]

        for model_name in models_to_try:
            try:
                self.embedder = SentenceTransformer(model_name)
                print(f"Successfully loaded {model_name}")
                break
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue
        else:
            raise Exception("Could not load any embedding model")
        
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Calculate input dimension for the neural network
        # emb1 + emb2 + abs_diff + element_mult + cos_sim
        self.input_dim = self.embed_dim * 4 + 1
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_model(self):
        """Create the neural network model."""
        model = PromptClassifierNN(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        )
        return model.to(self.device)
    
    def train(self, train_df, val_df=None, batch_size=32, epochs=50, 
                learning_rate=0.001, patience=10, save_path='models/'):
            """Train the classifier."""
            
            # Create datasets
            train_dataset = PromptPairDataset(train_df, self.embedder)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            if val_df is not None:
                val_dataset = PromptPairDataset(val_df, self.embedder)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Create model
            self.model = self._create_model()
            
            # Loss and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
            
            # Training history
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            print(f"Training on {self.device}")
            print(f"Model input dimension: {self.input_dim}")
            print(f"Embedding dimension: {self.embed_dim}")
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
                for batch_features, batch_labels in train_pbar:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    train_total += batch_labels.size(0)
                    train_correct += (predicted == batch_labels).sum().item()
                    
                    # Update progress bar
                    train_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{train_correct/train_total:.4f}'
                    })
                
                avg_train_loss = train_loss / len(train_loader)
                train_acc = train_correct / train_total
                
                history['train_loss'].append(avg_train_loss)
                history['train_acc'].append(train_acc)
                
                # Validation phase
                if val_df is not None:
                    self.model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                        for batch_features, batch_labels in val_pbar:
                            batch_features = batch_features.to(self.device)
                            batch_labels = batch_labels.to(self.device)
                            
                            outputs = self.model(batch_features)
                            loss = criterion(outputs, batch_labels)
                            
                            val_loss += loss.item()
                            predicted = (outputs > 0.5).float()
                            val_total += batch_labels.size(0)
                            val_correct += (predicted == batch_labels).sum().item()
                            
                            # Update progress bar
                            val_pbar.set_postfix({
                                'Loss': f'{loss.item():.4f}',
                                'Acc': f'{val_correct/val_total:.4f}'
                            })
                    
                    avg_val_loss = val_loss / len(val_loader)
                    val_acc = val_correct / val_total
                    
                    history['val_loss'].append(avg_val_loss)
                    history['val_acc'].append(val_acc)
                    
                    # Learning rate scheduling
                    scheduler.step(avg_val_loss)
                    
                    print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                        f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                        f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                    
                    # Early stopping
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        # Save best model
                        self._save_model(save_path, 'best_classifier.pkl')
                        print(f"New best model saved! Val Loss: {avg_val_loss:.4f}")
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break
                else:
                    print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
            
            # Save final model
            self._save_model(save_path, 'final_classifier.pkl')
            print("Training completed!")
            
            return history
    
    def _save_model(self, save_path, filename):
            """Save the trained model."""
            os.makedirs(save_path, exist_ok=True)
            model_path = os.path.join(save_path, filename)
            
            # Save model state and metadata
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'embed_dim': self.embed_dim,
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'dropout': self.dropout
            }, model_path)
            
            print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
            """Load a trained model."""
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Restore model parameters
            self.model_type = checkpoint.get('model_type', self.model_type)
            self.embed_dim = checkpoint.get('embed_dim', self.embed_dim)
            self.input_dim = checkpoint.get('input_dim', self.input_dim)
            self.hidden_dims = checkpoint.get('hidden_dims', self.hidden_dims)
            self.dropout = checkpoint.get('dropout', self.dropout)
            
            # Create and load model
            self.model = self._create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"Model loaded from {model_path}")
        
    def predict(self, df, batch_size=32):
            """Make predictions on a dataset."""
            if self.model is None:
                raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
            
            dataset = PromptPairDataset(df, self.embedder)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            self.model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                for batch_features, _ in tqdm(dataloader, desc='Predicting'):
                    batch_features = batch_features.to(self.device)
                    outputs = self.model(batch_features)
                    
                    probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
                    preds = (probs > 0.5).astype(int)

                    # Defensive: make sure these are always iterable
                    probabilities.extend(probs.tolist())
                    predictions.extend(preds.tolist())

            return np.array(predictions), np.array(probabilities)
        
    def predict_pair(self, prompt1, prompt2):
            """Make prediction on a single prompt pair."""
            # Create temporary dataframe
            temp_df = pd.DataFrame({
                'prompt1': [prompt1],
                'prompt2': [prompt2],
                'label': [0]  # Dummy label
            })
            
            predictions, probabilities = self.predict(temp_df, batch_size=1)
            return predictions[0], probabilities[0]
        
    def evaluate(self, test_df, batch_size=32):
            """Evaluate the model on test data."""
            predictions, probabilities = self.predict(test_df, batch_size)
            y_true = test_df['label'].values
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average='binary')
            roc_auc = roc_auc_score(y_true, probabilities)
            
            # Calculate confusion matrix components
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
            }
            
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(f"Accuracy:    {accuracy:.4f}")
            print(f"Precision:   {precision:.4f}")
            print(f"Recall:      {recall:.4f}")
            print(f"F1-Score:    {f1:.4f}")
            print(f"ROC-AUC:     {roc_auc:.4f}")
            print(f"Specificity: {metrics['specificity']:.4f}")
            print("\nConfusion Matrix:")
            print(f"True Positives:  {tp}")
            print(f"True Negatives:  {tn}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            print("="*50)
            
            return metrics, predictions, probabilities
        
    def plot_training_history(self, history, save_path=None):
            """Plot training history."""
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot loss
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
            if 'val_loss' in history and history['val_loss']:
                ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # Plot accuracy
            ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
            if 'val_acc' in history and history['val_acc']:
                ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Training history plot saved to {save_path}")
            
            plt.show()
        
    def analyze_errors(self, test_df, predictions, probabilities, save_path=None):
            """Analyze prediction errors."""
            y_true = test_df['label'].values
            
            # Find errors
            errors = predictions != y_true
            error_indices = np.where(errors)[0]
            
            print(f"\nError Analysis: {len(error_indices)} errors out of {len(test_df)} samples")
            print(f"Error rate: {len(error_indices)/len(test_df):.4f}")
            
            if len(error_indices) > 0:
                # Analyze false positives and false negatives
                false_positives = test_df.iloc[error_indices][(predictions[error_indices] == 1) & (y_true[error_indices] == 0)]
                false_negatives = test_df.iloc[error_indices][(predictions[error_indices] == 0) & (y_true[error_indices] == 1)]
                
                print(f"\nFalse Positives: {len(false_positives)}")
                print(f"False Negatives: {len(false_negatives)}")
                
                # Show some examples
                if len(false_positives) > 0:
                    print("\nSample False Positives (predicted same, actually different):")
                    for i, (_, row) in enumerate(false_positives.head(3).iterrows()):
                        prob = probabilities[error_indices][predictions[error_indices] == 1][i] if i < len(probabilities[error_indices][predictions[error_indices] == 1]) else 0.0
                        print(f"  Confidence: {prob:.3f}")
                        print(f"  Prompt 1: {row['prompt1']}")
                        print(f"  Prompt 2: {row['prompt2']}")
                        print()
                
                if len(false_negatives) > 0:
                    print("Sample False Negatives (predicted different, actually same):")
                    for i, (_, row) in enumerate(false_negatives.head(3).iterrows()):
                        prob = probabilities[error_indices][predictions[error_indices] == 0][i] if i < len(probabilities[error_indices][predictions[error_indices] == 0]) else 0.0
                        print(f"  Confidence: {1-prob:.3f}")
                        print(f"  Prompt 1: {row['prompt1']}")
                        print(f"  Prompt 2: {row['prompt2']}")
                        print()
            
            return error_indices

def main():
    parser = argparse.ArgumentParser(description='Train prompt paraphrase classifier')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data CSV')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data CSV')
    parser.add_argument('--model_type', type=str, default='labse', 
                       choices=['labse', 'mt5'],
                       help='Embedding model type')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--save_dir', type=str, default='models/',
                       help='Directory to save models')
    parser.add_argument('--plot_history', action='store_true',
                       help='Plot training history')
    parser.add_argument('--analyze_errors', action='store_true',
                       help='Analyze prediction errors')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 256, 128],
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Training positive ratio: {train_df['label'].mean():.3f}")
    print(f"Test positive ratio: {test_df['label'].mean():.3f}")
    
    # Split training data for validation
    if args.val_split > 0:
        train_df, val_df = train_test_split(
            train_df, test_size=args.val_split, 
            random_state=42, stratify=train_df['label']
        )
        print(f"Training samples after split: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
    else:
        val_df = None
    
    # Initialize classifier
    print(f"Initializing classifier with {args.model_type} embeddings...")
    classifier = PromptClassifier(
        model_type=args.model_type,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    )
    
    # Train the model
    print("Starting training...")
    history = classifier.train(
        train_df=train_df,
        val_df=val_df,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        save_path=args.save_dir
    )
    
    # Plot training history
    if args.plot_history:
        plot_path = os.path.join(args.save_dir, 'training_history.png')
        classifier.plot_training_history(history, plot_path)
    
    # Load best model for evaluation
    best_model_path = os.path.join(args.save_dir, 'best_classifier.pkl')
    if os.path.exists(best_model_path):
        classifier.load_model(best_model_path)
        print("Loaded best model for evaluation")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics, predictions, probabilities = classifier.evaluate(test_df)
    
    # Analyze errors
    if args.analyze_errors:
        error_indices = classifier.analyze_errors(test_df, predictions, probabilities)
    
    # Save evaluation results
    results_path = os.path.join(args.save_dir, 'evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'metrics': metrics,
            'predictions': predictions,
            'probabilities': probabilities,
            'test_labels': test_df['label'].values,
            'history': history
        }, f)
    
    print(f"\nEvaluation results saved to {results_path}")
    
    # Create comprehensive evaluation report
    report_path = os.path.join(args.save_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("PROMPT PARAPHRASE CLASSIFIER EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model Configuration:\n")
        f.write(f"  Model Type: {args.model_type}\n")
        f.write(f"  Hidden Dimensions: {args.hidden_dims}\n")
        f.write(f"  Dropout Rate: {args.dropout}\n")
        f.write(f"  Learning Rate: {args.learning_rate}\n")
        f.write(f"  Batch Size: {args.batch_size}\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"  Training Samples: {len(train_df)}\n")
        f.write(f"  Validation Samples: {len(val_df) if val_df is not None else 0}\n")
        f.write(f"  Test Samples: {len(test_df)}\n")
        f.write(f"  Training Positive Ratio: {train_df['label'].mean():.3f}\n")
        f.write(f"  Test Positive Ratio: {test_df['label'].mean():.3f}\n\n")
        
        f.write(f"Performance Metrics:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score: {metrics['f1']:.4f}\n")
        f.write(f"  ROC-AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"  Specificity: {metrics['specificity']:.4f}\n\n")
        
        f.write(f"Confusion Matrix:\n")
        f.write(f"  True Positives: {metrics['true_positives']}\n")
        f.write(f"  True Negatives: {metrics['true_negatives']}\n")
        f.write(f"  False Positives: {metrics['false_positives']}\n")
        f.write(f"  False Negatives: {metrics['false_negatives']}\n")
    
    print(f"Evaluation report saved to {report_path}")
    
    # Example usage
    print("\nExample predictions:")
    test_pairs = [
        ("a red bicycle under a tree", "a crimson bike beneath a tree"),
        ("a dog running in the park", "a cat sleeping on the couch"),
        ("beautiful sunset over the ocean", "hermoso atardecer sobre el océano")
    ]
    
    for prompt1, prompt2 in test_pairs:
        pred, prob = classifier.predict_pair(prompt1, prompt2)
        print(f"Pair: '{prompt1}' vs '{prompt2}'")
        print(f"Prediction: {'Same scene' if pred == 1 else 'Different scenes'} (confidence: {prob:.3f})")
        print()

if __name__ == "__main__":
    main()
