import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold
import argparse
import pickle
import os
import json
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import the classifier from your main module
from train_classifier import PromptClassifier  # Assuming the main code is in paste.py

class ModelEvaluator:
    """Comprehensive evaluation suite for the prompt paraphrase classifier."""
    
    def __init__(self, model_path=None, classifier=None):
        """Initialize evaluator with either a model path or classifier instance."""
        if classifier is not None:
            self.classifier = classifier
        elif model_path is not None:
            self.classifier = PromptClassifier()
            self.classifier.load_model(model_path)
        else:
            raise ValueError("Either model_path or classifier must be provided")
        
        self.results = {}
        
    def evaluate_basic_metrics(self, test_df, save_path=None):
        """Compute basic classification metrics."""
        print("Computing basic metrics...")
        
        # Get predictions
        predictions, probabilities = self.classifier.predict(test_df)
        y_true = test_df['label'].values
        
        # Basic metrics
        accuracy = accuracy_score(y_true, predictions)
        balanced_acc = balanced_accuracy_score(y_true, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average='binary')
        roc_auc = roc_auc_score(y_true, probabilities)
        avg_precision = average_precision_score(y_true, probabilities)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # False positive and negative rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(y_true),
            'positive_samples': int(y_true.sum()),
            'negative_samples': int(len(y_true) - y_true.sum())
        }
        
        self.results['basic_metrics'] = metrics
        self.results['predictions'] = predictions
        self.results['probabilities'] = probabilities
        self.results['true_labels'] = y_true
        
        # Print results
        print("\n" + "="*60)
        print("BASIC EVALUATION METRICS")
        print("="*60)
        print(f"Dataset Size: {len(y_true)} samples")
        print(f"Positive Samples: {int(y_true.sum())} ({y_true.mean():.1%})")
        print(f"Negative Samples: {int(len(y_true) - y_true.sum())} ({1-y_true.mean():.1%})")
        print("-"*40)
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Balanced Accuracy:  {balanced_acc:.4f}")
        print(f"Precision:          {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity:        {specificity:.4f}")
        print(f"F1-Score:           {f1:.4f}")
        print(f"ROC-AUC:            {roc_auc:.4f}")
        print(f"Average Precision:  {avg_precision:.4f}")
        print("-"*40)
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
        print("-"*40)
        print("Confusion Matrix:")
        print(f"  True Positives:   {tp}")
        print(f"  True Negatives:   {tn}")
        print(f"  False Positives:  {fp}")
        print(f"  False Negatives:  {fn}")
        print("="*60)
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Basic metrics saved to {save_path}")
        
        return metrics
    
    def evaluate_threshold_analysis(self, save_dir=None):
        """Analyze performance across different classification thresholds."""
        print("Performing threshold analysis...")
        
        if 'probabilities' not in self.results:
            raise ValueError("Must run evaluate_basic_metrics first")
        
        probabilities = self.results['probabilities']
        y_true = self.results['true_labels']
        
        # Test different thresholds
        thresholds = np.arange(0.1, 1.0, 0.05)
        threshold_results = []
        
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average='binary', zero_division=0)
            
            # Confusion matrix components
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            except ValueError:
                # Handle case where confusion matrix doesn't have 4 values
                specificity = 0.0
                fpr = 0.0
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'fpr': fpr
            })
        
        threshold_df = pd.DataFrame(threshold_results)
        self.results['threshold_analysis'] = threshold_df
        
        # Find optimal thresholds
        optimal_f1_idx = threshold_df['f1_score'].idxmax()
        optimal_accuracy_idx = threshold_df['accuracy'].idxmax()
        
        print(f"\nOptimal threshold for F1-score: {threshold_df.loc[optimal_f1_idx, 'threshold']:.3f} "
              f"(F1: {threshold_df.loc[optimal_f1_idx, 'f1_score']:.4f})")
        print(f"Optimal threshold for accuracy: {threshold_df.loc[optimal_accuracy_idx, 'threshold']:.3f} "
              f"(Accuracy: {threshold_df.loc[optimal_accuracy_idx, 'accuracy']:.4f})")
        
        # Plot threshold analysis
        if save_dir:
            self._plot_threshold_analysis(threshold_df, save_dir)
        
        return threshold_df
    
    def _plot_threshold_analysis(self, threshold_df, save_dir):
        """Plot threshold analysis results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Accuracy and F1-score vs threshold
        ax1.plot(threshold_df['threshold'], threshold_df['accuracy'], 'b-', label='Accuracy', linewidth=2)
        ax1.plot(threshold_df['threshold'], threshold_df['f1_score'], 'r-', label='F1-Score', linewidth=2)
        ax1.set_xlabel('Classification Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Accuracy and F1-Score vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Precision and Recall vs threshold
        ax2.plot(threshold_df['threshold'], threshold_df['precision'], 'g-', label='Precision', linewidth=2)
        ax2.plot(threshold_df['threshold'], threshold_df['recall'], 'orange', label='Recall', linewidth=2)
        ax2.set_xlabel('Classification Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision and Recall vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Specificity and FPR vs threshold
        ax3.plot(threshold_df['threshold'], threshold_df['specificity'], 'purple', label='Specificity', linewidth=2)
        ax3.plot(threshold_df['threshold'], threshold_df['fpr'], 'brown', label='False Positive Rate', linewidth=2)
        ax3.set_xlabel('Classification Threshold')
        ax3.set_ylabel('Score')
        ax3.set_title('Specificity and FPR vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Combined view of key metrics
        ax4.plot(threshold_df['threshold'], threshold_df['accuracy'], 'b-', label='Accuracy', linewidth=2)
        ax4.plot(threshold_df['threshold'], threshold_df['f1_score'], 'r-', label='F1-Score', linewidth=2)
        ax4.plot(threshold_df['threshold'], threshold_df['precision'], 'g-', label='Precision', linewidth=2)
        ax4.plot(threshold_df['threshold'], threshold_df['recall'], 'orange', label='Recall', linewidth=2)
        ax4.set_xlabel('Classification Threshold')
        ax4.set_ylabel('Score')
        ax4.set_title('All Metrics vs Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'threshold_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Threshold analysis plot saved to {save_path}")
    
    def plot_roc_and_pr_curves(self, save_dir=None):
        """Plot ROC and Precision-Recall curves."""
        print("Generating ROC and PR curves...")
        
        if 'probabilities' not in self.results:
            raise ValueError("Must run evaluate_basic_metrics first")
        
        probabilities = self.results['probabilities']
        y_true = self.results['true_labels']
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
        roc_auc = roc_auc_score(y_true, probabilities)
        
        # Calculate Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, probabilities)
        avg_precision = average_precision_score(y_true, probabilities)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        ax2.plot(recall, precision, 'r-', linewidth=2, label=f'PR Curve (AP = {avg_precision:.4f})')
        # Baseline for imbalanced dataset
        baseline = y_true.mean()
        ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.4f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'roc_pr_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"ROC and PR curves saved to {save_path}")
        else:
            plt.show()
        
        # Store curve data
        self.results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds, 'auc': roc_auc}
        self.results['pr_curve'] = {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds, 'avg_precision': avg_precision}
        
        return {'roc_auc': roc_auc, 'avg_precision': avg_precision}
    
    def analyze_errors_detailed(self, test_df, save_dir=None, n_examples=10):
        """Perform detailed error analysis."""
        print("Performing detailed error analysis...")
        
        if 'predictions' not in self.results:
            raise ValueError("Must run evaluate_basic_metrics first")
        
        predictions = self.results['predictions']
        probabilities = self.results['probabilities']
        y_true = self.results['true_labels']
        
        # Identify errors
        errors = predictions != y_true
        error_indices = np.where(errors)[0]
        
        # Separate false positives and false negatives
        fp_mask = (predictions == 1) & (y_true == 0)
        fn_mask = (predictions == 0) & (y_true == 1)
        
        fp_indices = np.where(fp_mask)[0]
        fn_indices = np.where(fn_mask)[0]
        
        # Analyze confidence distribution
        fp_confidences = probabilities[fp_indices] if len(fp_indices) > 0 else []
        fn_confidences = 1 - probabilities[fn_indices] if len(fn_indices) > 0 else []
        correct_confidences = np.concatenate([
            probabilities[predictions == y_true][y_true[predictions == y_true] == 1],
            1 - probabilities[predictions == y_true][y_true[predictions == y_true] == 0]
        ])
        
        error_analysis = {
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(test_df),
            'false_positives': len(fp_indices),
            'false_negatives': len(fn_indices),
            'fp_avg_confidence': np.mean(fp_confidences) if len(fp_confidences) > 0 else 0,
            'fn_avg_confidence': np.mean(fn_confidences) if len(fn_confidences) > 0 else 0,
            'correct_avg_confidence': np.mean(correct_confidences)
        }
        
        print(f"\nDetailed Error Analysis:")
        print(f"Total Errors: {error_analysis['total_errors']} ({error_analysis['error_rate']:.1%})")
        print(f"False Positives: {error_analysis['false_positives']}")
        print(f"False Negatives: {error_analysis['false_negatives']}")
        print(f"Average Confidence - FP: {error_analysis['fp_avg_confidence']:.3f}")
        print(f"Average Confidence - FN: {error_analysis['fn_avg_confidence']:.3f}")
        print(f"Average Confidence - Correct: {error_analysis['correct_avg_confidence']:.3f}")
        
        # Show examples
        print(f"\nExample False Positives (predicted same, actually different):")
        if len(fp_indices) > 0:
            fp_examples = []
            for i, idx in enumerate(fp_indices[:n_examples]):
                example = {
                    'prompt1': test_df.iloc[idx]['prompt1'],
                    'prompt2': test_df.iloc[idx]['prompt2'],
                    'confidence': probabilities[idx],
                    'index': idx
                }
                fp_examples.append(example)
                print(f"  {i+1}. Confidence: {probabilities[idx]:.3f}")
                print(f"     Prompt 1: {test_df.iloc[idx]['prompt1']}")
                print(f"     Prompt 2: {test_df.iloc[idx]['prompt2']}")
                print()
        
        print(f"Example False Negatives (predicted different, actually same):")
        if len(fn_indices) > 0:
            fn_examples = []
            for i, idx in enumerate(fn_indices[:n_examples]):
                example = {
                    'prompt1': test_df.iloc[idx]['prompt1'],
                    'prompt2': test_df.iloc[idx]['prompt2'],
                    'confidence': 1 - probabilities[idx],
                    'index': idx
                }
                fn_examples.append(example)
                print(f"  {i+1}. Confidence: {1-probabilities[idx]:.3f}")
                print(f"     Prompt 1: {test_df.iloc[idx]['prompt1']}")
                print(f"     Prompt 2: {test_df.iloc[idx]['prompt2']}")
                print()
        
        # Plot confidence distributions
        if save_dir:
            self._plot_confidence_distributions(fp_confidences, fn_confidences, correct_confidences, save_dir)
        
        self.results['error_analysis'] = error_analysis
        return error_analysis
    
    def _plot_confidence_distributions(self, fp_confidences, fn_confidences, correct_confidences, save_dir):
        """Plot confidence score distributions for different prediction types."""
        plt.figure(figsize=(12, 8))
        
        # Plot distributions
        if len(fp_confidences) > 0:
            plt.hist(fp_confidences, bins=20, alpha=0.7, label=f'False Positives (n={len(fp_confidences)})', color='red')
        if len(fn_confidences) > 0:
            plt.hist(fn_confidences, bins=20, alpha=0.7, label=f'False Negatives (n={len(fn_confidences)})', color='orange')
        if len(correct_confidences) > 0:
            plt.hist(correct_confidences, bins=20, alpha=0.7, label=f'Correct Predictions (n={len(correct_confidences)})', color='green')
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Confidence Scores by Prediction Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, 'confidence_distributions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confidence distributions plot saved to {save_path}")
    
    def cross_validate(self, full_df, cv_folds=5, save_dir=None):
        """Perform cross-validation evaluation."""
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df['label'])):
            print(f"Processing fold {fold + 1}/{cv_folds}...")
            
            train_df = full_df.iloc[train_idx].reset_index(drop=True)
            val_df = full_df.iloc[val_idx].reset_index(drop=True)
            
            # Create a new classifier for this fold
            cv_classifier = PromptClassifier(
                model_type=self.classifier.model_type,
                hidden_dims=self.classifier.hidden_dims,
                dropout=self.classifier.dropout
            )
            
            # Train on fold
            cv_classifier.train(
                train_df=train_df,
                val_df=None,  # No validation during CV
                batch_size=32,
                epochs=30,  # Reduced epochs for CV
                learning_rate=0.001,
                patience=10,
                save_path=None  # Don't save CV models
            )
            
            # Evaluate on validation set
            predictions, probabilities = cv_classifier.predict(val_df)
            y_true = val_df['label'].values
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average='binary')
            roc_auc = roc_auc_score(y_true, probabilities)
            
            fold_results = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'n_samples': len(val_df)
            }
            
            cv_results.append(fold_results)
            print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Aggregate results
        cv_df = pd.DataFrame(cv_results)
        
        cv_summary = {
            'mean_accuracy': cv_df['accuracy'].mean(),
            'std_accuracy': cv_df['accuracy'].std(),
            'mean_precision': cv_df['precision'].mean(),
            'std_precision': cv_df['precision'].std(),
            'mean_recall': cv_df['recall'].mean(),
            'std_recall': cv_df['recall'].std(),
            'mean_f1': cv_df['f1_score'].mean(),
            'std_f1': cv_df['f1_score'].std(),
            'mean_roc_auc': cv_df['roc_auc'].mean(),
            'std_roc_auc': cv_df['roc_auc'].std()
        }
        
        print(f"\nCross-Validation Results ({cv_folds} folds):")
        print(f"Accuracy:  {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        print(f"Precision: {cv_summary['mean_precision']:.4f} ± {cv_summary['std_precision']:.4f}")
        print(f"Recall:    {cv_summary['mean_recall']:.4f} ± {cv_summary['std_recall']:.4f}")
        print(f"F1-Score:  {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}")
        print(f"ROC-AUC:   {cv_summary['mean_roc_auc']:.4f} ± {cv_summary['std_roc_auc']:.4f}")
        
        self.results['cross_validation'] = {
            'fold_results': cv_df,
            'summary': cv_summary
        }
        
        if save_dir:
            cv_path = os.path.join(save_dir, 'cross_validation_results.json')
            with open(cv_path, 'w') as f:
                # Convert DataFrame to dict for JSON serialization
                cv_data = {
                    'fold_results': cv_df.to_dict('records'),
                    'summary': cv_summary
                }
                json.dump(cv_data, f, indent=2)
            print(f"Cross-validation results saved to {cv_path}")
        
        return cv_summary
    
    def generate_comprehensive_report(self, test_df, save_dir):
        """Generate a comprehensive evaluation report."""
        print("Generating comprehensive evaluation report...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Run all evaluations
        print("1. Basic metrics...")
        basic_metrics = self.evaluate_basic_metrics(test_df)
        
        print("2. Threshold analysis...")
        threshold_analysis = self.evaluate_threshold_analysis(save_dir)
        
        print("3. ROC and PR curves...")
        curve_metrics = self.plot_roc_and_pr_curves(save_dir)
        
        print("4. Error analysis...")
        error_analysis = self.analyze_errors_detailed(test_df, save_dir)
        
        # Create comprehensive report
        report_path = os.path.join(save_dir, 'comprehensive_evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE PROMPT PARAPHRASE CLASSIFIER EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Samples: {len(test_df)}\n")
            f.write(f"Positive Samples: {basic_metrics['positive_samples']} ({basic_metrics['positive_samples']/len(test_df):.1%})\n")
            f.write(f"Negative Samples: {basic_metrics['negative_samples']} ({basic_metrics['negative_samples']/len(test_df):.1%})\n\n")
            
            # Performance metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy:           {basic_metrics['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy:  {basic_metrics['balanced_accuracy']:.4f}\n")
            f.write(f"Precision:          {basic_metrics['precision']:.4f}\n")
            f.write(f"Recall:             {basic_metrics['recall']:.4f}\n")
            f.write(f"F1-Score:           {basic_metrics['f1_score']:.4f}\n")
            f.write(f"ROC-AUC:            {basic_metrics['roc_auc']:.4f}\n")
            f.write(f"Average Precision:  {basic_metrics['average_precision']:.4f}\n")
            f.write(f"Specificity:        {basic_metrics['specificity']:.4f}\n\n")
            
            # Confusion matrix
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 40 + "\n")
            f.write(f"True Positives:     {basic_metrics['true_positives']}\n")
            f.write(f"True Negatives:     {basic_metrics['true_negatives']}\n")
            f.write(f"False Positives:    {basic_metrics['false_positives']}\n")
            f.write(f"False Negatives:    {basic_metrics['false_negatives']}\n\n")
            
            # Error analysis
            f.write("ERROR ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Errors:       {error_analysis['total_errors']} ({error_analysis['error_rate']:.1%})\n")
            f.write(f"False Positives:    {error_analysis['false_positives']}\n")
            f.write(f"False Negatives:    {error_analysis['false_negatives']}\n")
            f.write(f"FP Avg Confidence:  {error_analysis['fp_avg_confidence']:.3f}\n")
            f.write(f"FN Avg Confidence:  {error_analysis['fn_avg_confidence']:.3f}\n")
            f.write(f"Correct Avg Conf:   {error_analysis['correct_avg_confidence']:.3f}\n\n")
            
            # Optimal thresholds
            optimal_f1_idx = threshold_analysis['f1_score'].idxmax()
            optimal_acc_idx = threshold_analysis['accuracy'].idxmax()
            f.write("OPTIMAL THRESHOLDS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best F1 Threshold:  {threshold_analysis.loc[optimal_f1_idx, 'threshold']:.3f} "
                   f"(F1: {threshold_analysis.loc[optimal_f1_idx, 'f1_score']:.4f})\n")
            f.write(f"Best Acc Threshold: {threshold_analysis.loc[optimal_acc_idx, 'threshold']:.3f} "
                   f"(Acc: {threshold_analysis.loc[optimal_acc_idx, 'accuracy']:.4f})\n\n")
            
            # Model configuration
            f.write("MODEL CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model Type:         {self.classifier.model_type}\n")
            f.write(f"Embedding Dim:      {self.classifier.embed_dim}\n")
            f.write(f"Input Dim:          {self.classifier.input_dim}\n")
            f.write(f"Hidden Dims:        {self.classifier.hidden_dims}\n")
            f.write(f"Dropout Rate:       {self.classifier.dropout}\n")
        
        # Save all results as pickle
        results_path = os.path.join(save_dir, 'all_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Comprehensive evaluation report saved to {report_path}")
        print(f"All results saved to {results_path}")
        
        return {
            'basic_metrics': basic_metrics,
            'threshold_analysis': threshold_analysis,
            'curve_metrics': curve_metrics,
            'error_analysis': error_analysis
        }
    
    def evaluate_language_robustness(self, multilingual_test_df, save_dir=None):
        """Evaluate model performance across different languages."""
        print("Evaluating language robustness...")
        
        if 'language' not in multilingual_test_df.columns:
            print("Warning: No language column found in test data")
            return None
        
        languages = multilingual_test_df['language'].unique()
        language_results = {}
        
        for lang in languages:
            lang_df = multilingual_test_df[multilingual_test_df['language'] == lang]
            if len(lang_df) == 0:
                continue
                
            print(f"Evaluating {lang} ({len(lang_df)} samples)...")
            
            # Get predictions for this language
            predictions, probabilities = self.classifier.predict(lang_df)
            y_true = lang_df['label'].values
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average='binary', zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_true, probabilities)
            except ValueError:
                roc_auc = 0.0  # Handle case where only one class is present
            
            language_results[lang] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'n_samples': len(lang_df),
                'positive_ratio': y_true.mean()
            }
        
        # Create summary
        lang_summary = pd.DataFrame(language_results).T
        lang_summary = lang_summary.sort_values('f1_score', ascending=False)
        
        print("\nLanguage-wise Performance:")
        print(lang_summary.round(4))
        
        # Save results
        if save_dir:
            lang_results_path = os.path.join(save_dir, 'language_robustness.csv')
            lang_summary.to_csv(lang_results_path)
            print(f"Language robustness results saved to {lang_results_path}")
        
        self.results['language_robustness'] = language_results
        return language_results
    
    def evaluate_similarity_threshold_sensitivity(self, test_df, save_dir=None):
        """Evaluate how sensitive the model is to different similarity thresholds."""
        print("Evaluating similarity threshold sensitivity...")
        
        if 'probabilities' not in self.results:
            raise ValueError("Must run evaluate_basic_metrics first")
        
        probabilities = self.results['probabilities']
        y_true = self.results['true_labels']
        
        # Define similarity ranges
        similarity_ranges = [
            (0.0, 0.3, "Low Similarity"),
            (0.3, 0.7, "Medium Similarity"), 
            (0.7, 1.0, "High Similarity")
        ]
        
        range_results = {}
        
        for min_sim, max_sim, range_name in similarity_ranges:
            # Filter samples in this similarity range
            mask = (probabilities >= min_sim) & (probabilities < max_sim)
            
            if mask.sum() == 0:
                continue
                
            range_probs = probabilities[mask]
            range_true = y_true[mask]
            range_preds = (range_probs >= 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(range_true, range_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(range_true, range_preds, average='binary', zero_division=0)
            
            range_results[range_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'n_samples': mask.sum(),
                'avg_probability': range_probs.mean(),
                'positive_ratio': range_true.mean()
            }
        
        # Create summary
        range_summary = pd.DataFrame(range_results).T
        
        print("\nSimilarity Range Performance:")
        print(range_summary.round(4))
        
        # Save results
        if save_dir:
            range_results_path = os.path.join(save_dir, 'similarity_range_analysis.csv')
            range_summary.to_csv(range_results_path)
            print(f"Similarity range analysis saved to {range_results_path}")
        
        self.results['similarity_range_analysis'] = range_results
        return range_results
    
    def plot_confusion_matrix(self, save_dir=None):
        """Plot detailed confusion matrix with percentages."""
        print("Generating confusion matrix visualization...")
        
        if 'predictions' not in self.results:
            raise ValueError("Must run evaluate_basic_metrics first")
        
        predictions = self.results['predictions']
        y_true = self.results['true_labels']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Different', 'Same'], yticklabels=['Different', 'Same'])
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Normalized percentages
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                   xticklabels=['Different', 'Same'], yticklabels=['Different', 'Same'])
        ax2.set_title('Confusion Matrix (Percentages)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        return cm, cm_normalized


def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of prompt paraphrase classifier')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data CSV')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/',
                       help='Directory to save evaluation results')
    parser.add_argument('--cv_data', type=str, default=None,
                       help='Path to full dataset for cross-validation')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--run_cv', action='store_true',
                       help='Run cross-validation evaluation')
    parser.add_argument('--multilingual_test', type=str, default=None,
                       help='Path to multilingual test data with language column')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(args.test_data)
    print(f"Test samples: {len(test_df)}")
    print(f"Positive ratio: {test_df['label'].mean():.3f}")
    
    # Initialize evaluator
    print("Loading model...")
    evaluator = ModelEvaluator(model_path=args.model_path)
    
    # Run comprehensive evaluation
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE EVALUATION")
    print("="*80)
    
    # Main evaluation
    eval_results = evaluator.generate_comprehensive_report(test_df, args.output_dir)
    
    # Additional evaluations
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSES")
    print("="*60)
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(args.output_dir)
    
    # Similarity threshold sensitivity
    evaluator.evaluate_similarity_threshold_sensitivity(test_df, args.output_dir)
    
    # Multilingual evaluation if provided
    if args.multilingual_test:
        print("\nEvaluating multilingual robustness...")
        multilingual_df = pd.read_csv(args.multilingual_test)
        evaluator.evaluate_language_robustness(multilingual_df, args.output_dir)
    
    # Cross-validation if requested
    if args.run_cv and args.cv_data:
        print("\nRunning cross-validation...")
        cv_df = pd.read_csv(args.cv_data)
        cv_results = evaluator.cross_validate(cv_df, args.cv_folds, args.output_dir)
        
        # Compare CV results with holdout test
        print(f"\nCross-validation vs Holdout Test Comparison:")
        print(f"CV Accuracy:    {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"Test Accuracy:  {eval_results['basic_metrics']['accuracy']:.4f}")
        print(f"CV F1-Score:    {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        print(f"Test F1-Score:  {eval_results['basic_metrics']['f1_score']:.4f}")
        print(f"CV ROC-AUC:     {cv_results['mean_roc_auc']:.4f} ± {cv_results['std_roc_auc']:.4f}")
        print(f"Test ROC-AUC:   {eval_results['basic_metrics']['roc_auc']:.4f}")
    
    # Generate final summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    summary_path = os.path.join(args.output_dir, 'evaluation_summary.json')
    summary_data = {
        'model_path': args.model_path,
        'test_data': args.test_data,
        'test_samples': len(test_df),
        'test_positive_ratio': float(test_df['label'].mean()),
        'performance': eval_results['basic_metrics'],
        'evaluation_timestamp': pd.Timestamp.now().isoformat()
    }
    
    if args.run_cv and args.cv_data:
        summary_data['cross_validation'] = cv_results
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Final evaluation summary saved to {summary_path}")
    print(f"All evaluation artifacts saved to {args.output_dir}")
    
    # Performance summary
    metrics = eval_results['basic_metrics']
    print(f"\nFINAL PERFORMANCE METRICS:")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1-Score:      {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:       {metrics['roc_auc']:.4f}")
    print(f"  Specificity:   {metrics['specificity']:.4f}")
    
    # Model recommendations
    print(f"\nMODEL RECOMMENDATIONS:")
    if metrics['roc_auc'] >= 0.9:
        print("  ✓ Excellent performance - model ready for production")
    elif metrics['roc_auc'] >= 0.8:
        print("  ✓ Good performance - consider additional validation")
    elif metrics['roc_auc'] >= 0.7:
        print("  ⚠ Fair performance - consider model improvements")
    else:
        print("  ✗ Poor performance - model needs significant improvements")
    
    if metrics['precision'] < 0.8:
        print("  ⚠ Low precision - consider adjusting threshold to reduce false positives")
    if metrics['recall'] < 0.8:
        print("  ⚠ Low recall - consider adjusting threshold to reduce false negatives")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()