"""
Evaluation Metrics Module for Brain Tumor MRI Classification
Phase 3.1: Quantitative Metrics (Accuracy, Precision, Recall, F1-score)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import json
from pathlib import Path
from datetime import datetime


class ModelEvaluator:
    """Comprehensive model evaluation with quantitative metrics."""
    
    def __init__(self, model, class_names):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Keras model
            class_names (list): List of class names
        """
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
    
    def predict(self, dataset):
        """
        Generate predictions for dataset.
        
        Args:
            dataset: TensorFlow dataset
            
        Returns:
            tuple: (y_true, y_pred, y_pred_proba)
        """
        print("\nGenerating predictions...")
        
        # Get predictions
        y_true_list = []
        y_pred_proba_list = []
        
        for images, labels in dataset:
            # Get predictions
            predictions = self.model.predict(images, verbose=0)
            
            # Store true labels and predictions
            y_true_list.append(labels.numpy())
            y_pred_proba_list.append(predictions)
        
        # Concatenate all batches
        self.y_true = np.concatenate(y_true_list, axis=0)
        self.y_pred_proba = np.concatenate(y_pred_proba_list, axis=0)
        
        # Convert probabilities to class predictions
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        
        # Convert one-hot encoded true labels to class indices
        if len(self.y_true.shape) > 1 and self.y_true.shape[1] > 1:
            self.y_true = np.argmax(self.y_true, axis=1)
        
        print(f"Predictions generated for {len(self.y_true)} samples")
        
        return self.y_true, self.y_pred, self.y_pred_proba
    
    def calculate_accuracy(self):
        """Calculate overall accuracy."""
        accuracy = accuracy_score(self.y_true, self.y_pred)
        self.metrics['accuracy'] = accuracy
        return accuracy
    
    def calculate_precision_recall_f1(self, average='weighted'):
        """
        Calculate precision, recall, and F1-score.
        
        Args:
            average (str): Averaging method ('micro', 'macro', 'weighted', None)
            
        Returns:
            dict: Precision, recall, and F1-score
        """
        precision = precision_score(self.y_true, self.y_pred, 
                                   average=average, zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, 
                             average=average, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, 
                     average=average, zero_division=0)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        self.metrics[f'{average}_metrics'] = metrics
        return metrics
    
    def calculate_per_class_metrics(self):
        """Calculate metrics for each class individually."""
        print("\nCalculating per-class metrics...")
        
        # Calculate metrics per class
        precision_per_class = precision_score(self.y_true, self.y_pred, 
                                             average=None, zero_division=0)
        recall_per_class = recall_score(self.y_true, self.y_pred, 
                                        average=None, zero_division=0)
        f1_per_class = f1_score(self.y_true, self.y_pred, 
                               average=None, zero_division=0)
        
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }
        
        self.metrics['per_class_metrics'] = per_class_metrics
        return per_class_metrics
    
    def calculate_confusion_matrix(self):
        """Calculate confusion matrix."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        self.metrics['confusion_matrix'] = cm.tolist()
        return cm
    
    def calculate_roc_auc(self):
        """Calculate ROC-AUC score for each class."""
        print("\nCalculating ROC-AUC scores...")
        
        # Convert y_true to one-hot if needed
        y_true_onehot = keras.utils.to_categorical(self.y_true, self.num_classes)
        
        try:
            # Overall ROC-AUC (macro average)
            roc_auc_macro = roc_auc_score(y_true_onehot, self.y_pred_proba, 
                                         average='macro', multi_class='ovr')
            
            # ROC-AUC per class
            roc_auc_per_class = {}
            for i, class_name in enumerate(self.class_names):
                try:
                    roc_auc = roc_auc_score(y_true_onehot[:, i], 
                                           self.y_pred_proba[:, i])
                    roc_auc_per_class[class_name] = float(roc_auc)
                except ValueError:
                    # Handle case where class is not present in y_true
                    roc_auc_per_class[class_name] = None
            
            self.metrics['roc_auc_macro'] = float(roc_auc_macro)
            self.metrics['roc_auc_per_class'] = roc_auc_per_class
            
            return roc_auc_macro, roc_auc_per_class
        
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")
            return None, None
    
    def get_classification_report(self):
        """Generate classification report."""
        report = classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        self.metrics['classification_report'] = report
        return report
    
    def evaluate_all(self, dataset):
        """
        Run complete evaluation pipeline.
        
        Args:
            dataset: TensorFlow dataset
            
        Returns:
            dict: All evaluation metrics
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Generate predictions
        self.predict(dataset)
        
        # Calculate all metrics
        print("\nCalculating metrics...")
        
        # Accuracy
        accuracy = self.calculate_accuracy()
        print(f"Accuracy: {accuracy:.4f}")
        
        # Precision, Recall, F1 (weighted average)
        weighted_metrics = self.calculate_precision_recall_f1('weighted')
        print(f"Precision (weighted): {weighted_metrics['precision']:.4f}")
        print(f"Recall (weighted): {weighted_metrics['recall']:.4f}")
        print(f"F1-Score (weighted): {weighted_metrics['f1_score']:.4f}")
        
        # Per-class metrics
        per_class_metrics = self.calculate_per_class_metrics()
        
        # Confusion matrix
        cm = self.calculate_confusion_matrix()
        
        # ROC-AUC
        roc_auc_macro, roc_auc_per_class = self.calculate_roc_auc()
        
        # Classification report
        report = self.get_classification_report()
        
        # Print summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
        if roc_auc_macro is not None:
            print(f"ROC-AUC (macro): {roc_auc_macro:.4f}")
        
        print("\nPer-Class Performance:")
        print("-" * 70)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("-" * 70)
        for class_name in self.class_names:
            metrics = per_class_metrics[class_name]
            roc_auc = roc_auc_per_class.get(class_name, 'N/A') if roc_auc_per_class else 'N/A'
            roc_auc_str = f"{roc_auc:.4f}" if isinstance(roc_auc, float) else str(roc_auc)
            print(f"{class_name:<15} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} "
                  f"{roc_auc_str:<12}")
        print("-" * 70)
        
        return self.metrics
    
    def save_metrics(self, output_path):
        """
        Save metrics to JSON file.
        
        Args:
            output_path (str): Path to save metrics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_to_save = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.ndarray):
                metrics_to_save[key] = value.tolist()
            else:
                metrics_to_save[key] = value
        
        # Add metadata
        metrics_to_save['metadata'] = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_samples': int(len(self.y_true)),
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"\nMetrics saved to: {output_path}")


def evaluate_model(model_path, dataset, class_names, output_dir='../results'):
    """
    Convenience function to evaluate a saved model.
    
    Args:
        model_path (str): Path to saved model
        dataset: TensorFlow dataset
        class_names (list): List of class names
        output_dir (str): Directory to save results
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    evaluator = ModelEvaluator(model, class_names)
    metrics = evaluator.evaluate_all(dataset)
    
    # Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f'evaluation_metrics_{timestamp}.json'
    evaluator.save_metrics(output_path)
    
    return metrics


if __name__ == "__main__":
    print("Model Evaluation Metrics Module")
    print("="*70)
    print("\nThis module provides comprehensive evaluation metrics:")
    print("  • Accuracy")
    print("  • Precision, Recall, F1-Score (per class and overall)")
    print("  • Confusion Matrix")
    print("  • ROC-AUC (per class and macro average)")
    print("  • Classification Report")
    print("\nUse evaluate_model() or ModelEvaluator class for evaluation.")
