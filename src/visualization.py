"""
Visualization Module for Brain Tumor MRI Classification
Phase 3.2: Training Curves, Confusion Matrix, ROC Curves, Sample Predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
import json


class EvaluationVisualizer:
    """Create visualizations for model evaluation."""
    
    def __init__(self, class_names, output_dir='../results/plots'):
        """
        Initialize visualizer.
        
        Args:
            class_names (list): List of class names
            output_dir (str): Directory to save plots
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_training_history(self, history, save_name='training_history.png'):
        """
        Plot training and validation accuracy/loss curves.
        
        Args:
            history (dict): Training history
            save_name (str): Filename to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        if 'accuracy' in history:
            axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        if 'loss' in history:
            axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_name='confusion_matrix.png', 
                             normalize=False):
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_name (str): Filename to save plot
            normalize (bool): Whether to normalize the confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved: {save_path}")
        plt.close()
    
    def plot_roc_curves(self, y_true, y_pred_proba, save_name='roc_curves.png'):
        """
        Plot ROC curves for each class.
        
        Args:
            y_true: True labels (one-hot encoded or class indices)
            y_pred_proba: Predicted probabilities
            save_name (str): Filename to save plot
        """
        # Convert y_true to one-hot if needed
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            y_true_onehot = keras.utils.to_categorical(y_true, self.num_classes)
        else:
            y_true_onehot = y_true
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            try:
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{class_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"Warning: Could not plot ROC for {class_name}: {e}")
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Multi-Class Classification', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved: {save_path}")
        plt.close()
    
    def plot_sample_predictions(self, images, y_true, y_pred, y_pred_proba, 
                               num_samples=16, save_name='sample_predictions.png'):
        """
        Plot sample predictions with images.
        
        Args:
            images: Array of images
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            num_samples (int): Number of samples to plot
            save_name (str): Filename to save plot
        """
        # Select random samples
        indices = np.random.choice(len(images), size=min(num_samples, len(images)), 
                                  replace=False)
        
        # Calculate grid size
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for idx, ax in enumerate(axes):
            if idx < len(indices):
                i = indices[idx]
                
                # Get image
                img = images[i]
                
                # Denormalize if needed (assuming 0-1 range)
                if img.max() <= 1.0:
                    img = img
                
                # Display image
                ax.imshow(img)
                
                # Get true and predicted labels
                true_label = self.class_names[y_true[i]]
                pred_label = self.class_names[y_pred[i]]
                confidence = y_pred_proba[i][y_pred[i]] * 100
                
                # Set title with color based on correctness
                is_correct = y_true[i] == y_pred[i]
                title_color = 'green' if is_correct else 'red'
                title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%"
                
                ax.set_title(title, fontsize=10, color=title_color, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved: {save_path}")
        plt.close()
    
    def plot_class_distribution(self, y_true, save_name='class_distribution.png'):
        """
        Plot class distribution bar chart.
        
        Args:
            y_true: True labels
            save_name (str): Filename to save plot
        """
        # Count samples per class
        unique, counts = np.unique(y_true, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
        bars = plt.bar([self.class_names[i] for i in unique], counts, color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.xlabel('Class', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
        plt.title('Class Distribution in Test Set', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution saved: {save_path}")
        plt.close()
    
    def plot_metrics_comparison(self, metrics_dict, save_name='metrics_comparison.png'):
        """
        Plot bar chart comparing metrics across classes.
        
        Args:
            metrics_dict (dict): Dictionary of per-class metrics
            save_name (str): Filename to save plot
        """
        classes = list(metrics_dict.keys())
        precision = [metrics_dict[c]['precision'] for c in classes]
        recall = [metrics_dict[c]['recall'] for c in classes]
        f1 = [metrics_dict[c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics by Class', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved: {save_path}")
        plt.close()
    
    def create_all_plots(self, history=None, y_true=None, y_pred=None, 
                        y_pred_proba=None, images=None, per_class_metrics=None):
        """
        Create all visualization plots.
        
        Args:
            history (dict): Training history
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            images: Sample images
            per_class_metrics (dict): Per-class metrics
        """
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
        if history is not None:
            print("\n1. Plotting training history...")
            self.plot_training_history(history)
        
        if y_true is not None and y_pred is not None:
            print("2. Plotting confusion matrix...")
            self.plot_confusion_matrix(y_true, y_pred)
            self.plot_confusion_matrix(y_true, y_pred, 
                                      save_name='confusion_matrix_normalized.png',
                                      normalize=True)
            
            print("3. Plotting class distribution...")
            self.plot_class_distribution(y_true)
        
        if y_true is not None and y_pred_proba is not None:
            print("4. Plotting ROC curves...")
            self.plot_roc_curves(y_true, y_pred_proba)
        
        if images is not None and y_true is not None and y_pred is not None:
            print("5. Plotting sample predictions...")
            self.plot_sample_predictions(images, y_true, y_pred, y_pred_proba)
        
        if per_class_metrics is not None:
            print("6. Plotting metrics comparison...")
            self.plot_metrics_comparison(per_class_metrics)
        
        print("\n" + "="*70)
        print(f"All plots saved to: {self.output_dir}")
        print("="*70)


if __name__ == "__main__":
    print("Evaluation Visualization Module")
    print("="*70)
    print("\nThis module provides visualization utilities:")
    print("  • Training/Validation curves")
    print("  • Confusion matrix (raw and normalized)")
    print("  • ROC curves (per class)")
    print("  • Sample predictions with images")
    print("  • Class distribution")
    print("  • Metrics comparison bar chart")
    print("\nUse EvaluationVisualizer class to create plots.")
