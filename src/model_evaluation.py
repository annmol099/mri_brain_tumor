"""
Complete Evaluation Pipeline for Brain Tumor MRI Classification
Phase 3: Comprehensive Model Evaluation with Metrics, Visualizations, and Grad-CAM
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path
from datetime import datetime

from evaluation_metrics import ModelEvaluator
from visualization import EvaluationVisualizer
from gradcam import GradCAM


class ComprehensiveEvaluator:
    """Complete evaluation pipeline combining metrics, visualizations, and Grad-CAM."""
    
    def __init__(self, model_path, class_names, output_dir='../results'):
        """
        Initialize comprehensive evaluator.
        
        Args:
            model_path (str): Path to trained model
            class_names (list): List of class names
            output_dir (str): Directory for results
        """
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.metrics_dir = self.output_dir / 'metrics'
        self.plots_dir = self.output_dir / 'plots'
        self.gradcam_dir = self.output_dir / 'gradcam'
        
        for dir_path in [self.metrics_dir, self.plots_dir, self.gradcam_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"\nLoading model from: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        
        # Initialize components
        self.evaluator = ModelEvaluator(self.model, class_names)
        self.visualizer = EvaluationVisualizer(class_names, str(self.plots_dir))
        self.gradcam = GradCAM(self.model)
        
        print("✓ Evaluator initialized")
    
    def load_test_images(self, dataset, max_samples=100):
        """
        Load images from dataset for visualization.
        
        Args:
            dataset: TensorFlow dataset
            max_samples (int): Maximum number of samples to load
            
        Returns:
            tuple: (images, labels)
        """
        images_list = []
        labels_list = []
        
        for images, labels in dataset:
            images_list.append(images.numpy())
            labels_list.append(labels.numpy())
            
            if sum(len(img) for img in images_list) >= max_samples:
                break
        
        images = np.concatenate(images_list, axis=0)[:max_samples]
        labels = np.concatenate(labels_list, axis=0)[:max_samples]
        
        # Convert one-hot to indices if needed
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = np.argmax(labels, axis=1)
        
        return images, labels
    
    def evaluate(self, test_dataset, training_history=None, generate_gradcam=True):
        """
        Run complete evaluation pipeline.
        
        Args:
            test_dataset: TensorFlow test dataset
            training_history (dict): Training history for plotting
            generate_gradcam (bool): Whether to generate Grad-CAM visualizations
            
        Returns:
            dict: Complete evaluation results
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*70)
        print(f"Model: {self.model_path.name}")
        print(f"Output directory: {self.output_dir}")
        
        # Step 1: Calculate quantitative metrics
        print("\n" + "="*70)
        print("STEP 1: CALCULATING METRICS")
        print("="*70)
        metrics = self.evaluator.evaluate_all(test_dataset)
        
        # Save metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = self.metrics_dir / f'metrics_{timestamp}.json'
        self.evaluator.save_metrics(metrics_path)
        
        # Step 2: Create visualizations
        print("\n" + "="*70)
        print("STEP 2: CREATING VISUALIZATIONS")
        print("="*70)
        
        # Load sample images for visualization
        print("Loading sample images...")
        images, labels = self.load_test_images(test_dataset, max_samples=100)
        
        # Get predictions
        y_true = self.evaluator.y_true
        y_pred = self.evaluator.y_pred
        y_pred_proba = self.evaluator.y_pred_proba
        
        # Create all plots
        self.visualizer.create_all_plots(
            history=training_history,
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            images=images,
            per_class_metrics=metrics.get('per_class_metrics')
        )
        
        # Step 3: Generate Grad-CAM visualizations
        if generate_gradcam:
            print("\n" + "="*70)
            print("STEP 3: GENERATING GRAD-CAM VISUALIZATIONS")
            print("="*70)
            
            self.gradcam.visualize_multiple_samples(
                images[:16],  # Use first 16 samples
                self.class_names,
                true_labels=labels[:16],
                num_samples=8,
                output_dir=str(self.gradcam_dir),
                save_name=f'gradcam_samples_{timestamp}.png'
            )
        
        # Step 4: Generate summary report
        print("\n" + "="*70)
        print("STEP 4: GENERATING SUMMARY REPORT")
        print("="*70)
        
        report = self.generate_report(metrics, timestamp)
        report_path = self.output_dir / f'evaluation_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Summary report saved: {report_path}")
        
        # Final summary
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print(f"\n📁 Results Location: {self.output_dir}")
        print(f"   • Metrics: {self.metrics_dir}")
        print(f"   • Plots: {self.plots_dir}")
        if generate_gradcam:
            print(f"   • Grad-CAM: {self.gradcam_dir}")
        print(f"\n📊 Overall Accuracy: {metrics['accuracy']:.2%}")
        print(f"📈 Macro F1-Score: {metrics['classification_report']['macro avg']['f1-score']:.4f}")
        
        return {
            'metrics': metrics,
            'output_dir': str(self.output_dir),
            'timestamp': timestamp
        }
    
    def generate_report(self, metrics, timestamp):
        """
        Generate text summary report.
        
        Args:
            metrics (dict): Evaluation metrics
            timestamp (str): Timestamp for report
            
        Returns:
            str: Formatted report text
        """
        report = []
        report.append("="*70)
        report.append("BRAIN TUMOR MRI CLASSIFICATION - EVALUATION REPORT")
        report.append("="*70)
        report.append(f"\nModel: {self.model_path.name}")
        report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of Classes: {len(self.class_names)}")
        report.append(f"Classes: {', '.join(self.class_names)}")
        
        report.append("\n" + "="*70)
        report.append("OVERALL PERFORMANCE")
        report.append("="*70)
        report.append(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
        
        weighted = metrics.get('weighted_metrics', {})
        if weighted:
            report.append(f"Precision (weighted): {weighted['precision']:.4f}")
            report.append(f"Recall (weighted): {weighted['recall']:.4f}")
            report.append(f"F1-Score (weighted): {weighted['f1_score']:.4f}")
        
        if 'roc_auc_macro' in metrics:
            report.append(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
        
        report.append("\n" + "="*70)
        report.append("PER-CLASS PERFORMANCE")
        report.append("="*70)
        report.append(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        report.append("-"*70)
        
        per_class = metrics.get('per_class_metrics', {})
        roc_auc = metrics.get('roc_auc_per_class', {})
        
        for class_name in self.class_names:
            if class_name in per_class:
                m = per_class[class_name]
                auc = roc_auc.get(class_name, 'N/A') if roc_auc else 'N/A'
                auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
                report.append(
                    f"{class_name:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                    f"{m['f1_score']:<12.4f} {auc_str:<12}"
                )
        
        report.append("\n" + "="*70)
        report.append("CONFUSION MATRIX")
        report.append("="*70)
        
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            report.append(f"\n{'Predicted →':<15} " + " ".join([f"{c[:8]:>8}" for c in self.class_names]))
            report.append("True ↓")
            for i, class_name in enumerate(self.class_names):
                row = " ".join([f"{cm[i, j]:>8}" for j in range(len(self.class_names))])
                report.append(f"{class_name[:8]:<15} {row}")
        
        report.append("\n" + "="*70)
        report.append("FILES GENERATED")
        report.append("="*70)
        report.append(f"• Metrics JSON: metrics/metrics_{timestamp}.json")
        report.append(f"• Training curves: plots/training_history.png")
        report.append(f"• Confusion matrix: plots/confusion_matrix.png")
        report.append(f"• ROC curves: plots/roc_curves.png")
        report.append(f"• Sample predictions: plots/sample_predictions.png")
        report.append(f"• Grad-CAM visualizations: gradcam/gradcam_samples_{timestamp}.png")
        
        report.append("\n" + "="*70)
        report.append("END OF REPORT")
        report.append("="*70)
        
        return "\n".join(report)


def run_evaluation(model_path, test_dataset, class_names, 
                  training_history_path=None, output_dir='../results'):
    """
    Convenience function to run complete evaluation.
    
    Args:
        model_path (str): Path to trained model
        test_dataset: TensorFlow test dataset
        class_names (list): List of class names
        training_history_path (str): Path to training history JSON
        output_dir (str): Output directory
        
    Returns:
        dict: Evaluation results
    """
    # Load training history if provided
    training_history = None
    if training_history_path and Path(training_history_path).exists():
        with open(training_history_path, 'r') as f:
            data = json.load(f)
            # Extract history from different possible structures
            if 'stage1_history' in data:
                training_history = data['stage1_history']
            elif 'history' in data:
                training_history = data['history']
            else:
                training_history = data
    
    # Create evaluator and run evaluation
    evaluator = ComprehensiveEvaluator(model_path, class_names, output_dir)
    results = evaluator.evaluate(test_dataset, training_history, generate_gradcam=True)
    
    return results


if __name__ == "__main__":
    print("Comprehensive Evaluation Pipeline")
    print("="*70)
    print("\nThis module provides complete model evaluation:")
    print("  1. Quantitative Metrics")
    print("     • Accuracy, Precision, Recall, F1-Score")
    print("     • Confusion Matrix")
    print("     • ROC-AUC per class")
    print("\n  2. Visualizations")
    print("     • Training/validation curves")
    print("     • Confusion matrix heatmap")
    print("     • ROC curves")
    print("     • Sample predictions")
    print("\n  3. Grad-CAM")
    print("     • Tumor localization")
    print("     • Model interpretability")
    print("\n  4. Summary Report")
    print("     • Text-based evaluation summary")
    print("\nUse ComprehensiveEvaluator class or run_evaluation() function.")
