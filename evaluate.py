"""
Command-Line Interface for Model Evaluation
Phase 3: Brain Tumor MRI Classification - Evaluation Script

Usage:
    python evaluate.py --model models/best_model.h5 --data data/processed
    python evaluate.py --model models/best_model.h5 --data data/processed --no-gradcam
    python evaluate.py --model models/best_model.h5 --config config.json
"""

import argparse
import json
import sys
from pathlib import Path
import tensorflow as tf

# Configure GPU for evaluation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU enabled for evaluation ({len(gpus)} GPU(s) found)")
    except RuntimeError as e:
        print(f"⚠ GPU configuration warning: {e}")

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from model_evaluation import ComprehensiveEvaluator, run_evaluation
from data_loaders import create_data_loaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate Brain Tumor MRI Classification Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate.py --model models/best_model.h5 --data data/processed

  # With custom output directory
  python evaluate.py --model models/best_model.h5 --data data/processed --output results/eval1

  # Skip Grad-CAM generation (faster)
  python evaluate.py --model models/best_model.h5 --data data/processed --no-gradcam

  # Use configuration file
  python evaluate.py --config config.json

  # With training history
  python evaluate.py --model models/best_model.h5 --data data/processed --history logs/training_history.json
        """
    )
    
    # Input arguments
    parser.add_argument(
        '--model',
        type=str,
        help='Path to trained model (.h5 or SavedModel directory)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to processed data directory containing test split'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config.json file (alternative to --model and --data)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output',
        type=str,
        default='../results',
        help='Output directory for evaluation results (default: ../results)'
    )
    
    parser.add_argument(
        '--history',
        type=str,
        help='Path to training history JSON file for plotting training curves'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    
    parser.add_argument(
        '--no-gradcam',
        action='store_true',
        help='Skip Grad-CAM generation (faster evaluation)'
    )
    
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
        help='Class names (default: Glioma Meningioma "No Tumor" Pituitary)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=224,
        help='Image size (default: 224)'
    )
    
    return parser.parse_args()


def load_config(config_path):
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_paths(model_path, data_path):
    """
    Validate that required paths exist.
    
    Args:
        model_path (str): Path to model
        data_path (str): Path to data directory
        
    Returns:
        tuple: (model_path, data_path) as Path objects
        
    Raises:
        FileNotFoundError: If paths don't exist
    """
    model_path = Path(model_path)
    data_path = Path(data_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    test_dir = data_path / 'test'
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    return model_path, data_path


def create_test_dataset(data_path, batch_size, img_size, class_names):
    """
    Create test dataset.
    
    Args:
        data_path (Path): Data directory path
        batch_size (int): Batch size
        img_size (int): Image size
        class_names (list): Class names
        
    Returns:
        tf.data.Dataset: Test dataset
    """
    print("\nLoading test dataset...")
    
    # Use data loaders module
    _, _, test_dataset = create_data_loaders(
        data_dir=str(data_path),
        batch_size=batch_size,
        img_size=img_size,
        augment_train=False,  # No augmentation needed for test
        validation_split=0.15,  # Not used but required by function
        seed=42
    )
    
    # Count samples
    num_samples = 0
    for batch in test_dataset:
        num_samples += len(batch[0])
    
    print(f"✓ Test dataset loaded: {num_samples} samples")
    
    # Recreate to reset iterator
    _, _, test_dataset = create_data_loaders(
        data_dir=str(data_path),
        batch_size=batch_size,
        img_size=img_size,
        augment_train=False,
        validation_split=0.15,
        seed=42
    )
    
    return test_dataset


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("="*70)
    print("BRAIN TUMOR MRI CLASSIFICATION - MODEL EVALUATION")
    print("="*70)
    
    # Load configuration if provided
    if args.config:
        print(f"\nLoading configuration from: {args.config}")
        config = load_config(args.config)
        
        model_path = config.get('best_model_path', args.model)
        data_path = config.get('processed_data_dir', args.data)
        class_names = config.get('class_names', args.classes)
        img_size = config.get('img_size', args.img_size)
    else:
        if not args.model or not args.data:
            print("\nError: Either --config or both --model and --data are required")
            print("Use --help for usage information")
            sys.exit(1)
        
        model_path = args.model
        data_path = args.data
        class_names = args.classes
        img_size = args.img_size
    
    # Validate paths
    try:
        model_path, data_path = validate_paths(model_path, data_path)
        print(f"\n✓ Model: {model_path}")
        print(f"✓ Data: {data_path}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output: {output_dir}")
    
    # Display configuration
    print("\nConfiguration:")
    print(f"  • Classes: {', '.join(class_names)}")
    print(f"  • Image size: {img_size}x{img_size}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Grad-CAM: {'Disabled' if args.no_gradcam else 'Enabled'}")
    if args.history:
        print(f"  • Training history: {args.history}")
    
    # Create test dataset
    try:
        test_dataset = create_test_dataset(
            data_path, 
            args.batch_size, 
            img_size, 
            class_names
        )
    except Exception as e:
        print(f"\nError loading test dataset: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Ensure test directory exists: data/processed/test/")
        print("  2. Check that test directory contains class subdirectories")
        print("  3. Verify images are in correct format (jpg, png)")
        sys.exit(1)
    
    # Initialize evaluator
    try:
        evaluator = ComprehensiveEvaluator(
            model_path=str(model_path),
            class_names=class_names,
            output_dir=str(output_dir)
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Check that model file exists and is not corrupted")
        print("  2. Ensure model was saved with TensorFlow/Keras")
        print("  3. Verify model architecture is compatible")
        sys.exit(1)
    
    # Load training history if provided
    training_history = None
    if args.history:
        history_path = Path(args.history)
        if history_path.exists():
            print(f"\nLoading training history from: {history_path}")
            with open(history_path, 'r') as f:
                data = json.load(f)
                # Extract history from different possible structures
                if 'stage1_history' in data:
                    training_history = data['stage1_history']
                elif 'history' in data:
                    training_history = data['history']
                else:
                    training_history = data
            print("✓ Training history loaded")
        else:
            print(f"\nWarning: Training history file not found: {history_path}")
            print("Continuing without training curves...")
    
    # Run evaluation
    print("\nStarting evaluation...")
    print("This may take several minutes depending on dataset size.")
    
    try:
        results = evaluator.evaluate(
            test_dataset=test_dataset,
            training_history=training_history,
            generate_gradcam=not args.no_gradcam
        )
        
        # Print final summary
        print("\n" + "="*70)
        print("✓ EVALUATION COMPLETED SUCCESSFULLY")
        print("="*70)
        
        metrics = results['metrics']
        
        print(f"\nKey Metrics:")
        print(f"  • Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
        
        if 'weighted_metrics' in metrics:
            wm = metrics['weighted_metrics']
            print(f"  • Precision (weighted): {wm['precision']:.4f}")
            print(f"  • Recall (weighted): {wm['recall']:.4f}")
            print(f"  • F1-Score (weighted): {wm['f1_score']:.4f}")
        
        if 'roc_auc_macro' in metrics:
            print(f"  • ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
        
        print(f"\nPer-Class Performance:")
        per_class = metrics.get('per_class_metrics', {})
        for class_name in class_names:
            if class_name in per_class:
                m = per_class[class_name]
                print(f"  • {class_name}:")
                print(f"    - Precision: {m['precision']:.4f}")
                print(f"    - Recall: {m['recall']:.4f}")
                print(f"    - F1-Score: {m['f1_score']:.4f}")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"   View plots: {output_dir / 'plots'}")
        if not args.no_gradcam:
            print(f"   View Grad-CAM: {output_dir / 'gradcam'}")
        
        timestamp = results["timestamp"]
        print(f"   Detailed report: {output_dir / f'evaluation_report_{timestamp}.txt'}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        print("\nEvaluation failed. Check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
