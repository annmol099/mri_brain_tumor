"""
Training Callbacks for Brain Tumor MRI Classification
Phase 2.4: Training Callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
"""

import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import time
from datetime import datetime


class CustomCallbacks:
    """Factory class for creating training callbacks."""
    
    def __init__(self, model_dir='../models', log_dir='../logs'):
        """
        Initialize callbacks factory.
        
        Args:
            model_dir (str): Directory to save models
            log_dir (str): Directory for logs
        """
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_early_stopping(self, monitor='val_loss', patience=5, 
                          restore_best_weights=True, verbose=1):
        """
        Create EarlyStopping callback.
        
        Args:
            monitor (str): Metric to monitor
            patience (int): Number of epochs with no improvement after which training stops
            restore_best_weights (bool): Restore model weights from best epoch
            verbose (int): Verbosity mode
            
        Returns:
            keras.callbacks.EarlyStopping
        """
        return keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=restore_best_weights,
            verbose=verbose,
            mode='auto'
        )
    
    def get_reduce_lr(self, monitor='val_loss', factor=0.5, patience=3, 
                     min_lr=1e-7, verbose=1):
        """
        Create ReduceLROnPlateau callback.
        
        Args:
            monitor (str): Metric to monitor
            factor (float): Factor by which learning rate will be reduced
            patience (int): Number of epochs with no improvement after which LR is reduced
            min_lr (float): Lower bound on the learning rate
            verbose (int): Verbosity mode
            
        Returns:
            keras.callbacks.ReduceLROnPlateau
        """
        return keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=verbose,
            mode='auto'
        )
    
    def get_model_checkpoint(self, stage='stage1', monitor='val_accuracy', 
                            save_best_only=True, verbose=1):
        """
        Create ModelCheckpoint callback.
        
        Args:
            stage (str): Training stage name for filename
            monitor (str): Metric to monitor
            save_best_only (bool): Only save when the model is better
            verbose (int): Verbosity mode
            
        Returns:
            keras.callbacks.ModelCheckpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.model_dir / f'best_model_{stage}_{timestamp}.h5'
        
        return keras.callbacks.ModelCheckpoint(
            filepath=str(filepath),
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            verbose=verbose,
            mode='auto'
        )
    
    def get_tensorboard(self, experiment_name='experiment'):
        """
        Create TensorBoard callback.
        
        Args:
            experiment_name (str): Name of the experiment
            
        Returns:
            keras.callbacks.TensorBoard
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f'{experiment_name}_{timestamp}'
        
        return keras.callbacks.TensorBoard(
            log_dir=str(log_path),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )
    
    def get_csv_logger(self, stage='stage1'):
        """
        Create CSVLogger callback.
        
        Args:
            stage (str): Training stage name for filename
            
        Returns:
            keras.callbacks.CSVLogger
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.log_dir / f'training_log_{stage}_{timestamp}.csv'
        
        return keras.callbacks.CSVLogger(
            filename=str(filepath),
            separator=',',
            append=False
        )
    
    def get_default_callbacks(self, stage='stage1', use_tensorboard=True):
        """
        Get a standard set of callbacks for training.
        
        Args:
            stage (str): Training stage name
            use_tensorboard (bool): Whether to include TensorBoard
            
        Returns:
            list: List of callbacks
        """
        callbacks = [
            self.get_early_stopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            self.get_reduce_lr(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            self.get_model_checkpoint(
                stage=stage,
                monitor='val_accuracy',
                save_best_only=True
            ),
            self.get_csv_logger(stage=stage)
        ]
        
        if use_tensorboard:
            callbacks.append(self.get_tensorboard(experiment_name=stage))
        
        return callbacks


class TrainingMonitor(keras.callbacks.Callback):
    """Custom callback to monitor training progress and log metrics."""
    
    def __init__(self, log_dir='../logs'):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_times = []
        self.start_time = None
        self.metrics_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self.start_time = time.time()
        print("\n" + "="*60)
        print("Training Started")
        print("="*60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        self.epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        # Store metrics
        for key in self.metrics_history.keys():
            if logs and key in logs:
                self.metrics_history[key].append(logs[key])
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
        if logs:
            print(f"  Loss: {logs.get('loss', 0):.4f}")
            print(f"  Accuracy: {logs.get('accuracy', 0):.4f}")
            print(f"  Val Loss: {logs.get('val_loss', 0):.4f}")
            print(f"  Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
            if 'lr' in logs:
                print(f"  Learning Rate: {logs['lr']:.6f}")
    
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        total_time = time.time() - self.start_time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
        
        print("\n" + "="*60)
        print("Training Completed")
        print("="*60)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Average epoch time: {avg_epoch_time:.2f}s")
        print(f"Total epochs: {len(self.epoch_times)}")
        
        # Save metrics history
        self.save_metrics_history()
    
    def save_metrics_history(self):
        """Save metrics history to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.log_dir / f'metrics_history_{timestamp}.json'
        
        history_data = {
            'metrics': self.metrics_history,
            'epoch_times': self.epoch_times,
            'total_epochs': len(self.epoch_times),
            'average_epoch_time': sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Metrics history saved to: {filepath}")


class LearningRateLogger(keras.callbacks.Callback):
    """Callback to log learning rate at each epoch."""
    
    def on_epoch_end(self, epoch, logs=None):
        """Log learning rate."""
        if logs is not None:
            lr = float(keras.backend.get_value(self.model.optimizer.lr))
            logs['lr'] = lr


def get_callbacks_for_stage(stage=1, model_dir='../models', log_dir='../logs', 
                            use_tensorboard=True):
    """
    Get appropriate callbacks for training stage.
    
    Args:
        stage (int): Training stage (1 or 2)
        model_dir (str): Directory to save models
        log_dir (str): Directory for logs
        use_tensorboard (bool): Whether to use TensorBoard
        
    Returns:
        list: List of callbacks
    """
    callback_factory = CustomCallbacks(model_dir=model_dir, log_dir=log_dir)
    
    stage_name = f'stage{stage}'
    
    if stage == 1:
        # Stage 1: Train custom head with frozen base
        callbacks = callback_factory.get_default_callbacks(
            stage=stage_name,
            use_tensorboard=use_tensorboard
        )
    else:
        # Stage 2: Fine-tuning with lower learning rate
        callbacks = [
            callback_factory.get_early_stopping(
                monitor='val_loss',
                patience=7,  # More patience for fine-tuning
                restore_best_weights=True
            ),
            callback_factory.get_reduce_lr(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-8  # Lower minimum LR
            ),
            callback_factory.get_model_checkpoint(
                stage=stage_name,
                monitor='val_accuracy',
                save_best_only=True
            ),
            callback_factory.get_csv_logger(stage=stage_name)
        ]
        
        if use_tensorboard:
            callbacks.append(callback_factory.get_tensorboard(experiment_name=stage_name))
    
    # Add custom callbacks
    callbacks.append(TrainingMonitor(log_dir=log_dir))
    callbacks.append(LearningRateLogger())
    
    return callbacks


if __name__ == "__main__":
    print("Training Callbacks Module")
    print("="*60)
    
    # Create callback factory
    callback_factory = CustomCallbacks(model_dir='../models', log_dir='../logs')
    
    # Stage 1 callbacks
    print("\nStage 1 Callbacks:")
    stage1_callbacks = get_callbacks_for_stage(stage=1)
    for i, callback in enumerate(stage1_callbacks, 1):
        print(f"  {i}. {callback.__class__.__name__}")
    
    # Stage 2 callbacks
    print("\nStage 2 Callbacks:")
    stage2_callbacks = get_callbacks_for_stage(stage=2)
    for i, callback in enumerate(stage2_callbacks, 1):
        print(f"  {i}. {callback.__class__.__name__}")
    
    print("\nCallbacks configured successfully!")
