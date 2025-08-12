"""
Advanced Training Utilities with Multi-threading and GPU Optimization

This module provides comprehensive training utilities including callbacks,
metrics tracking, and GPU memory management.
"""

import sys
import tensorflow as tf
from tensorflow.keras import callbacks as keras_callbacks
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, AdamW
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import (
    CategoricalAccuracy, Precision, Recall, AUC,
    TruePositives, TrueNegatives, FalsePositives, FalseNegatives
)
import numpy as np
from pathlib import Path
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import psutil
import GPUtil
import threading
import queue
import os

def _json_safe(obj):
    import numpy as np
    from datetime import datetime
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    return str(obj)


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-7
    loss_function: str = "categorical_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "auc"])
    save_best_only: bool = True
    save_weights_only: bool = False
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    verbose: int = 1
    use_mixed_precision: bool = False
    use_tensorboard: bool = True
    use_lr_scheduler: bool = True
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    class_weights: Optional[Dict[int, float]] = None
    gradient_clip_norm: Optional[float] = 1.0
    accumulation_steps: int = 1
    distributed_strategy: Optional[str] = None  # "mirrored", "multi_worker_mirrored"
    num_gpus: int = -1  # -1 for all available
    l2_regularization: float = 1e-2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items()}


class GPUManager:
    """Manage GPU resources and memory"""
    
    @staticmethod
    def setup_gpu_memory_growth():
        """Enable GPU memory growth to prevent OOM"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Enabled memory growth for {len(gpus)} GPUs")
            except RuntimeError as e:
                logging.error(f"GPU setup error: {e}")
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get current GPU information"""
        gpus = GPUtil.getGPUs()
        if not gpus:
            return {"available": False, "count": 0}
        
        gpu_info = {
            "available": True,
            "count": len(gpus),
            "devices": []
        }
        
        for gpu in gpus:
            gpu_info["devices"].append({
                "id": gpu.id,
                "name": gpu.name,
                "memory_total": gpu.memoryTotal,
                "memory_free": gpu.memoryFree,
                "memory_used": gpu.memoryUsed,
                "temperature": gpu.temperature,
                "load": gpu.load * 100
            })
        
        return gpu_info
    
    @staticmethod
    def set_mixed_precision(enabled: bool = True):
        """Enable or disable mixed precision training"""
        if enabled:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("Mixed precision training enabled (float16)")
        else:
            policy = tf.keras.mixed_precision.Policy('float32')
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("Mixed precision training disabled (float32)")


class CustomCallbacks:
    """Collection of custom callbacks for training"""
    
    @staticmethod
    def get_time_callback() -> keras_callbacks.Callback:
        """Callback to track epoch training time"""
        class TimeCallback(keras_callbacks.Callback):
            def __init__(self):
                self.times = []
                self.epoch_time = 0
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_time = time.time()
            
            def on_epoch_end(self, epoch, logs=None):
                self.times.append(time.time() - self.epoch_time)
                if logs is not None:
                    logs['time'] = self.times[-1]
        
        return TimeCallback()
    
    @staticmethod
    def get_memory_callback() -> keras_callbacks.Callback:
        """Callback to track memory usage"""
        class MemoryCallback(keras_callbacks.Callback):
            def __init__(self):
                self.memory_usage = []
            
            def on_epoch_end(self, epoch, logs=None):
                # CPU memory
                cpu_percent = psutil.virtual_memory().percent
                
                # GPU memory
                gpus = GPUtil.getGPUs()
                gpu_memory = gpus[0].memoryUtil * 100 if gpus else 0
                
                self.memory_usage.append({
                    'epoch': epoch,
                    'cpu_memory': cpu_percent,
                    'gpu_memory': gpu_memory
                })
                
                if logs is not None:
                    logs['cpu_memory'] = cpu_percent
                    logs['gpu_memory'] = gpu_memory
        
        return MemoryCallback()
    
    @staticmethod
    def get_lr_warmup_callback(warmup_epochs: int, initial_lr: float) -> keras_callbacks.Callback:
        """Learning rate warmup callback"""
        class LRWarmupCallback(keras_callbacks.Callback):
            def __init__(self, warmup_epochs, initial_lr):
                self.warmup_epochs = warmup_epochs
                self.initial_lr = initial_lr
                self.warmup_steps = 0
            
            def on_epoch_begin(self, epoch, logs=None):
                if epoch < self.warmup_epochs:
                    lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
                    if logs is not None:
                        logs['lr'] = lr
        
        return LRWarmupCallback(warmup_epochs, initial_lr)
    
    @staticmethod
    def get_best_model_callback(filepath: Path, monitor: str = 'val_loss') -> keras_callbacks.Callback:
        """Enhanced ModelCheckpoint with additional features"""
        class BestModelCallback(keras_callbacks.ModelCheckpoint):
            def __init__(self, filepath, monitor='val_loss', **kwargs):
                super().__init__(str(filepath), monitor=monitor, **kwargs)
                self.best_weights = None
                self.best_epoch = 0
            
            def on_epoch_end(self, epoch, logs=None):
                super().on_epoch_end(epoch, logs)
                current = logs.get(self.monitor)
                if current is not None:
                    if self.monitor_op(current, self.best):
                        self.best_epoch = epoch
                        self.best_weights = self.model.get_weights()
        
        return BestModelCallback(
            filepath,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode='min' if 'loss' in monitor else 'max',
            verbose=1
        )


class MetricsTracker:
    """Track and visualize training metrics"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = {}
        self.training_start = None
        self.training_end = None
   
    def update(self, epoch: int, logs: Dict[str, float]):
        """Update metrics for current epoch"""
        for metric, value in logs.items():
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(value)
    
    def save_history(self, model_name: str, dataset_name: str):
        """Save training history to JSON"""
        history_data = {
            "model": model_name,
            "dataset": dataset_name,
            "training_start": self.training_start.isoformat() if self.training_start else None,
            "training_end": self.training_end.isoformat() if self.training_end else None,
            "duration": str(self.training_end - self.training_start) if self.training_end and self.training_start else None,
            "epochs": len(self.history.get("loss", [])),
            "history": self.history
        }
        
        filename = f"{model_name}_{dataset_name}_history.json"
        with open(self.save_dir / filename, "w") as f:
            json.dump(history_data, f, indent=2, default=_json_safe)
    
    def plot_metrics(self, model_name: str, dataset_name: str):
        """Create comprehensive visualization of training metrics"""
        if not self.history:
            return
        
        # Determine number of subplots needed
        metrics = list(self.history.keys())
        n_metrics = len([m for m in metrics if not m.startswith('val_') and m not in ['time', 'lr', 'cpu_memory', 'gpu_memory']])
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            nrows=(n_metrics + 1) // 2,
            ncols=2,
            figsize=(15, 5 * ((n_metrics + 1) // 2))
        )
        
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each metric
        plot_idx = 0
        for metric in metrics:
            if metric.startswith('val_') or metric in ['time', 'lr', 'cpu_memory', 'gpu_memory']:
                continue
            
            ax = axes[plot_idx]
            
            # Plot training metric
            epochs = range(1, len(self.history[metric]) + 1)
            ax.plot(epochs, self.history[metric], 'b-', label=f'Training {metric}')
            
            # Plot validation metric if exists
            val_metric = f'val_{metric}'
            if val_metric in self.history:
                ax.plot(epochs, self.history[val_metric], 'r-', label=f'Validation {metric}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Remove empty subplots
        for idx in range(plot_idx, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'{model_name} on {dataset_name} - Training History', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / f"{model_name}_{dataset_name}_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional plots for special metrics
        self._plot_special_metrics(model_name, dataset_name)
    
    def _plot_special_metrics(self, model_name: str, dataset_name: str):
        """Plot special metrics like learning rate, memory usage, etc."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Learning rate
        if 'lr' in self.history:
            ax = axes[0]
            epochs = range(1, len(self.history['lr']) + 1)
            ax.plot(epochs, self.history['lr'], 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Training time per epoch
        if 'time' in self.history:
            ax = axes[1]
            epochs = range(1, len(self.history['time']) + 1)
            ax.bar(epochs, self.history['time'], color='skyblue')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Training Time per Epoch')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Memory usage
        if 'cpu_memory' in self.history and 'gpu_memory' in self.history:
            ax = axes[2]
            epochs = range(1, len(self.history['cpu_memory']) + 1)
            ax.plot(epochs, self.history['cpu_memory'], 'b-', label='CPU Memory %')
            ax.plot(epochs, self.history['gpu_memory'], 'r-', label='GPU Memory %')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Memory Usage (%)')
            ax.set_title('Memory Usage During Training')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Loss comparison
        if 'loss' in self.history and 'val_loss' in self.history:
            ax = axes[3]
            epochs = range(1, len(self.history['loss']) + 1)
            ax.plot(epochs, self.history['loss'], 'b-', label='Training Loss')
            ax.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
            
            # Mark best epoch
            best_epoch = np.argmin(self.history['val_loss'])
            ax.axvline(x=best_epoch + 1, color='green', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch + 1})')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} on {dataset_name} - Additional Metrics', fontsize=16)
        plt.tight_layout()
        
        plot_path = self.save_dir / f"{model_name}_{dataset_name}_special_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


class ModelTrainer:
    """Advanced model trainer with multi-threading support"""
    
    def __init__(self, config: TrainingConfig, save_dir: Path = Path("results")):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
        # Setup GPU
        GPUManager.setup_gpu_memory_growth()
        if self.config.use_mixed_precision:
            GPUManager.set_mixed_precision(True)
        
        # Setup distributed strategy if needed
        self.strategy = self._get_distribution_strategy()
        
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.save_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(
            log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
    
    def _get_distribution_strategy(self) -> Optional[tf.distribute.Strategy]:
        """Get distribution strategy based on configuration"""
        if self.config.distributed_strategy is None:
            return None
        
        if self.config.distributed_strategy == "mirrored":
            # Get available GPUs
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if len(gpus) > 1:
                if self.config.num_gpus > 0:
                    # Use specified number of GPUs
                    devices = [f"/gpu:{i}" for i in range(min(self.config.num_gpus, len(gpus)))]
                else:
                    # Use all available GPUs
                    devices = [f"/gpu:{i}" for i in range(len(gpus))]
                
                strategy = tf.distribute.MirroredStrategy(devices=devices)
                self.logger.info(f"Using MirroredStrategy with {len(devices)} GPUs")
                return strategy
            else:
                self.logger.warning("MirroredStrategy requested but only 1 GPU available")
                return None
        
        return None
    
    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Compile model with specified configuration"""
        # Get optimizer
        optimizer = self._get_optimizer()
        
        # Get loss function
        loss = self._get_loss_function()
        
        # Get metrics
        metrics = self._get_metrics()
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Get optimizer based on configuration"""
        lr = self.config.learning_rate
        
        if self.config.optimizer.lower() == "adam":
            return Adam(learning_rate=lr, clipnorm=self.config.gradient_clip_norm)
        elif self.config.optimizer.lower() == "adamw":
            return AdamW(learning_rate=lr, weight_decay=0.01, clipnorm=self.config.gradient_clip_norm)
        elif self.config.optimizer.lower() == "sgd":
            return SGD(learning_rate=lr, momentum=0.9, nesterov=True, clipnorm=self.config.gradient_clip_norm)
        elif self.config.optimizer.lower() == "rmsprop":
            return RMSprop(learning_rate=lr, clipnorm=self.config.gradient_clip_norm)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _get_loss_function(self):
        """Get loss function based on configuration"""
        if self.config.loss_function == "categorical_crossentropy":
            return CategoricalCrossentropy(
                label_smoothing=self.config.label_smoothing,
                from_logits=False
            )
        elif self.config.loss_function == "binary_crossentropy":
            return BinaryCrossentropy(
                label_smoothing=self.config.label_smoothing,
                from_logits=False
            )
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")
    
    def _get_metrics(self) -> List[tf.keras.metrics.Metric]:
        """Get metrics based on configuration"""
        metrics = []
        
        for metric_name in self.config.metrics:
            if metric_name == "accuracy":
                metrics.append(CategoricalAccuracy(name="accuracy"))
            elif metric_name == "precision":
                metrics.append(Precision(name="precision"))
            elif metric_name == "recall":
                metrics.append(Recall(name="recall"))
            elif metric_name == "auc":
                metrics.append(AUC(name="auc"))
            elif metric_name == "tp":
                metrics.append(TruePositives(name="tp"))
            elif metric_name == "tn":
                metrics.append(TrueNegatives(name="tn"))
            elif metric_name == "fp":
                metrics.append(FalsePositives(name="fp"))
            elif metric_name == "fn":
                metrics.append(FalseNegatives(name="fn"))
        
        return metrics
    
    def _get_callbacks(self, model_name: str, dataset_name: str) -> List[keras_callbacks.Callback]:
        """Get all callbacks for training"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_dir = self.save_dir / "checkpoints" / f"{model_name}_{dataset_name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "best_model.h5"
        callbacks.append(
            CustomCallbacks.get_best_model_callback(
                checkpoint_path,
                monitor=self.config.monitor_metric
            )
        )
        
        # Early stopping
        callbacks.append(
            keras_callbacks.EarlyStopping(
                monitor=self.config.monitor_metric,
                patience=self.config.early_stopping_patience,
                mode=self.config.monitor_mode,
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Learning rate reduction
        if self.config.use_lr_scheduler:
            callbacks.append(
                keras_callbacks.ReduceLROnPlateau(
                    monitor=self.config.monitor_metric,
                    factor=self.config.reduce_lr_factor,
                    patience=self.config.reduce_lr_patience,
                    min_lr=self.config.min_learning_rate,
                    mode=self.config.monitor_mode,
                    verbose=1
                )
            )
        
        # TensorBoard
        if self.config.use_tensorboard:
            tb_dir = self.save_dir / "tensorboard" / f"{model_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            callbacks.append(
                keras_callbacks.TensorBoard(
                    log_dir=str(tb_dir),
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch',
                    profile_batch=2
                )
            )
        
        # Custom callbacks
        callbacks.append(CustomCallbacks.get_time_callback())
        callbacks.append(CustomCallbacks.get_memory_callback())
        
        # Learning rate warmup
        if self.config.warmup_epochs > 0:
            callbacks.append(
                CustomCallbacks.get_lr_warmup_callback(
                    self.config.warmup_epochs,
                    self.config.learning_rate
                )
            )
        
        # CSV logger
        csv_path = self.save_dir / "logs" / f"{model_name}_{dataset_name}_training.csv"
        csv_path.parent.mkdir(exist_ok=True)
        callbacks.append(
            keras_callbacks.CSVLogger(str(csv_path), append=False)
        )
        
        return callbacks
    
    def train(self, model: tf.keras.Model, train_dataset: tf.data.Dataset,
              val_dataset: tf.data.Dataset, model_name: str, dataset_name: str,
              class_weights: Optional[Dict[int, float]] = None) -> Tuple[tf.keras.Model, Dict[str, Any]]:
        """
        Train model with advanced features
        
        Args:
            model: Keras model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            model_name: Name of the model
            dataset_name: Name of the dataset
            class_weights: Optional class weights for imbalanced datasets
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        self.logger.info(f"Starting training: {model_name} on {dataset_name}")
        self.logger.info(f"Configuration: {self.config.to_dict()}")
        
        
        # Get GPU info
        gpu_info = GPUManager.get_gpu_info()
        self.logger.info(f"GPU info: {gpu_info}")
        
        # Compile model
        if self.strategy:
            with self.strategy.scope():
                model = self.compile_model(model)
        else:
            model = self.compile_model(model)
        
        # Setup metrics tracker
        metrics_tracker = MetricsTracker(
            self.save_dir / "metrics" / f"{model_name}_{dataset_name}"
        )
        metrics_tracker.training_start = datetime.now()
        
        # Get callbacks
        callbacks = self._get_callbacks(model_name, dataset_name)
        
        # Add metrics tracking callback
        class MetricsCallback(keras_callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    metrics_tracker.update(epoch, logs)
        
        callbacks.append(MetricsCallback())
        
        # Train model
        try:
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=self.config.verbose,
                class_weight=class_weights or self.config.class_weights
            )
            
            metrics_tracker.training_end = datetime.now()
            
            # Save final model
            final_model_path = self.save_dir / "models" / f"{model_name}_{dataset_name}_final.h5"
            final_model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(final_model_path))
            
            # Save training history
            metrics_tracker.save_history(model_name, dataset_name)
            
            # Plot metrics
            metrics_tracker.plot_metrics(model_name, dataset_name)
            
            # Save training summary
            self._save_training_summary(
                model, model_name, dataset_name, history, metrics_tracker
            )
            
            self.logger.info(f"Training completed: {model_name} on {dataset_name}")
            
            return model, history.history
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _save_training_summary(self, model: tf.keras.Model, model_name: str,
                              dataset_name: str, history: tf.keras.callbacks.History,
                              metrics_tracker: MetricsTracker):
        """Save comprehensive training summary"""
        summary = {
            "model": model_name,
            "dataset": dataset_name,
            "configuration": self.config.to_dict(),
            "model_parameters": model.count_params(),
            "training_start": metrics_tracker.training_start.isoformat(),
            "training_end": metrics_tracker.training_end.isoformat(),
            "total_duration": str(metrics_tracker.training_end - metrics_tracker.training_start),
            "epochs_trained": len(history.history['loss']),
            "best_epoch": int(np.argmin(history.history[self.config.monitor_metric])) + 1,
            "best_val_loss": float(min(history.history.get('val_loss', [np.inf]))),
            "best_val_accuracy": float(max(history.history.get('val_accuracy', [0]))),
            "final_metrics": {
                metric: float(values[-1]) 
                for metric, values in history.history.items()
            },
            "gpu_info": GPUManager.get_gpu_info(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "python_version": sys.version
            }
        }
        
        summary_path = self.save_dir / "summaries" / f"{model_name}_{dataset_name}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=_json_safe)
        
        self.logger.info(f"Training summary saved to {summary_path}")


class ParallelTrainer:
    """Train multiple models in parallel using threading"""
    
    def __init__(self, num_workers: int = 2):
        self.num_workers = num_workers
        self.queue = queue.Queue()
        self.results = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_training_job(self, job_id: str, model_fn, train_dataset: tf.data.Dataset,
                        val_dataset: tf.data.Dataset, model_name: str,
                        dataset_name: str, config: TrainingConfig,
                        save_dir: Path, class_weights: Optional[Dict[int, float]] = None):
        """Add a training job to the queue"""
        self.queue.put({
            "job_id": job_id,
            "model_fn": model_fn,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "config": config,
            "save_dir": save_dir,
            "class_weights": class_weights
        })
    
    def _worker(self):
        """Worker thread for training"""
        while True:
            job = self.queue.get()
            if job is None:
                break
            
            try:
                self.logger.info(f"Starting job: {job['job_id']}")
                
                # Create model
                model = job['model_fn']()
                
                # Create trainer
                trainer = ModelTrainer(job['config'], job['save_dir'])
                
                # Train model
                trained_model, history = trainer.train(
                    model,
                    job['train_dataset'],
                    job['val_dataset'],
                    job['model_name'],
                    job['dataset_name'],
                    job['class_weights']
                )
                
                # Store results
                self.results[job['job_id']] = {
                    "success": True,
                    "model": trained_model,
                    "history": history
                }
                
                self.logger.info(f"Completed job: {job['job_id']}")
                
            except Exception as e:
                self.logger.error(f"Job failed: {job['job_id']} - {e}")
                self.results[job['job_id']] = {
                    "success": False,
                    "error": str(e)
                }
            
            finally:
                self.queue.task_done()
    
    def run_all(self) -> Dict[str, Any]:
        """Run all training jobs in parallel"""
        # Start worker threads
        threads = []
        for _ in range(self.num_workers):
            t = threading.Thread(target=self._worker)
            t.start()
            threads.append(t)
        
        # Wait for all jobs to complete
        self.queue.join()
        
        # Stop workers
        for _ in range(self.num_workers):
            self.queue.put(None)
        
        for t in threads:
            t.join()
        
        return self.results


# Helper functions
def set_gpu_memory_growth():
    """Enable GPU memory growth"""
    GPUManager.setup_gpu_memory_growth()


def set_mixed_precision(enabled: bool = True):
    """Enable/disable mixed precision training"""
    GPUManager.set_mixed_precision(enabled)


def get_default_config() -> TrainingConfig:
    """Get default training configuration"""
    return TrainingConfig()


def create_trainer(config: Optional[TrainingConfig] = None, save_dir: Path = Path("results")) -> ModelTrainer:
    """Create a model trainer instance"""
    if config is None:
        config = get_default_config()
    return ModelTrainer(config, save_dir)
