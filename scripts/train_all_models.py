#!/usr/bin/env python3
"""
Train All Models Script with Multi-threading and Hyperparameter Optimization

This script trains all model architectures on all datasets with support for:
- Multi-threaded parallel training
- Hyperparameter optimization using Optuna
- Selective model/dataset training
- Comprehensive logging and visualization
"""

import sys
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
import optuna
from optuna.trial import Trial
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import gc

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.architectures import create_model, get_all_architectures
from src.dataset_loader import DatasetLoader, DatasetSource, DatasetConfig
from src.training_utils import (
    ModelTrainer, TrainingConfig, ParallelTrainer,
    set_gpu_memory_growth, set_mixed_precision
)
from config.model_configs import get_model_config, get_all_model_configs
from config.dataset_configs import get_dataset_config, DatasetType

from src.training_utils import _json_safe

class OptunaOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, n_trials: int = 20, n_jobs: int = 1):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_objective(self, model_name: str, dataset_name: str,
                        train_dataset: tf.data.Dataset,
                        val_dataset: tf.data.Dataset,
                        class_weights: Optional[Dict[int, float]] = None):
        """Create Optuna objective function"""
        
        def objective(trial: Trial) -> float:
            # Suggest hyperparameters
            suggested_batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            config = TrainingConfig(
                learning_rate=trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
                optimizer=trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
                dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.7),
                label_smoothing=trial.suggest_float('label_smoothing', 0.0, 0.2),
                epochs=30,  # Fixed for optimization
                early_stopping_patience=7,
                use_mixed_precision=trial.suggest_categorical('mixed_precision', [True, False]),
                warmup_epochs=trial.suggest_int('warmup_epochs', 0, 5),
            )
            
            # Note: batch_size affects the dataset, not just training config
            # For simplicity, we'll use the suggested batch_size in evaluation
            # In production, you'd want to rebatch the dataset
            
            # Create model with suggested architecture params
            if model_name.startswith('AttackNet'):
                input_shape = (256, 256, 3)
                if 'V1' in model_name:
                    leaky_alpha = trial.suggest_float('leaky_relu_alpha', 0.1, 0.3)
                    dense_units = trial.suggest_categorical('dense_units', [64, 128, 256])
            else:
                input_shape = (256, 256, 3)
            
            # Create and train model
            try:
                model = create_model(model_name, input_shape, dropout_rate=model_config.dropout_rate, l2_reg=config.l2_regularization)
                model_instance = model.get_model()
                
                # Create trainer
                trainer = ModelTrainer(config, save_dir=Path(f"results/optuna/{trial.number}"))
                
                # Train
                _, history = trainer.train(
                    model_instance,
                    train_dataset,
                    val_dataset,
                    model_name,
                    dataset_name,
                    class_weights
                )
                
                # Return best validation loss
                best_val_loss = min(history['val_loss'])
                
                # Clean up
                tf.keras.backend.clear_session()
                del model_instance
                gc.collect()
                
                return best_val_loss
                
            except Exception as e:
                self.logger.error(f"Trial failed: {e}")
                return float('inf')
        
        return objective
    
    def optimize(self, model_name: str, dataset_name: str,
                train_dataset: tf.data.Dataset,
                val_dataset: tf.data.Dataset,
                class_weights: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        self.logger.info(f"Starting Optuna optimization for {model_name} on {dataset_name}")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=f"{model_name}_{dataset_name}",
            load_if_exists=True
        )
        
        # Create objective
        objective = self.create_objective(
            model_name, dataset_name, train_dataset, val_dataset, class_weights
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            gc_after_trial=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"Optimization complete. Best value: {best_value}")
        self.logger.info(f"Best parameters: {best_params}")
        
        # Save optimization results
        results = {
            "model": model_name,
            "dataset": dataset_name,
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
            "optimization_history": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": str(trial.state)
                }
                for trial in study.trials
            ]
        }
        
        return results


class ModelTrainingOrchestrator:
    """Orchestrate training of multiple models on multiple datasets"""
    
    def _log_training_info(self, model_config, dataset_config, train_ds, val_ds, model_name, dataset_name):
        """Log comprehensive training information"""
        
        self.logger.info(f"\n{'='*40}")
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info(f"{'='*40}")
        
        # Model info
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Dataset: {dataset_name}")
        
        # Batch size info
        configured_batch = model_config.batch_size
        
        # Get actual batch sizes
        for train_batch in train_ds.take(1):
            actual_train_batch = train_batch[0].shape[0] 
            break
        
        for val_batch in val_ds.take(1):
            actual_val_batch = val_batch[0].shape[0]
            break
            
        self.logger.info(f"\nBatch sizes:")
        self.logger.info(f"  Configured: {configured_batch}")
        self.logger.info(f"  Train actual: {actual_train_batch}")
        self.logger.info(f"  Val actual: {actual_val_batch}")
        
        # Training params
        self.logger.info(f"\nTraining parameters:")
        self.logger.info(f"  Learning rate: {model_config.learning_rate}")
        self.logger.info(f"  Optimizer: {model_config.optimizer}")
        self.logger.info(f"  Epochs: {model_config.epochs}")
        self.logger.info(f"  Early stopping patience: {model_config.early_stopping_patience}")
        self.logger.info(f"  Dropout_rate: {model_config.dropout_rate}")
        self.logger.info(f"  L2 regularization: {model_config.l2_regularization}")
        
        # Data info
        self.logger.info(f"\nData configuration:")
        self.logger.info(f"  Augmentation: {dataset_config.augment}")
        self.logger.info(f"  Augmentation level: {dataset_config.augmentation_level}")
        self.logger.info(f"  Validation split: {dataset_config.validation_split}")
        
        self.logger.info(f"{'='*40}\n")    
    
    def __init__(self, results_dir: Path = Path("results"), num_threads: int = 2):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
        # Setup GPU
        set_gpu_memory_growth()
        
        # Training results
        self.training_results = {}
        self.optimization_results = {}
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.results_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        if self.logger.handlers:
            return 
        
        # File handler
        fh = logging.FileHandler(
            log_dir / f"training_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.DEBUG)

        self.logger.propagate = False
    
    def train_single_combination(self, model_name: str, dataset_name: str,
                               optimize_hyperparams: bool = False,
                               optuna_trials: int = 20) -> Dict[str, Any]:
        """Train a single model-dataset combination"""
                
        model_config = get_model_config(model_name, dataset_name)
        
        dataset_config = DatasetConfig(
            batch_size=model_config.batch_size,
            augment=False,
            augmentation_level="light",
            validation_split=model_config.validation_split,
            mixed_precision=False
        )

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Training {model_name} on {dataset_name}")
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Model batch size: {model_config.batch_size}")
        self.logger.info(f"  Dataset batch size: {dataset_config.batch_size}")
        self.logger.info(f"  Learning rate: {model_config.learning_rate}")
        self.logger.info(f"  Epochs: {model_config.epochs}")
        self.logger.info(f"  Augmentation: {dataset_config.augment}")
        self.logger.info(f"{'='*60}")

        # Load dataset
        self.logger.info("Loading dataset...")
        loader = DatasetLoader(dataset_config)
        
        try:
            train_ds, val_ds, test_ds, data_info = loader.load_dataset(
                dataset_name, ensure_no_leak=True
            )
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return {"success": False, "error": str(e)}
        
        self._log_training_info(model_config, dataset_config, train_ds, val_ds, model_name, dataset_name)

        # Get class weights
        class_weights = data_info.get('class_weights')
        
        # Hyperparameter optimization if requested
        if optimize_hyperparams:
            self.logger.info("Running hyperparameter optimization...")
            optimizer = OptunaOptimizer(n_trials=optuna_trials)
            opt_results = optimizer.optimize(
                model_name, dataset_name, train_ds, val_ds, class_weights
            )
            
            # Save optimization results
            opt_path = self.results_dir / "optimization" / f"{model_name}_{dataset_name}_optuna.json"
            opt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(opt_path, "w") as f:
                json.dump(opt_results, f, indent=2, default=_json_safe)
            
            # Use best parameters for training
            best_params = opt_results['best_params']
            config = TrainingConfig(
                learning_rate=best_params.get('learning_rate', 1e-4),
                optimizer=best_params.get('optimizer', 'adam'),
                epochs=50,  # Full training
                use_mixed_precision=best_params.get('mixed_precision', False),
                warmup_epochs=best_params.get('warmup_epochs', 0),
                label_smoothing=best_params.get('label_smoothing', 0.0)
            )
        else:
            # Use default configuration
            #model_config = get_model_config(model_name)
            #model_config = get_model_config(model_name, dataset_name)
            config = TrainingConfig(
                learning_rate=model_config.learning_rate,
                optimizer=model_config.optimizer,
                epochs=model_config.epochs,
                early_stopping_patience=model_config.early_stopping_patience,
                reduce_lr_patience=model_config.reduce_lr_patience,
                use_mixed_precision=False
            )
        
        # Create model
        self.logger.info("Creating model...")
        model = create_model(model_name, data_info['input_shape'], dropout_rate=model_config.dropout_rate, l2_reg=config.l2_regularization)
        model_instance = model.get_model()
        
        # Create trainer
        trainer = ModelTrainer(
            config,
            save_dir=self.results_dir / "training" / f"{model_name}_{dataset_name}"
        )
        
        # Train model
        self.logger.info("Starting training...")
        start_time = time.time()
        
        try:
            trained_model, history = trainer.train(
                model_instance,
                train_ds,
                val_ds,
                model_name,
                dataset_name,
                class_weights
            )
            
            training_time = time.time() - start_time
            
            # Evaluate on test set
            self.logger.info("Evaluating on test set...")
            test_results = trained_model.evaluate(test_ds, verbose=0)
            test_metrics = dict(zip(trained_model.metrics_names, test_results))
            
            # Prepare results
            results = {
                "success": True,
                "model": model_name,
                "dataset": dataset_name,
                "training_time": training_time,
                "epochs_trained": len(history['loss']),
                "best_val_loss": float(min(history.get('val_loss', [float('inf')]))),
                "best_val_accuracy": float(max(history.get('val_accuracy', [0]))),
                "test_metrics": test_metrics,
                "data_info": data_info,
                "config": config.to_dict(),
                "optimized": optimize_hyperparams
            }
            
            if optimize_hyperparams:
                results["optimization_results"] = opt_results
            
            # Save results
            results_path = self.results_dir / "results" / f"{model_name}_{dataset_name}_results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=_json_safe)
            
            self.logger.info(f"Training complete! Test accuracy: {test_metrics.get('accuracy', 0):.4f}")
            
            # Clean up
            tf.keras.backend.clear_session()
            del model_instance
            del trained_model
            gc.collect()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                "success": False,
                "model": model_name,
                "dataset": dataset_name,
                "error": str(e)
            }
    
    def train_all_combinations(self, model_names: Optional[List[str]] = None,
                             dataset_names: Optional[List[str]] = None,
                             optimize_hyperparams: bool = False,
                             optuna_trials: int = 20) -> Dict[str, Any]:
        """Train all model-dataset combinations"""
        
        # Get all models if not specified
        if model_names is None:
            model_names = ["LivenessNet", "AttackNetV1", "AttackNetV2_1", "AttackNetV2_2"]
        
        # Get all datasets if not specified
        if dataset_names is None:
            dataset_names = [ds.value for ds in DatasetSource if ds != DatasetSource.COMBINED_ALL]
        
        self.logger.info(f"Starting training orchestration")
        self.logger.info(f"Models: {model_names}")
        self.logger.info(f"Datasets: {dataset_names}")
        self.logger.info(f"Total combinations: {len(model_names) * len(dataset_names)}")
        self.logger.info(f"Threads: {self.num_threads}")
        self.logger.info(f"Hyperparameter optimization: {optimize_hyperparams}")
        
        # Create training jobs
        jobs = []
        for model_name in model_names:
            for dataset_name in dataset_names:
                jobs.append((model_name, dataset_name))
        
        # Execute training jobs with thread pool
        all_results = {}
        failed_jobs = []
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(
                    self.train_single_combination,
                    model_name,
                    dataset_name,
                    optimize_hyperparams,
                    optuna_trials
                ): (model_name, dataset_name)
                for model_name, dataset_name in jobs
            }
            
            # Process completed jobs
            for future in as_completed(future_to_job):
                model_name, dataset_name = future_to_job[future]
                job_key = f"{model_name}_{dataset_name}"
                
                try:
                    result = future.result()
                    all_results[job_key] = result
                    
                    if not result['success']:
                        failed_jobs.append(job_key)
                        self.logger.error(f"Job failed: {job_key}")
                    else:
                        self.logger.info(f"Job completed: {job_key}")
                    
                except Exception as e:
                    self.logger.error(f"Job execution failed: {job_key} - {e}")
                    failed_jobs.append(job_key)
                    all_results[job_key] = {
                        "success": False,
                        "error": str(e)
                    }
        
        # Generate summary report
        summary = self._generate_summary_report(all_results, failed_jobs)
        
        # Save summary
        summary_path = self.results_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=_json_safe)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Training orchestration complete!")
        self.logger.info(f"Successful: {len(all_results) - len(failed_jobs)}/{len(all_results)}")
        self.logger.info(f"Failed: {len(failed_jobs)}")
        self.logger.info(f"Summary saved to: {summary_path}")
        
        return summary
    
    def _generate_summary_report(self, results: Dict[str, Any], 
                               failed_jobs: List[str]) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        
        # Extract successful results
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        # Best models per dataset
        best_per_dataset = {}
        for dataset in set(r['dataset'] for r in successful_results.values()):
            dataset_results = {
                k: v for k, v in successful_results.items() 
                if v['dataset'] == dataset
            }
            
            if dataset_results:
                best_key = max(
                    dataset_results.keys(),
                    key=lambda k: dataset_results[k]['test_metrics'].get('accuracy', 0)
                )
                best_per_dataset[dataset] = {
                    "model": dataset_results[best_key]['model'],
                    "accuracy": dataset_results[best_key]['test_metrics'].get('accuracy', 0),
                    "loss": dataset_results[best_key]['test_metrics'].get('loss', float('inf'))
                }
        
        # Best models overall
        if successful_results:
            best_overall = max(
                successful_results.items(),
                key=lambda x: x[1]['test_metrics'].get('accuracy', 0)
            )
        else:
            best_overall = None
        
        # Training statistics
        training_times = [r['training_time'] for r in successful_results.values()]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_jobs": len(results),
            "successful_jobs": len(successful_results),
            "failed_jobs": len(failed_jobs),
            "failed_job_names": failed_jobs,
            "best_models_per_dataset": best_per_dataset,
            "best_overall": {
                "combination": best_overall[0] if best_overall else None,
                "accuracy": best_overall[1]['test_metrics'].get('accuracy', 0) if best_overall else 0,
                "model": best_overall[1]['model'] if best_overall else None,
                "dataset": best_overall[1]['dataset'] if best_overall else None
            } if best_overall else None,
            "training_statistics": {
                "total_time": sum(training_times) if training_times else 0,
                "average_time": np.mean(training_times) if training_times else 0,
                "min_time": min(training_times) if training_times else 0,
                "max_time": max(training_times) if training_times else 0
            },
            "detailed_results": results
        }
        
        return summary


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train all models on all datasets")
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["LivenessNet", "AttackNetV1", "AttackNetV2_1", "AttackNetV2_2"],
        help="Models to train (default: all)"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["msspoof", "3dmad", "csmad", "replay_attack", "our_dataset"],
        help="Datasets to use (default: all)"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Number of parallel training threads (default: 2)"
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable hyperparameter optimization with Optuna"
    )
    
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=20,
        help="Number of Optuna trials for optimization (default: 20)"
    )
    
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training"
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory (default: results)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        set_mixed_precision(True)
    
    # Create orchestrator
    orchestrator = ModelTrainingOrchestrator(
        results_dir=args.results_dir,
        num_threads=args.threads
    )
    
    # Train models
    if args.models and args.datasets and len(args.models) == 1 and len(args.datasets) == 1:
        # Train single combination
        result = orchestrator.train_single_combination(
            args.models[0],
            args.datasets[0],
            args.optimize,
            args.optuna_trials
        )
        
        print(f"\nTraining complete!")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Test accuracy: {result['test_metrics'].get('accuracy', 0):.4f}")
    else:
        # Train multiple combinations
        summary = orchestrator.train_all_combinations(
            model_names=args.models,
            dataset_names=args.datasets,
            optimize_hyperparams=args.optimize,
            optuna_trials=args.optuna_trials
        )
        
        print(f"\nTraining complete!")
        print(f"Successful: {summary['successful_jobs']}/{summary['total_jobs']}")
        print(f"\nBest models per dataset:")
        for dataset, info in summary['best_models_per_dataset'].items():
            print(f"  {dataset}: {info['model']} (accuracy: {info['accuracy']:.4f})")
        
        if summary['best_overall']:
            print(f"\nBest overall: {summary['best_overall']['model']} on {summary['best_overall']['dataset']}")
            print(f"  Accuracy: {summary['best_overall']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
