#!/usr/bin/env python3
"""
Train All Models on Combined Dataset

This script trains all model architectures on the combined dataset
for improved cross-dataset generalization.
"""

import sys
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time
import gc

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.architectures import create_model
from src.dataset_loader import DatasetLoader, DatasetConfig
from src.training_utils import ModelTrainer, TrainingConfig, set_gpu_memory_growth
from src.evaluation_utils import ModelEvaluator
from config.model_configs import get_model_config

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup comprehensive logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("CombinedTraining")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(
        log_dir / f"combined_training_{timestamp}.log",
        encoding='utf-8'
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
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

class CombinedModelTrainer:
    """Train models on combined dataset with enhanced configurations."""
    
    def __init__(self, results_dir: Path = Path("results/combined"),
                 combined_dataset: str = "combined_all"):
        """
        Initialize trainer for combined dataset.
        
        Args:
            results_dir: Directory to save results
            combined_dataset: Name of the combined dataset to use
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.combined_dataset = combined_dataset
        self.logger = setup_logging(self.results_dir / "logs")
        
        # Setup GPU
        set_gpu_memory_growth()
        
        # Store training results
        self.training_results = {}
        
        self.logger.info(f"Combined Model Trainer initialized")
        self.logger.info(f"  Results directory: {self.results_dir}")
        self.logger.info(f"  Combined dataset: {self.combined_dataset}")
    
    def get_optimized_config(self, model_name: str) -> TrainingConfig:
        """
        Get optimized training configuration for combined dataset.
        
        Combined datasets need different hyperparameters due to:
        - Larger size
        - More diversity
        - Different convergence characteristics
        """
        # Get base config from model_configs.py
        base_config = get_model_config(model_name)
        
        # Create enhanced config for combined dataset
        config = TrainingConfig(
            optimizer=base_config.optimizer,
            learning_rate=base_config.learning_rate * 0.5,  # Lower LR for stability
            epochs=20,  # More epochs for larger dataset
            batch_size=16,  # Smaller batch for diversity
            early_stopping_patience=20,
            reduce_lr_patience=3,
            reduce_lr_factor=0.5,
            min_learning_rate=1e-9,
            use_mixed_precision=False,
            use_tensorboard=True,
            warmup_epochs=0,  # Warmup for stability
            #label_smoothing=0.1,  # Smoothing for generalization
            gradient_clip_norm=1.0,
            l2_regularization=base_config.l2_regularization
        )
        
        if model_name == "LivenessNet":
            config.learning_rate = 1e-7  # Conservative for simple model
            config.dropout_rate = 0.01  # Less dropout
        elif model_name == "AttackNetV1":
            config.learning_rate = 1e-5
            config.dropout_rate = 0.05
        elif model_name == "AttackNetV2_1":
            config.learning_rate = 1e-6
            config.dropout_rate = 0.05
            #config.warmup_epochs = 5  # More warmup for complex model
        elif model_name == "AttackNetV2_2":
            config.learning_rate = 1e-6
            config.dropout_rate = 0.05
        
        return config
    
    def train_single_model(self, model_name: str, 
                          augmentation: bool = False,
                          augmentation_level: str = "medium") -> Dict[str, Any]:
        """
        Train a single model on the combined dataset.
        
        Args:
            model_name: Name of the model architecture
            augmentation: Whether to use data augmentation
            augmentation_level: Level of augmentation (light/medium/heavy)
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Training {model_name} on {self.combined_dataset}")
        self.logger.info(f"  Augmentation: {augmentation} ({augmentation_level if augmentation else 'N/A'})")
        self.logger.info(f"{'='*70}")
        
        # Get optimized configuration
        config = self.get_optimized_config(model_name)
        
        # Load combined dataset
        dataset_config = DatasetConfig(
            batch_size=config.batch_size,
            augment=augmentation,
            augmentation_level=augmentation_level,
            validation_split=0.15,
            shuffle=True,
            mixed_precision=config.use_mixed_precision
        )
        
        self.logger.info("Loading combined dataset...")
        loader = DatasetLoader(dataset_config)
        
        try:
            train_ds, val_ds, test_ds, data_info = loader.load_dataset(
                self.combined_dataset, 
                ensure_no_leak=True
            )
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return {"success": False, "error": str(e)}
        
        # Log dataset info
        self.logger.info(f"Dataset loaded:")
        self.logger.info(f"  Train samples: {data_info['train_samples']:,}")
        self.logger.info(f"  Val samples: {data_info['val_samples']:,}")
        self.logger.info(f"  Test samples: {data_info['test_samples']:,}")
        self.logger.info(f"  Input shape: {data_info['input_shape']}")
        
        # Get model config for dropout
        model_config = get_model_config(model_name)
        
        # Create model
        self.logger.info(f"Creating {model_name} model...")
        model = create_model(
            model_name, 
            data_info['input_shape'],
            dropout_rate=model_config.dropout_rate,
            l2_reg=config.l2_regularization
        )
        model_instance = model.get_model()
        
        # Log model info
        self.logger.info(f"Model parameters: {model_instance.count_params():,}")
        
        # Create trainer
        trainer = ModelTrainer(
            config,
            save_dir=self.results_dir / model_name
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
                self.combined_dataset,
                data_info.get('class_weights')
            )
            
            training_time = time.time() - start_time
            
            # Evaluate on test set
            self.logger.info("Evaluating on test set...")
            test_results = trained_model.evaluate(test_ds, verbose=1)
            test_metrics = dict(zip(trained_model.metrics_names, test_results))
            
            # Additional evaluation with detailed metrics
            evaluator = ModelEvaluator(save_dir=self.results_dir / "evaluation")
            detailed_eval = evaluator.evaluate_model(
                trained_model,
                test_ds,
                model_name,
                self.combined_dataset,
                save_predictions=True
            )
            
            # Prepare results
            results = {
                "success": True,
                "model": model_name,
                "dataset": self.combined_dataset,
                "training_time": training_time,
                "epochs_trained": len(history['loss']),
                "best_val_loss": float(min(history.get('val_loss', [float('inf')]))),
                "best_val_accuracy": float(max(history.get('val_accuracy', [0]))),
                "test_metrics": test_metrics,
                "detailed_evaluation": detailed_eval,
                "data_info": {
                    "train_samples": data_info['train_samples'],
                    "val_samples": data_info['val_samples'],
                    "test_samples": data_info['test_samples'],
                    "included_datasets": data_info.get('dataset_stats', {}).get('included_datasets', [])
                },
                "config": config.to_dict(),
                "augmentation": augmentation,
                "augmentation_level": augmentation_level
            }
            
            # Save results
            results_path = self.results_dir / "results" / f"{model_name}_results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Training complete for {model_name}!")
            self.logger.info(f"  Training time: {training_time/60:.2f} minutes")
            self.logger.info(f"  Test accuracy: {test_metrics.get('accuracy', 0):.4f}")
            self.logger.info(f"  Test loss: {test_metrics.get('loss', 0):.4f}")
            self.logger.info(f"  ACER: {detailed_eval['biometric_metrics']['acer']:.4f}")
            self.logger.info(f"  EER: {detailed_eval['biometric_metrics']['eer']:.4f}")
            self.logger.info(f"{'='*50}")
            
            # Store in results
            self.training_results[model_name] = results
            
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
                "dataset": self.combined_dataset,
                "error": str(e)
            }
    
    def train_all_models(self, augmentation: bool = False,
                        augmentation_level: str = "medium") -> Dict[str, Any]:
        """
        Train all four models on the combined dataset.
        
        Args:
            augmentation: Whether to use data augmentation
            augmentation_level: Level of augmentation
            
        Returns:
            Summary of all training results
        """
        model_names = ["LivenessNet", "AttackNetV1", "AttackNetV2_1", "AttackNetV2_2"]
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"TRAINING ALL MODELS ON COMBINED DATASET")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Dataset: {self.combined_dataset}")
        self.logger.info(f"Models: {', '.join(model_names)}")
        self.logger.info(f"Augmentation: {augmentation} ({augmentation_level})")
        
        # Train each model
        all_results = {}
        successful_models = []
        failed_models = []
        
        for model_name in model_names:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Model {model_names.index(model_name)+1}/{len(model_names)}: {model_name}")
            self.logger.info(f"{'='*70}")
            
            result = self.train_single_model(
                model_name, 
                augmentation=augmentation,
                augmentation_level=augmentation_level
            )
            
            all_results[model_name] = result
            
            if result['success']:
                successful_models.append(model_name)
            else:
                failed_models.append(model_name)
        
        # Generate summary
        summary = self._generate_summary(all_results, successful_models, failed_models)
        
        # Save summary
        summary_path = self.results_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print final summary
        self._print_summary(summary)
        
        return summary
    
    def _generate_summary(self, results: Dict[str, Any], 
                         successful: List[str], 
                         failed: List[str]) -> Dict[str, Any]:
        """Generate comprehensive training summary."""
        
        # Extract metrics from successful models
        metrics_comparison = {}
        for model_name in successful:
            result = results[model_name]
            metrics_comparison[model_name] = {
                "test_accuracy": result['test_metrics'].get('accuracy', 0),
                "test_loss": result['test_metrics'].get('loss', 0),
                "val_accuracy": result['best_val_accuracy'],
                "val_loss": result['best_val_loss'],
                "acer": result['detailed_evaluation']['biometric_metrics']['acer'],
                "eer": result['detailed_evaluation']['biometric_metrics']['eer'],
                "training_time_minutes": result['training_time'] / 60,
                "epochs_trained": result['epochs_trained']
            }
        
        # Find best models
        best_models = {}
        if metrics_comparison:
            best_models = {
                "accuracy": max(metrics_comparison.keys(), 
                              key=lambda k: metrics_comparison[k]['test_accuracy']),
                "acer": min(metrics_comparison.keys(), 
                           key=lambda k: metrics_comparison[k]['acer']),
                "eer": min(metrics_comparison.keys(), 
                          key=lambda k: metrics_comparison[k]['eer'])
            }
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "combined_dataset": self.combined_dataset,
            "total_models": len(results),
            "successful_models": successful,
            "failed_models": failed,
            "success_rate": len(successful) / len(results) if results else 0,
            "metrics_comparison": metrics_comparison,
            "best_models": best_models,
            "training_configuration": {
                "augmentation": results[successful[0]]['augmentation'] if successful else None,
                "augmentation_level": results[successful[0]]['augmentation_level'] if successful else None
            },
            "detailed_results": results
        }
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print formatted training summary."""
        
        print(f"\n{'='*70}")
        print("TRAINING SUMMARY - COMBINED DATASET")
        print(f"{'='*70}")
        print(f"Dataset: {summary['combined_dataset']}")
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        
        if summary['successful_models']:
            print(f"\n✅ Successful models: {', '.join(summary['successful_models'])}")
        
        if summary['failed_models']:
            print(f"\n❌ Failed models: {', '.join(summary['failed_models'])}")
        
        if summary['metrics_comparison']:
            print(f"\n{'='*70}")
            print("PERFORMANCE COMPARISON")
            print(f"{'='*70}")
            print(f"{'Model':<15} {'Accuracy':<10} {'ACER':<10} {'EER':<10} {'Time (min)':<12}")
            print("-" * 70)
            
            for model, metrics in summary['metrics_comparison'].items():
                print(f"{model:<15} "
                      f"{metrics['test_accuracy']:<10.4f} "
                      f"{metrics['acer']:<10.4f} "
                      f"{metrics['eer']:<10.4f} "
                      f"{metrics['training_time_minutes']:<12.2f}")
            
            print(f"\n{'='*70}")
            print("BEST MODELS")
            print(f"{'='*70}")
            for metric, model in summary['best_models'].items():
                value = summary['metrics_comparison'][model][
                    'test_accuracy' if metric == 'accuracy' else metric
                ]
                print(f"Best {metric}: {model} ({value:.4f})")
    
    def evaluate_cross_dataset_performance(self, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a model trained on combined dataset on individual datasets.
        
        This tests how well the combined training improves generalization.
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"CROSS-DATASET EVALUATION: {model_name}")
        self.logger.info(f"{'='*70}")
        
        # Load trained model
        model_path = self.results_dir / model_name / "checkpoints" / f"{model_name}_{self.combined_dataset}" / "best_model.h5"
        
        if not model_path.exists():
            # Try alternative path
            model_path = self.results_dir / model_name / "checkpoints" / "best_model.h5"
        
        if not model_path.exists():
            self.logger.error(f"Model not found: {model_path}")
            return {"error": "Model not found"}
        
        model = tf.keras.models.load_model(model_path)
        
        # Test on individual datasets
        individual_datasets = ["3dmad", "csmad", "replay_attack", "msspoof", "our_dataset"]
        evaluator = ModelEvaluator(save_dir=self.results_dir / "cross_evaluation")
        
        cross_results = {}
        for dataset_name in individual_datasets:
            self.logger.info(f"\nEvaluating on {dataset_name}...")
            
            try:
                # Load dataset
                loader = DatasetLoader(DatasetConfig(batch_size=32, augment=False))
                _, _, test_ds, _ = loader.load_dataset(dataset_name)
                
                # Evaluate
                eval_result = evaluator.evaluate_model(
                    model, test_ds, 
                    f"{model_name}_combined", 
                    dataset_name,
                    save_predictions=False
                )
                
                cross_results[dataset_name] = {
                    "accuracy": eval_result['standard_metrics']['accuracy'],
                    "acer": eval_result['biometric_metrics']['acer'],
                    "eer": eval_result['biometric_metrics']['eer']
                }
                
                self.logger.info(f"  Accuracy: {cross_results[dataset_name]['accuracy']:.4f}")
                self.logger.info(f"  ACER: {cross_results[dataset_name]['acer']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate on {dataset_name}: {e}")
                cross_results[dataset_name] = {"error": str(e)}
        
        # Calculate average performance
        valid_results = [r for r in cross_results.values() if 'accuracy' in r]
        if valid_results:
            avg_accuracy = np.mean([r['accuracy'] for r in valid_results])
            avg_acer = np.mean([r['acer'] for r in valid_results])
            avg_eer = np.mean([r['eer'] for r in valid_results])
            
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"AVERAGE CROSS-DATASET PERFORMANCE")
            self.logger.info(f"  Average Accuracy: {avg_accuracy:.4f}")
            self.logger.info(f"  Average ACER: {avg_acer:.4f}")
            self.logger.info(f"  Average EER: {avg_eer:.4f}")
            self.logger.info(f"{'='*50}")
        
        return cross_results


def main():
    """Main training script for combined dataset."""
    parser = argparse.ArgumentParser(description="Train models on combined dataset")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="combined_all",
        help="Name of the combined dataset (default: combined_all)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["LivenessNet", "AttackNetV1", "AttackNetV2_1", "AttackNetV2_2"],
        help="Specific models to train (default: all)"
    )
    
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable data augmentation"
    )
    
    parser.add_argument(
        "--augmentation-level",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Augmentation level (default: medium)"
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/combined"),
        help="Results directory (default: results/combined)"
    )
    
    parser.add_argument(
        "--cross-evaluation",
        action="store_true",
        help="Perform cross-dataset evaluation after training"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print(f"\n{'='*70}")
    print("COMBINED DATASET MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Augmentation: {not args.no_augmentation}")
    print(f"Results directory: {args.results_dir}")
    
    # Create trainer
    trainer = CombinedModelTrainer(
        results_dir=args.results_dir,
        combined_dataset=args.dataset
    )
    
    # Train models
    if args.models:
        # Train specific models
        print(f"\nTraining specific models: {', '.join(args.models)}")
        results = {}
        for model_name in args.models:
            result = trainer.train_single_model(
                model_name,
                augmentation=not args.no_augmentation,
                augmentation_level=args.augmentation_level
            )
            results[model_name] = result
    else:
        # Train all models
        print("\nTraining all models...")
        summary = trainer.train_all_models(
            augmentation=not args.no_augmentation,
            augmentation_level=args.augmentation_level
        )
    
    # Perform cross-dataset evaluation if requested
    if args.cross_evaluation:
        print(f"\n{'='*70}")
        print("CROSS-DATASET EVALUATION")
        print(f"{'='*70}")
        
        models_to_evaluate = args.models if args.models else [
            "LivenessNet", "AttackNetV1", "AttackNetV2_1", "AttackNetV2_2"
        ]
        
        cross_results = {}
        for model_name in models_to_evaluate:
            if model_name in trainer.training_results and trainer.training_results[model_name]['success']:
                cross_results[model_name] = trainer.evaluate_cross_dataset_performance(model_name)
        
        # Save cross-evaluation results
        cross_path = args.results_dir / "cross_evaluation_results.json"
        with open(cross_path, "w") as f:
            json.dump(cross_results, f, indent=2, default=str)
        
        print(f"\nCross-evaluation results saved to: {cross_path}")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {args.results_dir}")
    print(f"\nNext steps:")
    print("1. Review training results in:", args.results_dir / "training_summary.json")
    print("2. Compare with single-dataset results")
    print("3. Test on completely unseen data")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
