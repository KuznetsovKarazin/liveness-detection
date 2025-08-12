#!/usr/bin/env python3
"""
Evaluate All Models Script

This script performs comprehensive evaluation of trained models including:
- Standard classification metrics
- Biometric-specific metrics (APCER, BPCER, EER)
- Cross-dataset evaluation
- Extensive visualization
- Model comparison and ranking
"""

import sys
import argparse
import logging
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import gc

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset_loader import DatasetLoader, DatasetSource, DatasetConfig
from src.evaluation_utils import ModelEvaluator, BiometricMetrics
from src.architectures import create_model


class ModelEvaluationOrchestrator:
    """Orchestrate evaluation of multiple models"""
    
    def __init__(self, results_dir: Path = Path("results"), 
                 models_dir: Path = None):
        self.results_dir = Path(results_dir)
        # Changed: models_dir now points to results/training by default
        self.models_dir = Path(models_dir) if models_dir else self.results_dir / "training"
        self.eval_dir = self.results_dir / "evaluation"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(save_dir=self.eval_dir)
        
        # Store all evaluation results
        self.all_results = []
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.results_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(
            log_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            self.logger.setLevel(logging.DEBUG)
    
    def find_trained_models(self) -> List[Dict[str, str]]:
        """Find all trained model checkpoints with the new directory structure"""
        trained_models = []
        
        # Check if models_dir exists
        if not self.models_dir.exists():
            self.logger.warning(f"Models directory does not exist: {self.models_dir}")
            return trained_models
        
        self.logger.info(f"Looking for models in: {self.models_dir}")
        
        # NEW: Look for model directories in results/training/
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            # Check for checkpoints subdirectory
            checkpoints_dir = model_dir / "checkpoints"
            if not checkpoints_dir.exists():
                self.logger.debug(f"No checkpoints directory in {model_dir}")
                continue
            
            # Look for best_model.h5 in the checkpoints directory
            model_file = checkpoints_dir / "best_model.h5"
            
            # Check multiple possible locations for the model file
            if not model_file.exists():
                # Check in subdirectories within checkpoints
                possible_locations = [
                    checkpoints_dir / model_dir.name / "best_model.h5",  # e.g., checkpoints/AttackNetV1_3dmad/best_model.h5
                    checkpoints_dir / model_dir.name,  # Check if it's a directory
                    checkpoints_dir / "*.h5",  # Any .h5 file in checkpoints
                ]
                
                # First, check if there's a subdirectory with the same name
                subdir_path = checkpoints_dir / model_dir.name
                if subdir_path.exists() and subdir_path.is_dir():
                    # Look for .h5 files in the subdirectory
                    h5_files = list(subdir_path.glob("*.h5"))
                    keras_files = list(subdir_path.glob("*.keras"))
                    
                    if h5_files:
                        model_file = h5_files[0]
                        self.logger.info(f"Found .h5 model in subdirectory: {model_file}")
                    elif keras_files:
                        model_file = keras_files[0]
                        self.logger.info(f"Found .keras model in subdirectory: {model_file}")
                    else:
                        # Check if the subdirectory itself is a SavedModel format
                        saved_model_pb = subdir_path / "saved_model.pb"
                        if saved_model_pb.exists():
                            model_file = subdir_path  # Use the directory as the model path
                            self.logger.info(f"Found SavedModel format in: {model_file}")
                        else:
                            self.logger.debug(f"No model files found in subdirectory {subdir_path}")
                            continue
                else:
                    # Try to find any .h5 or .keras files directly in checkpoints
                    h5_files = list(checkpoints_dir.glob("*.h5"))
                    keras_files = list(checkpoints_dir.glob("*.keras"))
                    
                    if h5_files:
                        model_file = h5_files[0]
                        self.logger.info(f"Found .h5 model: {model_file}")
                    elif keras_files:
                        model_file = keras_files[0]
                        self.logger.info(f"Found .keras model: {model_file}")
                    else:
                        self.logger.debug(f"No model files found in {checkpoints_dir}")
                        continue
            
            # Extract model and dataset names from directory name
            dir_name = model_dir.name
            self.logger.debug(f"Processing directory: {dir_name}")
            
            # Parse the directory name to extract model and dataset
            parts = dir_name.split('_')
            
            # Handle different model name patterns
            if 'AttackNetV1_' in dir_name and not ('AttackNetV2' in dir_name):
                model_name = 'AttackNetV1'
                dataset_name = '_'.join(parts[1:])
            elif 'AttackNetV2_1_' in dir_name:
                model_name = 'AttackNetV2_1'
                # Find where the dataset name starts (after AttackNetV2_1)
                dataset_parts = dir_name.split('AttackNetV2_1_', 1)
                dataset_name = dataset_parts[1] if len(dataset_parts) > 1 else '_'.join(parts[2:])
            elif 'AttackNetV2_2_' in dir_name:
                model_name = 'AttackNetV2_2'
                # Find where the dataset name starts (after AttackNetV2_2)
                dataset_parts = dir_name.split('AttackNetV2_2_', 1)
                dataset_name = dataset_parts[1] if len(dataset_parts) > 1 else '_'.join(parts[2:])
            elif 'LivenessNet_' in dir_name:
                model_name = 'LivenessNet'
                dataset_name = '_'.join(parts[1:])
            else:
                # Default parsing
                model_name = parts[0]
                dataset_name = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
            
            trained_models.append({
                'model_name': model_name,
                'dataset_name': dataset_name,
                'model_path': str(model_file),
                'dir_name': dir_name
            })
            
            self.logger.info(f"Found model: {model_name} trained on {dataset_name}")
        
        self.logger.info(f"Found {len(trained_models)} trained models in total")
        
        # Log all found models for debugging
        for model in trained_models:
            self.logger.debug(f"  - {model['model_name']} on {model['dataset_name']} at {model['model_path']}")
        
        return trained_models
    
    def evaluate_single_model(self, model_info: Dict[str, str],
                            test_on_all_datasets: bool = False) -> List[Dict[str, Any]]:
        """
        Evaluate a single model
        
        Args:
            model_info: Dictionary with model information
            test_on_all_datasets: Whether to test on all datasets (cross-dataset)
            
        Returns:
            List of evaluation results
        """
        results = []
        model_name = model_info['model_name']
        trained_dataset = model_info['dataset_name']
        model_path = model_info['model_path']
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Evaluating {model_name} (trained on {trained_dataset})")
        self.logger.info(f"Model path: {model_path}")
        self.logger.info(f"{'='*60}")
        
        # Check if model file exists
        if not Path(model_path).exists():
            self.logger.error(f"Model file does not exist: {model_path}")
            return results
        
        # Load model
        try:
            model_path_obj = Path(model_path)
            
            # Check if it's a directory (SavedModel format) or a file (.h5/.keras)
            if model_path_obj.is_dir():
                self.logger.info(f"Loading SavedModel from directory: {model_path}")
                model = tf.keras.models.load_model(model_path)
            else:
                self.logger.info(f"Loading model from file: {model_path}")
                model = tf.keras.models.load_model(model_path)
            
            self.logger.info(f"Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return results
        
        # Determine which datasets to test on
        if test_on_all_datasets:
            test_datasets = [ds.value for ds in DatasetSource if ds != DatasetSource.COMBINED_ALL]
        else:
            test_datasets = [trained_dataset]
        
        # Evaluate on each dataset
        for dataset_name in test_datasets:
            self.logger.info(f"\nEvaluating on {dataset_name} dataset...")
            
            # Load test dataset
            try:
                loader = DatasetLoader(DatasetConfig(
                    batch_size=32,
                    augment=False,  # No augmentation for evaluation
                    shuffle=False
                ))
                
                _, _, test_dataset, data_info = loader.load_dataset(dataset_name)
                
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
                continue
            
            # Evaluate model
            try:
                eval_result = self.evaluator.evaluate_model(
                    model=model,
                    test_dataset=test_dataset,
                    model_name=f"{model_name}_trained_on_{trained_dataset}",
                    dataset_name=dataset_name,
                    save_predictions=True
                )
                
                # Add additional info
                eval_result['trained_on'] = trained_dataset
                eval_result['cross_dataset'] = (trained_dataset != dataset_name)
                eval_result['model_path'] = model_path
                
                results.append(eval_result)
                self.all_results.append(eval_result)
                
                # Log key metrics
                self.logger.info(f"Evaluation complete:")
                self.logger.info(f"  Accuracy: {eval_result['standard_metrics']['accuracy']:.4f}")
                self.logger.info(f"  ACER: {eval_result['biometric_metrics']['acer']:.4f}")
                self.logger.info(f"  EER: {eval_result['biometric_metrics']['eer']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Evaluation failed: {e}")
                continue
        
        # Clean up
        tf.keras.backend.clear_session()
        del model
        gc.collect()
        
        return results
    
    def evaluate_specific_model(self, model_name: str, dataset_name: Optional[str] = None,
                               test_on_all_datasets: bool = False) -> List[Dict[str, Any]]:
        """
        Evaluate a specific model by name
        
        Args:
            model_name: Name of the model to evaluate
            dataset_name: Optional dataset the model was trained on
            test_on_all_datasets: Whether to test on all datasets
            
        Returns:
            List of evaluation results
        """
        # Find all trained models
        trained_models = self.find_trained_models()
        
        # Filter for the specific model
        matching_models = []
        for model_info in trained_models:
            if model_info['model_name'] == model_name:
                if dataset_name is None or model_info['dataset_name'] == dataset_name:
                    matching_models.append(model_info)
        
        if not matching_models:
            self.logger.warning(f"No trained model found for {model_name}" + 
                              (f" on {dataset_name}" if dataset_name else ""))
            return []
        
        # Evaluate all matching models
        all_results = []
        for model_info in matching_models:
            results = self.evaluate_single_model(model_info, test_on_all_datasets)
            all_results.extend(results)
        
        return all_results
    
    def evaluate_all_models(self, test_on_all_datasets: bool = False) -> Dict[str, Any]:
        """
        Evaluate all trained models
        
        Args:
            test_on_all_datasets: Whether to perform cross-dataset evaluation
            
        Returns:
            Summary of all evaluations
        """
        # Find all trained models
        trained_models = self.find_trained_models()
        
        if not trained_models:
            self.logger.warning("No trained models found!")
            self.logger.info(f"Searched in: {self.models_dir}")
            self.logger.info("Expected structure: results/training/ModelName_DatasetName/checkpoints/best_model.h5")
            return {}
        
        self.logger.info(f"Starting evaluation of {len(trained_models)} models")
        if test_on_all_datasets:
            self.logger.info("Cross-dataset evaluation enabled")
        
        # Evaluate each model
        for model_info in tqdm(trained_models, desc="Evaluating models"):
            self.evaluate_single_model(model_info, test_on_all_datasets)
        
        # Generate comprehensive report
        summary = self._generate_evaluation_summary()
        
        # Convert numpy types to Python types for JSON serialization
        summary = self._convert_numpy_types(summary)
        
        # Save summary
        summary_path = self.eval_dir / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=self._json_default)
        
        self.logger.info(f"\nEvaluation complete!")
        self.logger.info(f"Summary saved to: {summary_path}")
        
        return summary
    
    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation summary"""
        
        if not self.all_results:
            return {"error": "No evaluation results available"}
        
        # Create DataFrame for easier analysis
        data = []
        for result in self.all_results:
            row = {
                'model': result['model'].split('_trained_on_')[0],
                'trained_on': result['trained_on'],
                'tested_on': result['dataset'],
                'cross_dataset': result['cross_dataset'],
                'accuracy': result['standard_metrics']['accuracy'],
                'precision': result['standard_metrics']['precision'],
                'recall': result['standard_metrics']['recall'],
                'f1_score': result['standard_metrics']['f1_score'],
                'roc_auc': result['standard_metrics']['roc_auc'],
                'apcer': result['biometric_metrics']['apcer'],
                'bpcer': result['biometric_metrics']['bpcer'],
                'acer': result['biometric_metrics']['acer'],
                'eer': result['biometric_metrics']['eer'],
                'far': result['biometric_metrics']['far'],
                'frr': result['biometric_metrics']['frr']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Best models per metric
        best_models = {
            'accuracy': df.loc[df['accuracy'].idxmax()].to_dict(),
            'acer': df.loc[df['acer'].idxmin()].to_dict(),
            'eer': df.loc[df['eer'].idxmin()].to_dict(),
            'f1_score': df.loc[df['f1_score'].idxmax()].to_dict()
        }
        
        # Cross-dataset performance
        cross_dataset_results = df[df['cross_dataset']].copy()
        if not cross_dataset_results.empty:
            # Average performance per model across different test datasets
            cross_performance = cross_dataset_results.groupby(['model', 'trained_on']).agg({
                'accuracy': 'mean',
                'acer': 'mean',
                'eer': 'mean'
            }).round(4)
            
            # Find best performer (convert tuple index to string)
            if not cross_performance.empty:
                best_idx = cross_performance['accuracy'].idxmax()
                if isinstance(best_idx, tuple):
                    model_name, dataset_name = best_idx
                    best_cross_dataset = {
                        'model': model_name,
                        'dataset': dataset_name,
                        'accuracy': float(cross_performance.loc[best_idx, 'accuracy']),
                        'acer': float(cross_performance.loc[best_idx, 'acer']),
                        'eer': float(cross_performance.loc[best_idx, 'eer'])
                    }
                else:
                    best_cross_dataset = {
                        'model': str(best_idx),
                        'accuracy': float(cross_performance['accuracy'].max()),
                        'acer': float(cross_performance.loc[best_idx, 'acer']),
                        'eer': float(cross_performance.loc[best_idx, 'eer'])
                    }
                
                # Convert cross_performance to a serializable format
                cross_perf_dict = {}
                for idx, row in cross_performance.iterrows():
                    if isinstance(idx, tuple):
                        key = f"{idx[0]}_{idx[1]}"
                    else:
                        key = str(idx)
                    cross_perf_dict[key] = {
                        'accuracy': float(row['accuracy']),
                        'acer': float(row['acer']),
                        'eer': float(row['eer'])
                    }
            else:
                best_cross_dataset = None
                cross_perf_dict = {}
        else:
            cross_performance = None
            best_cross_dataset = None
            cross_perf_dict = None
        
        # Model ranking
        model_ranking = self._create_model_ranking(df)
        
        # Generate visualizations
        self._create_summary_visualizations(df)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_evaluations': len(self.all_results),
            'models_evaluated': df['model'].nunique(),
            'datasets_used': df['tested_on'].nunique(),
            'best_models': best_models,
            'model_ranking': model_ranking,
            'cross_dataset_evaluation': {
                'performed': df['cross_dataset'].any(),
                'best_performer': best_cross_dataset,
                'detailed_results': cross_perf_dict if cross_perf_dict is not None else None
            },
            'detailed_results': data
        }
        
        # Save detailed results to CSV
        df.to_csv(self.eval_dir / "evaluation_results.csv", index=False)
        
        return summary
    
    def _json_default(self, obj):
        """Default JSON serializer for numpy types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        else:
            return str(obj)
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.api.types.is_bool_dtype(type(obj)):
            return bool(obj)
        else:
            return obj
    
    def _create_model_ranking(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create overall model ranking based on multiple metrics"""
        
        # Calculate ranking score for each model
        # Lower is better for: acer, eer, apcer, bpcer
        # Higher is better for: accuracy, precision, recall, f1_score, roc_auc
        
        rankings = []
        
        for model in df['model'].unique():
            model_results = df[df['model'] == model]
            
            # Average metrics across all test datasets
            avg_metrics = {
                'accuracy': model_results['accuracy'].mean(),
                'precision': model_results['precision'].mean(),
                'recall': model_results['recall'].mean(),
                'f1_score': model_results['f1_score'].mean(),
                'roc_auc': model_results['roc_auc'].mean(),
                'acer': model_results['acer'].mean(),
                'eer': model_results['eer'].mean()
            }
            
            # Normalize metrics to 0-1 range
            # For metrics where lower is better, invert the score
            score_components = {
                'accuracy': avg_metrics['accuracy'],
                'f1_score': avg_metrics['f1_score'],
                'roc_auc': avg_metrics['roc_auc'],
                'acer': 1 - avg_metrics['acer'],  # Invert
                'eer': 1 - avg_metrics['eer']     # Invert
            }
            
            # Calculate weighted average score
            weights = {
                'accuracy': 0.25,
                'f1_score': 0.2,
                'roc_auc': 0.2,
                'acer': 0.2,
                'eer': 0.15
            }
            
            overall_score = sum(score_components[metric] * weight 
                              for metric, weight in weights.items())
            
            rankings.append({
                'model': model,
                'overall_score': float(overall_score),
                'metrics': avg_metrics,
                'num_evaluations': len(model_results)
            })
        
        # Sort by overall score (descending)
        rankings.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Add rank
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def _create_summary_visualizations(self, df: pd.DataFrame):
        """Create summary visualizations"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        viz_dir = self.eval_dir / "summary_visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Overall performance heatmap
        plt.figure(figsize=(12, 8))
        
        # Pivot data for heatmap
        metrics_to_plot = ['accuracy', 'acer', 'eer']
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(1, 3, i+1)
            
            pivot_data = df.pivot_table(
                index='model',
                columns='tested_on',
                values=metric,
                aggfunc='mean'
            )
            
            # Create heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.3f', 
                       cmap='RdYlGn' if metric == 'accuracy' else 'RdYlGn_r',
                       cbar_kws={'label': metric.upper()})
            
            plt.title(f'{metric.upper()} by Model and Dataset')
            plt.xlabel('Test Dataset')
            plt.ylabel('Model')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cross-dataset performance
        if df['cross_dataset'].any():
            plt.figure(figsize=(15, 10))
            
            cross_df = df[df['cross_dataset']].copy()
            
            # Group by model and calculate mean performance
            model_performance = cross_df.groupby('model')[['accuracy', 'acer', 'eer']].mean()
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Bar plot of average accuracy
            ax = axes[0, 0]
            model_performance['accuracy'].sort_values(ascending=False).plot(kind='bar', ax=ax)
            ax.set_ylabel('Average Accuracy')
            ax.set_title('Cross-Dataset Average Accuracy')
            ax.grid(True, alpha=0.3)
            
            # Bar plot of average ACER
            ax = axes[0, 1]
            model_performance['acer'].sort_values().plot(kind='bar', ax=ax)
            ax.set_ylabel('Average ACER')
            ax.set_title('Cross-Dataset Average ACER (lower is better)')
            ax.grid(True, alpha=0.3)
            
            # Scatter plot: Accuracy vs ACER
            ax = axes[1, 0]
            for model in model_performance.index:
                ax.scatter(model_performance.loc[model, 'accuracy'],
                          model_performance.loc[model, 'acer'],
                          s=100, label=model)
            ax.set_xlabel('Average Accuracy')
            ax.set_ylabel('Average ACER')
            ax.set_title('Accuracy vs ACER Trade-off')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Box plot of accuracy distribution
            ax = axes[1, 1]
            cross_df.boxplot(column='accuracy', by='model', ax=ax)
            ax.set_xlabel('Model')
            ax.set_ylabel('Accuracy')
            ax.set_title('Cross-Dataset Accuracy Distribution')
            plt.suptitle('')  # Remove automatic title
            
            plt.tight_layout()
            plt.savefig(viz_dir / "cross_dataset_performance.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Model comparison radar chart
        self._create_radar_chart(df, viz_dir)
    
    def _create_radar_chart(self, df: pd.DataFrame, save_dir: Path):
        """Create radar chart for model comparison"""
        import matplotlib.pyplot as plt
        from math import pi
        
        # Prepare data
        models = df['model'].unique()
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Calculate average metrics for each model
        model_metrics = {}
        for model in models:
            model_df = df[df['model'] == model]
            model_metrics[model] = [model_df[metric].mean() for metric in metrics]
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Number of variables
        num_vars = len(metrics)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        
        # Plot data
        for model, values in model_metrics.items():
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.15)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        
        # Set y-axis limits and labels
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        
        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.title('Model Performance Comparison\n(Average across all datasets)', 
                 size=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(save_dir / "model_comparison_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis on evaluation results"""
        
        if not self.all_results:
            return {"error": "No results available for analysis"}
        
        # Convert to DataFrame
        data = []
        for result in self.all_results:
            data.append({
                'model': result['model'].split('_trained_on_')[0],
                'dataset': result['dataset'],
                'accuracy': result['standard_metrics']['accuracy'],
                'acer': result['biometric_metrics']['acer']
            })
        
        df = pd.DataFrame(data)
        
        # Perform statistical tests
        from scipy import stats
        
        analysis_results = {}
        
        # 1. ANOVA for accuracy across models
        models = df['model'].unique()
        if len(models) > 2:
            accuracy_groups = [df[df['model'] == model]['accuracy'].values for model in models]
            f_stat, p_value = stats.f_oneway(*accuracy_groups)
            
            analysis_results['anova_accuracy'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        
        # 2. Pairwise comparisons
        if len(models) > 1:
            pairwise_results = {}
            
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    acc1 = df[df['model'] == model1]['accuracy'].values
                    acc2 = df[df['model'] == model2]['accuracy'].values
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(acc1, acc2)
                    
                    pairwise_results[f"{model1}_vs_{model2}"] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'mean_diff': float(acc1.mean() - acc2.mean())
                    }
            
            analysis_results['pairwise_comparisons'] = pairwise_results
        
        # 3. Effect size (Cohen's d)
        if len(models) == 2:
            acc1 = df[df['model'] == models[0]]['accuracy'].values
            acc2 = df[df['model'] == models[1]]['accuracy'].values
            
            # Cohen's d
            pooled_std = np.sqrt((np.std(acc1)**2 + np.std(acc2)**2) / 2)
            d = (acc1.mean() - acc2.mean()) / pooled_std
            
            analysis_results['cohens_d'] = {
                'value': float(d),
                'interpretation': 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'
            }
        
        # Save analysis results
        analysis_path = self.eval_dir / "statistical_analysis.json"
        # Convert numpy types before saving
        analysis_results = self._convert_numpy_types(analysis_results)
        with open(analysis_path, "w") as f:
            json.dump(analysis_results, f, indent=2)
        
        return analysis_results


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,  # Will use results/training by default
        help="Directory containing trained models (default: results/training)"
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory"
    )
    
    parser.add_argument(
        "--cross-dataset",
        action="store_true",
        help="Perform cross-dataset evaluation"
    )
    
    parser.add_argument(
        "--statistical-analysis",
        action="store_true",
        help="Perform statistical analysis of results"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Evaluate specific model only"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Evaluate on specific dataset only"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create orchestrator
    orchestrator = ModelEvaluationOrchestrator(
        results_dir=args.results_dir,
        models_dir=args.models_dir
    )
    
    # Perform evaluation
    if args.model:
        # Evaluate specific model
        results = orchestrator.evaluate_specific_model(
            args.model, 
            args.dataset,
            test_on_all_datasets=args.cross_dataset
        )
        
        if results:
            print(f"\nEvaluation complete for {args.model}!")
            for result in results:
                print(f"\nDataset: {result['dataset']}")
                print(f"  Accuracy: {result['standard_metrics']['accuracy']:.4f}")
                print(f"  ACER: {result['biometric_metrics']['acer']:.4f}")
                print(f"  EER: {result['biometric_metrics']['eer']:.4f}")
        else:
            print(f"\nNo trained models found for {args.model}")
    else:
        # Evaluate all models
        summary = orchestrator.evaluate_all_models(test_on_all_datasets=args.cross_dataset)
        
        if summary and 'best_models' in summary:
            print(f"\nEvaluation complete!")
            print(f"Total evaluations: {summary['total_evaluations']}")
            print(f"\nBest models:")
            print(f"  Accuracy: {summary['best_models']['accuracy']['model']} "
                  f"({summary['best_models']['accuracy']['accuracy']:.4f})")
            print(f"  ACER: {summary['best_models']['acer']['model']} "
                  f"({summary['best_models']['acer']['acer']:.4f})")
            print(f"  EER: {summary['best_models']['eer']['model']} "
                  f"({summary['best_models']['eer']['eer']:.4f})")
            
            if summary['cross_dataset_evaluation']['performed']:
                best_cross = summary['cross_dataset_evaluation']['best_performer']
                if best_cross:
                    print(f"\nBest cross-dataset performer: {best_cross['model']}")
                    print(f"  Average accuracy: {best_cross['accuracy']:.4f}")
                    print(f"  Average ACER: {best_cross['acer']:.4f}")
        else:
            print("\nNo models found to evaluate.")
            print(f"Please check that trained models exist in: {orchestrator.models_dir}")
            print("Expected structure: results/training/ModelName_DatasetName/checkpoints/best_model.h5")
    
    # Perform statistical analysis if requested
    if args.statistical_analysis and orchestrator.all_results:
        print("\nPerforming statistical analysis...")
        analysis = orchestrator.perform_statistical_analysis()
        
        if 'anova_accuracy' in analysis:
            print(f"\nANOVA results (accuracy):")
            print(f"  F-statistic: {analysis['anova_accuracy']['f_statistic']:.4f}")
            print(f"  p-value: {analysis['anova_accuracy']['p_value']:.4f}")
            print(f"  Significant: {analysis['anova_accuracy']['significant']}")
        
        if 'pairwise_comparisons' in analysis:
            print("\nPairwise comparisons:")
            for comparison, results in analysis['pairwise_comparisons'].items():
                if results['significant']:
                    print(f"  {comparison}: p={results['p_value']:.4f} (significant)")
        
        print(f"\nStatistical analysis saved to: {orchestrator.eval_dir}/statistical_analysis.json")


if __name__ == "__main__":
    main()
