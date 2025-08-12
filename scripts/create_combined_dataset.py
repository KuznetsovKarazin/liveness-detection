#!/usr/bin/env python3
"""
Combined Dataset Creator

This script creates a unified dataset by combining multiple existing datasets
from pickle files for improved cross-dataset generalization.
"""

import sys
import pickle
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

def setup_logging() -> logging.Logger:
    """Setup comprehensive logging."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        f'logs/combined_dataset_creation_{timestamp}.log', 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler],
        force=True
    )
    
    return logging.getLogger("CombinedDatasetCreator")

class CombinedDatasetCreator:
    """Create a unified dataset from multiple pickle files."""
    
    def __init__(self, train_ratio: float = 0.8, 
                 balance_across_datasets: bool = True,
                 max_samples_per_dataset: int = None,
                 random_seed: int = 42):
        """
        Initialize combined dataset creator.
        
        Args:
            train_ratio: Ratio of data for training
            balance_across_datasets: Whether to balance samples across datasets
            max_samples_per_dataset: Maximum samples to take from each dataset
            random_seed: Random seed for reproducibility
        """
        self.train_ratio = train_ratio
        self.balance_across_datasets = balance_across_datasets
        self.max_samples_per_dataset = max_samples_per_dataset
        self.random_seed = random_seed
        self.logger = setup_logging()
        
        np.random.seed(random_seed)
        
        # Ensure directories exist
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Combined Dataset Creator initialized:")
        self.logger.info(f"  Train ratio: {train_ratio}")
        self.logger.info(f"  Balance across datasets: {balance_across_datasets}")
        self.logger.info(f"  Max samples per dataset: {max_samples_per_dataset}")
        self.logger.info(f"  Random seed: {random_seed}")
    
    def load_dataset_pickle(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load a single dataset from pickle file.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, stats)
        """
        pkl_path = Path(f"data/processed/{dataset_name.lower()}.pkl")
        stats_path = Path(f"data/processed/{dataset_name.lower()}_stats.json")
        
        if not pkl_path.exists():
            self.logger.warning(f"Dataset not found: {pkl_path}")
            return None, None, None, None, {}
        
        # Load pickle
        with open(pkl_path, "rb") as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
        
        # Load stats if available
        stats = {}
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
        
        self.logger.info(f"Loaded {dataset_name}:")
        self.logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, stats
    
    def balance_dataset_samples(self, X: np.ndarray, y: np.ndarray, 
                               target_samples: int, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance samples from a dataset to target number.
        
        Args:
            X: Input data
            y: Labels
            target_samples: Target number of samples
            dataset_name: Name of dataset for logging
            
        Returns:
            Balanced X and y arrays
        """
        current_samples = len(X)
        
        if current_samples == target_samples:
            return X, y
        
        if current_samples > target_samples:
            # Undersample - take most diverse samples
            self.logger.info(f"  Undersampling {dataset_name}: {current_samples} -> {target_samples}")
            
            # Stratified sampling to maintain class balance
            y_labels = np.argmax(y, axis=1)
            indices = np.arange(current_samples)
            
            # Calculate samples per class
            unique_classes, class_counts = np.unique(y_labels, return_counts=True)
            class_ratios = class_counts / current_samples
            
            selected_indices = []
            for cls, ratio in zip(unique_classes, class_ratios):
                cls_indices = indices[y_labels == cls]
                n_samples = int(target_samples * ratio)
                
                # Random selection within class
                selected = np.random.choice(cls_indices, size=min(n_samples, len(cls_indices)), 
                                          replace=False)
                selected_indices.extend(selected)
            
            selected_indices = np.array(selected_indices)
            np.random.shuffle(selected_indices)
            
            # Adjust to exact target if needed
            if len(selected_indices) > target_samples:
                selected_indices = selected_indices[:target_samples]
            elif len(selected_indices) < target_samples:
                # Add more random samples
                remaining = target_samples - len(selected_indices)
                unselected = np.setdiff1d(indices, selected_indices)
                if len(unselected) > 0:
                    additional = np.random.choice(unselected, 
                                                size=min(remaining, len(unselected)), 
                                                replace=False)
                    selected_indices = np.concatenate([selected_indices, additional])
            
            return X[selected_indices], y[selected_indices]
        
        else:
            # Oversample - replicate samples
            self.logger.info(f"  Oversampling {dataset_name}: {current_samples} -> {target_samples}")
            
            # Calculate how many times to replicate
            n_replications = target_samples // current_samples
            remainder = target_samples % current_samples
            
            # Replicate entire dataset
            X_replicated = np.tile(X, (n_replications, 1, 1, 1))
            y_replicated = np.tile(y, (n_replications, 1))
            
            # Add remainder
            if remainder > 0:
                indices = np.random.choice(current_samples, size=remainder, replace=False)
                X_replicated = np.concatenate([X_replicated, X[indices]])
                y_replicated = np.concatenate([y_replicated, y[indices]])
            
            return X_replicated, y_replicated
    
    def create_combined_dataset(self, dataset_names: List[str], 
                              output_name: str = "combined_all") -> bool:
        """
        Create combined dataset from multiple datasets.
        
        Args:
            dataset_names: List of dataset names to combine
            output_name: Name for the output combined dataset
            
        Returns:
            Success status
        """
        self.logger.info(f"\nCreating combined dataset from: {dataset_names}")
        
        # Load all datasets
        all_X_train, all_X_test = [], []
        all_y_train, all_y_test = [], []
        dataset_stats = {}
        valid_datasets = []
        
        for dataset_name in dataset_names:
            X_train, X_test, y_train, y_test, stats = self.load_dataset_pickle(dataset_name)
            
            if X_train is not None:
                all_X_train.append(X_train)
                all_X_test.append(X_test)
                all_y_train.append(y_train)
                all_y_test.append(y_test)
                dataset_stats[dataset_name] = stats
                valid_datasets.append(dataset_name)
            else:
                self.logger.warning(f"Skipping {dataset_name} - not found")
        
        if len(valid_datasets) == 0:
            self.logger.error("No valid datasets found!")
            return False
        
        self.logger.info(f"\nCombining {len(valid_datasets)} datasets: {valid_datasets}")
        
        # Determine target samples per dataset if balancing
        if self.balance_across_datasets:
            # Find minimum dataset size or use max_samples_per_dataset
            train_sizes = [len(X) for X in all_X_train]
            test_sizes = [len(X) for X in all_X_test]
            
            self.logger.info(f"Original train sizes: {dict(zip(valid_datasets, train_sizes))}")
            self.logger.info(f"Original test sizes: {dict(zip(valid_datasets, test_sizes))}")
            
            if self.max_samples_per_dataset:
                target_train = min(min(train_sizes), self.max_samples_per_dataset)
                target_test = min(min(test_sizes), self.max_samples_per_dataset)
            else:
                # Use median to avoid being limited by smallest dataset
                target_train = int(np.median(train_sizes))
                target_test = int(np.median(test_sizes))
            
            self.logger.info(f"Target samples - Train: {target_train}, Test: {target_test}")
            
            # Balance each dataset
            balanced_X_train, balanced_y_train = [], []
            balanced_X_test, balanced_y_test = [], []
            
            for i, dataset_name in enumerate(valid_datasets):
                X_train_balanced, y_train_balanced = self.balance_dataset_samples(
                    all_X_train[i], all_y_train[i], target_train, dataset_name
                )
                X_test_balanced, y_test_balanced = self.balance_dataset_samples(
                    all_X_test[i], all_y_test[i], target_test, dataset_name
                )
                
                balanced_X_train.append(X_train_balanced)
                balanced_y_train.append(y_train_balanced)
                balanced_X_test.append(X_test_balanced)
                balanced_y_test.append(y_test_balanced)
            
            all_X_train = balanced_X_train
            all_y_train = balanced_y_train
            all_X_test = balanced_X_test
            all_y_test = balanced_y_test
        
        # Concatenate all datasets
        self.logger.info("\nConcatenating datasets...")
        X_train_combined = np.concatenate(all_X_train, axis=0)
        y_train_combined = np.concatenate(all_y_train, axis=0)
        X_test_combined = np.concatenate(all_X_test, axis=0)
        y_test_combined = np.concatenate(all_y_test, axis=0)
        
        # Shuffle combined data
        self.logger.info("Shuffling combined data...")
        X_train_combined, y_train_combined = shuffle(
            X_train_combined, y_train_combined, random_state=self.random_seed
        )
        X_test_combined, y_test_combined = shuffle(
            X_test_combined, y_test_combined, random_state=self.random_seed
        )
        
        # Calculate statistics
        y_train_labels = np.argmax(y_train_combined, axis=1)
        y_test_labels = np.argmax(y_test_combined, axis=1)
        
        combined_stats = {
            'dataset_name': output_name,
            'included_datasets': valid_datasets,
            'creation_time': datetime.now().isoformat(),
            'total_samples': len(X_train_combined) + len(X_test_combined),
            'train_samples': len(X_train_combined),
            'test_samples': len(X_test_combined),
            'train_bonafide': int(np.sum(y_train_labels == 0)),
            'train_attack': int(np.sum(y_train_labels == 1)),
            'test_bonafide': int(np.sum(y_test_labels == 0)),
            'test_attack': int(np.sum(y_test_labels == 1)),
            'train_balance_ratio': float(np.sum(y_train_labels == 1) / len(y_train_labels)),
            'test_balance_ratio': float(np.sum(y_test_labels == 1) / len(y_test_labels)),
            'balance_across_datasets': self.balance_across_datasets,
            'samples_per_dataset': {
                dataset: {
                    'train': len(all_X_train[i]),
                    'test': len(all_X_test[i])
                }
                for i, dataset in enumerate(valid_datasets)
            },
            'input_shape': X_train_combined.shape[1:],
            'original_dataset_stats': dataset_stats
        }
        
        # Save combined dataset
        output_path = Path(f"data/processed/{output_name.lower()}.pkl")
        self.logger.info(f"\nSaving combined dataset to {output_path}")
        
        dataset = (X_train_combined, X_test_combined, y_train_combined, y_test_combined)
        with open(output_path, "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save statistics
        stats_path = Path(f"data/processed/{output_name.lower()}_stats.json")
        with open(stats_path, "w") as f:
            json.dump(combined_stats, f, indent=2, ensure_ascii=False)
        
        # Print summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"SUCCESS: Combined dataset '{output_name}' created!")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Included datasets: {', '.join(valid_datasets)}")
        self.logger.info(f"Total samples: {combined_stats['total_samples']:,}")
        self.logger.info(f"Train: {combined_stats['train_samples']:,} "
                        f"(Bonafide: {combined_stats['train_bonafide']}, "
                        f"Attack: {combined_stats['train_attack']})")
        self.logger.info(f"Test: {combined_stats['test_samples']:,} "
                        f"(Bonafide: {combined_stats['test_bonafide']}, "
                        f"Attack: {combined_stats['test_attack']})")
        self.logger.info(f"Balance ratio - Train: {combined_stats['train_balance_ratio']:.2%}, "
                        f"Test: {combined_stats['test_balance_ratio']:.2%}")
        
        if self.balance_across_datasets:
            self.logger.info("\nSamples per dataset (after balancing):")
            for dataset, counts in combined_stats['samples_per_dataset'].items():
                self.logger.info(f"  {dataset}: Train={counts['train']}, Test={counts['test']}")
        
        return True
    
    def create_multiple_combinations(self) -> Dict[str, bool]:
        """Create multiple dataset combinations for experiments."""
        results = {}
        
        # All available datasets
        all_datasets = ["3dmad", "csmad", "replay_attack", "msspoof", "our_dataset"]
        
        # 1. All datasets combined
        self.logger.info("\n" + "="*70)
        self.logger.info("Creating ALL DATASETS COMBINED")
        self.logger.info("="*70)
        results['combined_all'] = self.create_combined_dataset(
            all_datasets, 
            "combined_all"
        )
        
        # 2. Leave-one-out combinations (for testing generalization)
        for exclude_dataset in all_datasets:
            combo_name = f"combined_without_{exclude_dataset}"
            include_datasets = [d for d in all_datasets if d != exclude_dataset]
            
            self.logger.info("\n" + "="*70)
            self.logger.info(f"Creating combination WITHOUT {exclude_dataset.upper()}")
            self.logger.info("="*70)
            
            results[combo_name] = self.create_combined_dataset(
                include_datasets,
                combo_name
            )
        
        # 3. High-quality only combination (MSSpoof + CSMAD)
        self.logger.info("\n" + "="*70)
        self.logger.info("Creating HIGH-QUALITY combination (MSSpoof + CSMAD)")
        self.logger.info("="*70)
        results['combined_high_quality'] = self.create_combined_dataset(
            ["msspoof", "csmad"],
            "combined_high_quality"
        )
        
        # 4. Video-based combination (Replay-Attack + Our Dataset)
        self.logger.info("\n" + "="*70)
        self.logger.info("Creating VIDEO-BASED combination")
        self.logger.info("="*70)
        results['combined_video'] = self.create_combined_dataset(
            ["replay_attack", "our_dataset"],
            "combined_video"
        )
        
        return results


def main():
    """Main function to create combined datasets."""
    print("\n" + "="*70)
    print("COMBINED DATASET CREATOR")
    print("Creating unified datasets for improved generalization")
    print("="*70)
    
    # Configuration
    TRAIN_RATIO = 0.8
    BALANCE_ACROSS_DATASETS = True  # Important for fair representation
    MAX_SAMPLES_PER_DATASET = None  # None = use median, or set specific number
    RANDOM_SEED = 42
    
    # Create dataset creator
    creator = CombinedDatasetCreator(
        train_ratio=TRAIN_RATIO,
        balance_across_datasets=BALANCE_ACROSS_DATASETS,
        max_samples_per_dataset=MAX_SAMPLES_PER_DATASET,
        random_seed=RANDOM_SEED
    )
    
    # Create main combined dataset (all datasets)
    print("\nCreating main combined dataset...")
    success = creator.create_combined_dataset(
        dataset_names=["3dmad", "csmad", "replay_attack", "msspoof", "our_dataset"],
        output_name="combined_all"
    )
    
    if success:
        print("\n✅ Combined dataset created successfully!")
        print("\nYou can also create additional combinations:")
        print("1. Leave-one-out combinations (for robustness testing)")
        print("2. High-quality only (MSSpoof + CSMAD)")
        print("3. Video-based only (Replay-Attack + Our Dataset)")
        
        create_all = input("\nCreate all combinations? (y/n): ").lower().strip()
        
        if create_all == 'y':
            results = creator.create_multiple_combinations()
            
            print("\n" + "="*70)
            print("SUMMARY OF ALL COMBINATIONS")
            print("="*70)
            for combo_name, success in results.items():
                status = "✅" if success else "❌"
                print(f"{status} {combo_name}")
    else:
        print("\n❌ Failed to create combined dataset")
        return False
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("1. Train models on combined dataset: python scripts/train_combined_models.py")
    print("2. Evaluate cross-dataset performance")
    print("3. Compare with single-dataset results")
    print("="*70)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logging.exception("Fatal error in main")
        sys.exit(1)
