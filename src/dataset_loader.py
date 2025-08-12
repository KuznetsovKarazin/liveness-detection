"""
Dataset Loader with Advanced Augmentation and Data Leak Prevention

This module provides efficient data loading with augmentation support,
proper train/val/test splitting, and data leak prevention mechanisms.
"""

import pickle
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple, Dict, Optional, List, Union, Any
import logging
import json
from dataclasses import dataclass
from enum import Enum
import albumentations as A
from albumentations import (
    RandomRotate90, Flip, Transpose, GridDistortion, OpticalDistortion,
    HueSaturationValue, RandomBrightnessContrast, GaussNoise, MotionBlur,
    MedianBlur, Blur, CLAHE, Sharpen, Emboss, RandomGamma, CoarseDropout,
    Cutout, ChannelShuffle, RGBShift, Downscale, ImageCompression
)


class DatasetSource(Enum):
    """Available dataset sources"""
    MSSPOOF = "msspoof"
    THREEDMAD = "3dmad"
    CSMAD = "csmad"
    REPLAY_ATTACK = "replay_attack"
    OUR_DATASET = "our_dataset"
    COMBINED_ALL = "combined_all"


@dataclass
class DatasetConfig:
    """Configuration for dataset loading"""
    batch_size: int = 32
    shuffle: bool = True
    augment: bool = True
    augmentation_level: str = "medium"  # light, medium, heavy
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    prefetch_buffer: int = tf.data.AUTOTUNE
    cache: bool = True
    num_parallel_calls: int = tf.data.AUTOTUNE
    mixed_precision: bool = False
    normalize: bool = True
    class_weights: Optional[Dict[int, float]] = None


class AdvancedAugmentation:
    """Advanced augmentation pipeline with multiple strategies"""
    
    @staticmethod
    def get_augmentation_pipeline(level: str = "medium") -> A.Compose:
        """Get augmentation pipeline based on level"""
        
        if level == "light":
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            ])
        
        elif level == "medium":
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.Transpose(p=0.3),
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ], p=0.4),
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.Blur(blur_limit=5, p=1.0),
                ], p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                A.ImageCompression(quality_lower=70, quality_upper=90, p=0.3),
            ])
        
        else:  # heavy
            return A.Compose([
                A.RandomRotate90(p=0.7),
                A.Flip(p=0.7),
                A.Transpose(p=0.5),
                A.OneOf([
                    A.GridDistortion(num_steps=7, distort_limit=0.5, p=1.0),
                    A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.RandomGamma(gamma_limit=(50, 150), p=1.0),
                ], p=0.7),
                A.OneOf([
                    A.GaussNoise(var_limit=(20.0, 50.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.10), intensity=(0.2, 0.7), p=1.0),
                    A.MultiplicativeNoise(multiplier=[0.8, 1.2], p=1.0),
                ], p=0.6),
                A.OneOf([
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=7, p=1.0),
                    A.Blur(blur_limit=7, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.CLAHE(clip_limit=4.0, p=1.0),
                    A.Sharpen(alpha=(0.3, 0.7), lightness=(0.5, 1.0), p=1.0),
                    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1.0),
                    A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=1.0),
                ], p=0.4),
                A.OneOf([
                    A.ChannelShuffle(p=1.0),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                    A.ChannelDropout(channel_drop_range=(1, 1), fill_value=128, p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.Downscale(scale_min=0.5, scale_max=0.9, p=1.0),
                    A.ImageCompression(quality_lower=50, quality_upper=80, p=1.0),
                ], p=0.4),
            ])


class DatasetLoader:
    """Enhanced dataset loader with augmentation and data leak prevention"""
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        """Initialize dataset loader"""
        self.config = config or DatasetConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
        if self.config.mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        self.augmentation_pipeline = None
        if self.config.augment:
            self.augmentation_pipeline = AdvancedAugmentation.get_augmentation_pipeline(
                self.config.augmentation_level
            )
        
        self._data_info = {}
        self._subject_splits = {}
    
    def _setup_logging(self):
        """Setup logging configuration"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
    
    def load_dataset(self, dataset_name: Union[str, DatasetSource], 
                    ensure_no_leak: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
        """
        Load dataset with proper train/val/test split
        
        Args:
            dataset_name: Name of the dataset to load
            ensure_no_leak: Whether to ensure no data leakage between splits
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset, info_dict)
        """
        if isinstance(dataset_name, str):
            dataset_name = DatasetSource(dataset_name.lower())
        
        self.logger.info(f"Loading dataset: {dataset_name.value}")
        
        # Load data from pickle
        data_path = Path(f"data/processed/{dataset_name.value}.pkl")
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        with open(data_path, "rb") as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
        
        # Load dataset statistics
        stats_path = Path(f"data/processed/{dataset_name.value}_stats.json")
        if stats_path.exists():
            with open(stats_path, "r") as f:
                dataset_stats = json.load(f)
        else:
            dataset_stats = {}
        
        # Ensure proper data types
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        # Split training data into train/val
        if ensure_no_leak:
            X_train_split, X_val_split, y_train_split, y_val_split = self._split_with_leak_prevention(
                X_train, y_train, dataset_name
            )
        else:
            # Simple random split
            val_size = int(len(X_train) * self.config.validation_split)
            indices = np.random.RandomState(self.config.random_seed).permutation(len(X_train))
            
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            
            X_train_split = X_train[train_indices]
            X_val_split = X_train[val_indices]
            y_train_split = y_train[train_indices]
            y_val_split = y_train[val_indices]
        
        # Calculate class weights if needed
        if self.config.class_weights is None:
            self.config.class_weights = self._calculate_class_weights(y_train_split)
        
        # Create TensorFlow datasets
        train_dataset = self._create_tf_dataset(X_train_split, y_train_split, is_training=True)
        val_dataset = self._create_tf_dataset(X_val_split, y_val_split, is_training=False)
        test_dataset = self._create_tf_dataset(X_test, y_test, is_training=False)
        
        # Prepare info dictionary
        info = {
            "dataset_name": dataset_name.value,
            "train_samples": len(X_train_split),
            "val_samples": len(X_val_split),
            "test_samples": len(X_test),
            "original_train_samples": len(X_train),
            "input_shape": X_train.shape[1:],
            "num_classes": y_train.shape[1],
            "class_weights": self.config.class_weights,
            "dataset_stats": dataset_stats,
            "augmentation": self.config.augment,
            "augmentation_level": self.config.augmentation_level if self.config.augment else None,
            "normalization": self.config.normalize,
            "train_class_distribution": self._get_class_distribution(y_train_split),
            "val_class_distribution": self._get_class_distribution(y_val_split),
            "test_class_distribution": self._get_class_distribution(y_test),
        }
        
        self._data_info[dataset_name.value] = info
        
        self.logger.info(f"Dataset loaded successfully:")
        self.logger.info(f"  Train: {info['train_samples']} samples")
        self.logger.info(f"  Val: {info['val_samples']} samples")
        self.logger.info(f"  Test: {info['test_samples']} samples")
        self.logger.info(f"  Input shape: {info['input_shape']}")
        self.logger.info(f"  Classes: {info['num_classes']}")
        
        return train_dataset, val_dataset, test_dataset, info
    
    def _split_with_leak_prevention(self, X: np.ndarray, y: np.ndarray, 
                                   dataset_name: DatasetSource) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data ensuring no subject/session appears in multiple splits
        
        This is crucial for video-based datasets where frames from the same
        video/subject should not appear in different splits
        """
        # For video datasets, we need to ensure frames from same video don't leak
        if dataset_name in [DatasetSource.REPLAY_ATTACK, DatasetSource.OUR_DATASET]:
            # Use hash-based grouping to identify frames from same source
            # This is a simplified approach - in production, you'd want subject IDs
            
            # Create pseudo-subject IDs based on sequential frame grouping
            samples_per_video = 20  # Approximate based on your frame extraction
            num_videos = len(X) // samples_per_video
            
            subject_ids = np.repeat(np.arange(num_videos), samples_per_video)
            if len(subject_ids) < len(X):
                # Handle remainder
                subject_ids = np.concatenate([
                    subject_ids,
                    np.full(len(X) - len(subject_ids), num_videos)
                ])
            
            # Get unique subjects
            unique_subjects = np.unique(subject_ids)
            np.random.RandomState(self.config.random_seed).shuffle(unique_subjects)
            
            # Split subjects
            val_size = int(len(unique_subjects) * self.config.validation_split)
            val_subjects = unique_subjects[:val_size]
            train_subjects = unique_subjects[val_size:]
            
            # Create masks
            val_mask = np.isin(subject_ids, val_subjects)
            train_mask = np.isin(subject_ids, train_subjects)
            
            # Apply masks
            X_train = X[train_mask]
            X_val = X[val_mask]
            y_train = y[train_mask]
            y_val = y[val_mask]
            
            self.logger.info(f"Leak prevention: Split {len(unique_subjects)} subjects")
            self.logger.info(f"  Train subjects: {len(train_subjects)}")
            self.logger.info(f"  Val subjects: {len(val_subjects)}")
        
        else:
            # For image datasets, standard stratified split is fine
            from sklearn.model_selection import train_test_split
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config.validation_split,
                random_state=self.config.random_seed,
                stratify=np.argmax(y, axis=1)
            )
        
        return X_train, X_val, y_train, y_val
    
    def _create_tf_dataset(self, X: np.ndarray, y: np.ndarray, 
                          is_training: bool = True) -> tf.data.Dataset:
        """Create optimized TensorFlow dataset"""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        # Apply augmentation if training
        if is_training and self.config.augment:
            dataset = dataset.map(
                self._augment_image,
                num_parallel_calls=self.config.num_parallel_calls
            )
        
        # Normalize if needed
        if self.config.normalize:
            dataset = dataset.map(
                self._normalize_image,
                num_parallel_calls=self.config.num_parallel_calls
            )
        
        # Cache before shuffling for better performance
        if self.config.cache:
            dataset = dataset.cache()
        
        # Shuffle if training
        if is_training and self.config.shuffle:
            dataset = dataset.shuffle(buffer_size=min(10000, len(X)))
        
        # Batch
        dataset = dataset.batch(self.config.batch_size)
        
        # Prefetch
        dataset = dataset.prefetch(self.config.prefetch_buffer)
        
        return dataset
    
    def _augment_image(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply augmentation to image"""
        def augment_fn(img):
            # Convert to numpy for albumentations
            img_np = img.numpy()
            
            # Ensure correct format (0-255 uint8)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            
            # Apply augmentation
            augmented = self.augmentation_pipeline(image=img_np)
            img_aug = augmented["image"]
            
            # Convert back to float32
            img_aug = img_aug.astype(np.float32) / 255.0
            
            return img_aug
        
        # Use tf.py_function for numpy operations
        augmented_image = tf.py_function(
            augment_fn,
            [image],
            tf.float32
        )
        
        # Restore shape information
        augmented_image.set_shape(image.shape)
        
        return augmented_image, label
    
    def _normalize_image(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Normalize image to [-1, 1] range"""
        # Assuming images are in [0, 1] range
        normalized_image = (image - 0.5) * 2.0
        return normalized_image, label
    
    def _calculate_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced datasets"""
        class_counts = np.sum(y, axis=0)
        total_samples = len(y)
        num_classes = len(class_counts)
        
        class_weights = {}
        for i in range(num_classes):
            class_weights[i] = total_samples / (num_classes * class_counts[i])
        
        # Normalize weights
        max_weight = max(class_weights.values())
        for i in range(num_classes):
            class_weights[i] = class_weights[i] / max_weight
        
        return class_weights
    
    def _get_class_distribution(self, y: np.ndarray) -> Dict[int, float]:
        """Get class distribution as percentages"""
        class_counts = np.sum(y, axis=0)
        total = np.sum(class_counts)
        
        distribution = {}
        for i, count in enumerate(class_counts):
            distribution[i] = float(count / total)
        
        return distribution
    
    def load_multiple_datasets(self, dataset_names: List[Union[str, DatasetSource]], 
                             combine: bool = False) -> Union[
                                 List[Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]],
                                 Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]
                             ]:
        """
        Load multiple datasets with option to combine them
        
        Args:
            dataset_names: List of dataset names to load
            combine: Whether to combine all datasets into one
            
        Returns:
            Either list of dataset tuples or single combined dataset tuple
        """
        all_datasets = []
        
        for name in dataset_names:
            datasets = self.load_dataset(name)
            all_datasets.append(datasets)
        
        if not combine:
            return all_datasets
        
        # Combine datasets
        self.logger.info("Combining datasets...")
        
        # Collect all data
        X_train_all, y_train_all = [], []
        X_val_all, y_val_all = [], []
        X_test_all, y_test_all = [], []
        
        for train_ds, val_ds, test_ds, info in all_datasets:
            # Convert back to numpy for combining
            for x, y in train_ds.unbatch():
                X_train_all.append(x.numpy())
                y_train_all.append(y.numpy())
            
            for x, y in val_ds.unbatch():
                X_val_all.append(x.numpy())
                y_val_all.append(y.numpy())
            
            for x, y in test_ds.unbatch():
                X_test_all.append(x.numpy())
                y_test_all.append(y.numpy())
        
        # Stack all data
        X_train_combined = np.stack(X_train_all)
        y_train_combined = np.stack(y_train_all)
        X_val_combined = np.stack(X_val_all)
        y_val_combined = np.stack(y_val_all)
        X_test_combined = np.stack(X_test_all)
        y_test_combined = np.stack(y_test_all)
        
        # Shuffle combined training data
        train_indices = np.random.RandomState(self.config.random_seed).permutation(len(X_train_combined))
        X_train_combined = X_train_combined[train_indices]
        y_train_combined = y_train_combined[train_indices]
        
        # Create combined datasets
        train_dataset = self._create_tf_dataset(X_train_combined, y_train_combined, is_training=True)
        val_dataset = self._create_tf_dataset(X_val_combined, y_val_combined, is_training=False)
        test_dataset = self._create_tf_dataset(X_test_combined, y_test_combined, is_training=False)
        
        # Combined info
        combined_info = {
            "dataset_name": "combined",
            "included_datasets": [name.value if isinstance(name, DatasetSource) else name for name in dataset_names],
            "train_samples": len(X_train_combined),
            "val_samples": len(X_val_combined),
            "test_samples": len(X_test_combined),
            "input_shape": X_train_combined.shape[1:],
            "num_classes": y_train_combined.shape[1],
            "class_weights": self._calculate_class_weights(y_train_combined),
            "augmentation": self.config.augment,
            "augmentation_level": self.config.augmentation_level if self.config.augment else None,
        }
        
        self.logger.info(f"Combined dataset created:")
        self.logger.info(f"  Train: {combined_info['train_samples']} samples")
        self.logger.info(f"  Val: {combined_info['val_samples']} samples")
        self.logger.info(f"  Test: {combined_info['test_samples']} samples")
        
        return train_dataset, val_dataset, test_dataset, combined_info
    
    def get_keras_augmentation(self) -> ImageDataGenerator:
        """Get Keras ImageDataGenerator for comparison/compatibility"""
        if self.config.augmentation_level == "light":
            return ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.9, 1.1],
            )
        elif self.config.augmentation_level == "medium":
            return ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.15,
                height_shift_range=0.15,
                shear_range=0.15,
                zoom_range=0.15,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                channel_shift_range=20,
            )
        else:  # heavy
            return ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=[0.7, 1.3],
                channel_shift_range=30,
                fill_mode='reflect',
            )
    
    def save_data_info(self, save_path: Path):
        """Save dataset information for later reference"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w") as f:
            json.dump(self._data_info, f, indent=2)
        
        self.logger.info(f"Dataset info saved to {save_path}")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create loader with custom config
    config = DatasetConfig(
        # batch_size=32,
        augment=True,
        augmentation_level="medium",
        validation_split=0.15,
        mixed_precision=False
    )
    
    loader = DatasetLoader(config)
    
    # Load single dataset
    print("\nLoading single dataset...")
    train_ds, val_ds, test_ds, info = loader.load_dataset(DatasetSource.CSMAD)
    
    print(f"\nDataset info:")
    print(f"  Train batches: {len(list(train_ds.take(1000)))}")
    print(f"  Input shape: {info['input_shape']}")
    print(f"  Class weights: {info['class_weights']}")
    
    # Load multiple datasets
    print("\nLoading multiple datasets...")
    datasets = loader.load_multiple_datasets(
        [DatasetSource.CSMAD, DatasetSource.REPLAY_ATTACK],
        combine=False
    )
    
    print(f"Loaded {len(datasets)} datasets")
    
    # Test augmentation
    print("\nTesting augmentation...")
    for images, labels in train_ds.take(1):
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
        break
