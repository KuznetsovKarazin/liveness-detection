#!/usr/bin/env python3
"""
Datasets Creator

This script creates high-quality balanced datasets with improved preprocessing,
better error handling, and enhanced quality filtering for liveness detection.
"""

import sys
import pickle
import cv2
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils import shuffle
from tqdm import tqdm
import logging
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import Counter
import hashlib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def setup_logging() -> None:
    """Setup comprehensive logging with Windows encoding compatibility."""
    # Ensure logs directory exists
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatter with more detailed info
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
    
    # File handler with UTF-8 encoding and rotation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        f'logs/dataset_creation_{timestamp}.log', 
        encoding='utf-8', 
        mode='w'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler with simpler format
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

class QualityAssessment:
    """Advanced image quality assessment utilities."""
    
    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Calculate image sharpness using multiple methods."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Laplacian variance (primary metric)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Sobel gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2).mean()
        
        # Combined sharpness score
        sharpness_score = (laplacian_var * 0.7 + sobel_magnitude * 0.3)
        
        return float(sharpness_score)
    
    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Calculate image contrast using RMS contrast."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # RMS contrast
        mean_val = np.mean(gray)
        rms_contrast = np.sqrt(np.mean((gray - mean_val) ** 2))
        
        return float(rms_contrast)
    
    @staticmethod
    def calculate_brightness_score(image: np.ndarray) -> Tuple[float, bool]:
        """Calculate brightness score and check if acceptable."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        mean_brightness = np.mean(gray)
        
        # Optimal brightness range: 40-220 (avoiding too dark/bright)
        is_acceptable = 40 <= mean_brightness <= 220
        
        # Brightness score: closer to 128 (middle) is better
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128
        
        return float(brightness_score), is_acceptable
    
    @staticmethod
    def detect_blur_quality(image: np.ndarray) -> Tuple[bool, float]:
        """Detect if image is too blurry using multiple blur detection methods."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Method 1: Laplacian variance (traditional)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Method 2: Tenengrad variance
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = gx**2 + gy**2
        tenengrad_var = np.var(tenengrad)
        
        # Combined blur score
        blur_score = (laplacian_var * 0.6 + tenengrad_var * 0.4) / 1000.0
        
        # Adaptive threshold based on image characteristics
        threshold = 80.0  # Increased threshold for higher quality
        is_sharp = blur_score > threshold
        
        return is_sharp, float(blur_score)

class EnhancedDatasetCreator:
    """Enhanced dataset creator with advanced preprocessing and quality control."""
    
    def __init__(self, target_size=(256, 256), train_ratio=0.8, balance_data=True, 
                 quality_threshold=0.6, max_workers=4):
        """
        Initialize enhanced dataset creator.
        
        Args:
            target_size: Target image size (height, width) - upgraded to 256x256
            train_ratio: Ratio of data for training
            balance_data: Whether to balance classes
            quality_threshold: Minimum quality score for image acceptance
            max_workers: Number of threads for parallel processing
        """
        self.target_size = target_size
        self.train_ratio = train_ratio
        self.balance_data = balance_data
        self.quality_threshold = quality_threshold
        self.max_workers = max_workers
        self.dataset_stats = {}
        self.quality_assessor = QualityAssessment()
        
        # Thread-safe logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = threading.Lock()
        
        # Setup logging
        setup_logging()
        
        # Ensure directories exist
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("data/quality_reports").mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Enhanced DatasetCreator v2.0 initialized:")
        self.logger.info(f"  Target size: {target_size}")
        self.logger.info(f"  Train ratio: {train_ratio}")
        self.logger.info(f"  Balance data: {balance_data}")
        self.logger.info(f"  Quality threshold: {quality_threshold}")
        self.logger.info(f"  Max workers: {max_workers}")
        self.logger.info(f"  Real data only - no synthetic generation")
    
    def advanced_image_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply state-of-the-art image enhancement techniques.
        
        Args:
            image: Input image array
            
        Returns:
            Enhanced image
        """
        enhanced = image.copy().astype(np.float32)
        
        # Step 1: Noise reduction with edge preservation
        enhanced = cv2.bilateralFilter(enhanced.astype(np.uint8), 9, 75, 75)
        enhanced = enhanced.astype(np.float32)
        
        # Step 2: Adaptive histogram equalization in LAB space
        if len(enhanced.shape) == 3:
            lab = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # More aggressive CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            lab[:, :, 0] = l_channel
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
        
        # Step 3: Advanced sharpening with USM (Unsharp Masking)
        gaussian = cv2.GaussianBlur(enhanced.astype(np.uint8), (0, 0), 2.0)
        enhanced_usm = cv2.addWeighted(enhanced.astype(np.uint8), 1.5, gaussian, -0.5, 0)
        
        # Step 4: Gamma correction for better dynamic range
        gamma = 1.2
        gamma_corrected = np.power(enhanced_usm.astype(np.float32) / 255.0, 1.0/gamma)
        gamma_corrected = (gamma_corrected * 255.0).astype(np.uint8)
        
        # Step 5: Final quality enhancement
        final_enhanced = cv2.convertScaleAbs(gamma_corrected, alpha=1.1, beta=5)
        
        return final_enhanced
    
    def comprehensive_quality_check(self, image: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """
        Perform comprehensive image quality assessment.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (is_acceptable, quality_metrics)
        """
        metrics = {}
        
        # Calculate individual quality metrics
        metrics['sharpness'] = self.quality_assessor.calculate_sharpness(image)
        metrics['contrast'] = self.quality_assessor.calculate_contrast(image)
        metrics['brightness_score'], brightness_ok = self.quality_assessor.calculate_brightness_score(image)
        is_sharp, metrics['blur_score'] = self.quality_assessor.detect_blur_quality(image)
        
        # Calculate composite quality score
        quality_weights = {
            'sharpness': 0.35,
            'contrast': 0.25,
            'brightness_score': 0.20,
            'blur_score': 0.20
        }
        
        normalized_metrics = {
            'sharpness': min(metrics['sharpness'] / 100.0, 1.0),
            'contrast': min(metrics['contrast'] / 50.0, 1.0),
            'brightness_score': metrics['brightness_score'],
            'blur_score': min(metrics['blur_score'] / 100.0, 1.0)
        }
        
        composite_score = sum(
            normalized_metrics[metric] * weight 
            for metric, weight in quality_weights.items()
        )
        
        metrics['composite_score'] = composite_score
        
        # Determine if image is acceptable
        is_acceptable = (
            composite_score >= self.quality_threshold and
            brightness_ok and
            is_sharp and
            metrics['contrast'] > 15.0
        )
        
        return is_acceptable, metrics
    
    def process_single_image(self, img_path: Path) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
        """
        Process a single image with comprehensive quality checking.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Tuple of (processed_image, quality_metrics) or None if rejected
        """
        try:
            # Load image with multiple fallback methods
            img = cv2.imread(str(img_path))
            
            if img is None:
                # Try grayscale loading for special formats
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            if img is None:
                self.logger.warning(f"Could not load image: {img_path}")
                return None
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Check original quality
            is_acceptable, quality_metrics = self.comprehensive_quality_check(img_rgb)
            
            if not is_acceptable:
                return None
            
            # High-quality resize with LANCZOS interpolation
            img_resized = cv2.resize(
                img_rgb, 
                self.target_size, 
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Apply advanced enhancement
            img_enhanced = self.advanced_image_enhancement(img_resized)
            
            # Final quality check after enhancement
            is_final_acceptable, final_metrics = self.comprehensive_quality_check(img_enhanced)
            
            if not is_final_acceptable:
                return None
            
            # Combine metrics
            combined_metrics = {**quality_metrics, **{f"final_{k}": v for k, v in final_metrics.items()}}
            
            return img_enhanced, combined_metrics
            
        except Exception as e:
            self.logger.error(f"Error processing {img_path}: {e}")
            return None
    
    def load_images_from_directory_parallel(self, image_dir: Path, class_name: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Load and enhance images from directory with parallel processing.
        
        Args:
            image_dir: Directory containing images
            class_name: Name of the class (for logging)
            
        Returns:
            Tuple of (enhanced_images_array, quality_metrics_list)
        """
        if not image_dir.exists():
            self.logger.warning(f"Directory {image_dir} does not exist")
            return np.array([]), []
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.pgm', '.tiff', '.tif'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        self.logger.info(f"Found {len(image_files)} images in {image_dir.name} ({class_name})")
        
        if len(image_files) == 0:
            self.logger.warning(f"No images found in {image_dir}")
            return np.array([]), []
        
        images = []
        quality_metrics = []
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_image, img_path): img_path 
                for img_path in image_files
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_path), 
                             total=len(image_files), 
                             desc=f"Processing {class_name}"):
                
                img_path = future_to_path[future]
                try:
                    result = future.result()
                    if result is not None:
                        img_enhanced, metrics = result
                        images.append(img_enhanced)
                        quality_metrics.append(metrics)
                except Exception as e:
                    self.logger.error(f"Error processing {img_path}: {e}")
        
        success_rate = len(images) / len(image_files) if image_files else 0
        self.logger.info(f"Successfully processed {len(images)}/{len(image_files)} images from {class_name}")
        self.logger.info(f"Success rate: {success_rate:.2%}")
        
        if quality_metrics:
            avg_quality = np.mean([m['composite_score'] for m in quality_metrics])
            self.logger.info(f"Average quality score: {avg_quality:.3f}")
        
        return np.array(images, dtype=np.uint8), quality_metrics
    
    def extract_video_frames_enhanced(self, video_dir: Path, class_name: str, 
                                    max_frames_per_video=20) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract high-quality frames from videos with enhanced processing.
        
        Args:
            video_dir: Directory containing videos
            class_name: Name of the class
            max_frames_per_video: Maximum frames to extract per video
            
        Returns:
            Tuple of (enhanced_frames_array, quality_metrics_list)
        """
        if not video_dir.exists():
            self.logger.warning(f"Directory {video_dir} does not exist")
            return np.array([]), []
        
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(ext))
            video_files.extend(video_dir.glob(ext.upper()))
        
        self.logger.info(f"Found {len(video_files)} videos in {video_dir.name} ({class_name})")
        
        if len(video_files) == 0:
            self.logger.warning(f"No videos found in {video_dir}")
            return np.array([]), []
        
        all_frames = []
        all_quality_metrics = []
        
        for video_path in tqdm(video_files, desc=f"Processing {class_name} videos"):
            try:
                frames, metrics = self._process_single_video(video_path, max_frames_per_video)
                all_frames.extend(frames)
                all_quality_metrics.extend(metrics)
                
            except Exception as e:
                self.logger.error(f"Error processing video {video_path}: {e}")
                continue
        
        self.logger.info(f"Successfully extracted {len(all_frames)} high-quality frames from {class_name}")
        
        return np.array(all_frames, dtype=np.uint8), all_quality_metrics
    
    def _process_single_video(self, video_path: Path, max_frames_per_video: int) -> Tuple[List[np.ndarray], List[Dict]]:
        """Process a single video file and extract quality frames."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            self.logger.warning(f"Could not open video: {video_path}")
            return [], []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames < 10:
            self.logger.warning(f"Video too short: {video_path}")
            cap.release()
            return [], []
        
        # Smart frame sampling strategy
        start_frame = max(15, int(0.15 * total_frames))
        end_frame = min(total_frames - 15, int(0.85 * total_frames))
        
        if end_frame <= start_frame:
            frame_indices = [total_frames // 2]
        else:
            # Use temporal diversity sampling
            step = max(1, (end_frame - start_frame) // max_frames_per_video)
            frame_indices = list(range(start_frame, end_frame, step))[:max_frames_per_video]
        
        frames = []
        quality_metrics = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Check frame quality
                is_acceptable, metrics = self.comprehensive_quality_check(frame_rgb)
                
                if is_acceptable:
                    # High-quality resize
                    frame_resized = cv2.resize(
                        frame_rgb, 
                        self.target_size, 
                        interpolation=cv2.INTER_LANCZOS4
                    )
                    
                    # Apply video-specific enhancement
                    frame_enhanced = self.enhance_video_frame(frame_resized)
                    
                    frames.append(frame_enhanced)
                    quality_metrics.append(metrics)
        
        cap.release()
        return frames, quality_metrics
    
    def enhance_video_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply video-specific enhancement to reduce compression artifacts."""
        enhanced = frame.copy()
        
        # Step 1: Temporal noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 80, 80)
        
        # Step 2: Advanced denoising for video artifacts
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # Step 3: Enhance contrast specifically for video compression artifacts
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        lab[:, :, 0] = l_channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Step 4: Mild sharpening optimized for video
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) * 0.6
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend for subtle enhancement
        enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return enhanced
    
    def intelligent_class_balancing(self, bonafide_data: np.ndarray, attack_data: np.ndarray, 
                                  bonafide_metrics: List[Dict], attack_metrics: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[Dict]]:
        """
        Intelligently balance classes based on quality scores.
        
        Args:
            bonafide_data: Bonafide images array
            attack_data: Attack images array
            bonafide_metrics: Quality metrics for bonafide images
            attack_metrics: Quality metrics for attack images
            
        Returns:
            Balanced arrays and metrics
        """
        if not self.balance_data:
            return bonafide_data, attack_data, bonafide_metrics, attack_metrics
        
        bonafide_count = len(bonafide_data)
        attack_count = len(attack_data)
        
        self.logger.info(f"Original counts - Bonafide: {bonafide_count}, Attack: {attack_count}")
        
        if bonafide_count == 0 or attack_count == 0:
            self.logger.error("One of the classes has no data!")
            return bonafide_data, attack_data, bonafide_metrics, attack_metrics
        
        if bonafide_count == attack_count:
            return bonafide_data, attack_data, bonafide_metrics, attack_metrics
        
        # Use quality-aware balancing
        target_count = min(bonafide_count, attack_count)
        self.logger.info(f"Balancing to {target_count} samples per class using quality-aware selection")
        
        # Quality-aware undersampling
        if bonafide_count > target_count:
            quality_scores = [m['composite_score'] for m in bonafide_metrics]
            indices = np.argsort(quality_scores)[::-1][:target_count]  # Take highest quality
            bonafide_data = bonafide_data[indices]
            bonafide_metrics = [bonafide_metrics[i] for i in indices]
        
        if attack_count > target_count:
            quality_scores = [m['composite_score'] for m in attack_metrics]
            indices = np.argsort(quality_scores)[::-1][:target_count]  # Take highest quality
            attack_data = attack_data[indices]
            attack_metrics = [attack_metrics[i] for i in indices]
        
        self.logger.info(f"Balanced counts - Bonafide: {len(bonafide_data)}, Attack: {len(attack_data)}")
        return bonafide_data, attack_data, bonafide_metrics, attack_metrics
    
    def create_dataset(self, dataset_name: str, base_dir: Path, is_video: bool = False) -> bool:
        """
        Create a high-quality dataset from real data with comprehensive processing.
        
        Args:
            dataset_name: Name of the dataset
            base_dir: Base directory containing the data
            is_video: Whether this is a video-based dataset
            
        Returns:
            Success status
        """
        self.logger.info(f"Creating enhanced {dataset_name} dataset...")
        
        try:
            # Load all data with quality metrics
            if is_video:
                self.logger.info(f"Processing VIDEO dataset: {dataset_name}")
                all_attack, attack_metrics = self._load_video_data(base_dir, "attack")
                all_bonafide, bonafide_metrics = self._load_video_data(base_dir, "bonafide")
            else:
                self.logger.info(f"Processing IMAGE dataset: {dataset_name}")
                all_attack, attack_metrics = self._load_image_data(base_dir, "attack")
                all_bonafide, bonafide_metrics = self._load_image_data(base_dir, "bonafide")
            
            # Check if we have data
            if len(all_attack) == 0 or len(all_bonafide) == 0:
                self.logger.error(f"No data found for {dataset_name}")
                return False
            
            # Intelligent class balancing
            all_bonafide, all_attack, bonafide_metrics, attack_metrics = self.intelligent_class_balancing(
                all_bonafide, all_attack, bonafide_metrics, attack_metrics
            )
            
            # Create final dataset
            success = self._finalize_dataset(
                dataset_name, all_bonafide, all_attack, 
                bonafide_metrics, attack_metrics, is_video
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"FAILED to create {dataset_name}: {e}")
            return False
    
    def _load_image_data(self, base_dir: Path, class_type: str) -> Tuple[np.ndarray, List[Dict]]:
        """Load image data from training and validation directories."""
        all_data = []
        all_metrics = []
        
        for split in ["training", "validation"]:
            dir_path = base_dir / f"{class_type}_{split}"
            if dir_path.exists():
                data, metrics = self.load_images_from_directory_parallel(dir_path, f"{class_type}_{split}")
                all_data.extend(data)
                all_metrics.extend(metrics)
        
        return np.array(all_data), all_metrics
    
    def _load_video_data(self, base_dir: Path, class_type: str) -> Tuple[np.ndarray, List[Dict]]:
        """Load video data from training and validation directories."""
        all_data = []
        all_metrics = []
        
        for split in ["training", "validation"]:
            dir_path = base_dir / f"{class_type}_{split}"
            if dir_path.exists():
                data, metrics = self.extract_video_frames_enhanced(dir_path, f"{class_type}_{split}")
                all_data.extend(data)
                all_metrics.extend(metrics)
        
        return np.array(all_data), all_metrics
    
    def _finalize_dataset(self, dataset_name: str, bonafide_data: np.ndarray, attack_data: np.ndarray,
                         bonafide_metrics: List[Dict], attack_metrics: List[Dict], is_video: bool) -> bool:
        """Finalize dataset creation with proper splitting and saving."""
        
        # Combine all data
        X_all = np.concatenate([bonafide_data, attack_data], axis=0)
        y_all = np.concatenate([
            np.zeros(len(bonafide_data)),  # bonafide = 0
            np.ones(len(attack_data))      # attack = 1
        ])
        
        # Combine metrics
        all_metrics = bonafide_metrics + attack_metrics
        
        # Shuffle all data maintaining correspondence
        indices = np.arange(len(X_all))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        X_all = X_all[indices]
        y_all = y_all[indices]
        all_metrics = [all_metrics[i] for i in indices]
        
        # Stratified split to maintain class balance
        splitter = StratifiedShuffleSplit(
            n_splits=1, 
            train_size=self.train_ratio, 
            random_state=42
        )
        
        train_idx, test_idx = next(splitter.split(X_all, y_all))
        
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train_labels, y_test_labels = y_all[train_idx], y_all[test_idx]
        
        # Convert to one-hot encoding
        y_train = np.eye(2)[y_train_labels.astype(int)]
        y_test = np.eye(2)[y_test_labels.astype(int)]
        
        # Normalize images to [0, 1] range
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        
        # Calculate comprehensive statistics
        train_metrics = [all_metrics[i] for i in train_idx]
        test_metrics = [all_metrics[i] for i in test_idx]
        
        stats = {
            'dataset_name': dataset_name,
            'creation_time': datetime.now().isoformat(),
            'target_size': self.target_size,
            'train_ratio': self.train_ratio,
            'quality_threshold': self.quality_threshold,
            'total_samples': len(X_all),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_bonafide': int(np.sum(y_train_labels == 0)),
            'train_attack': int(np.sum(y_train_labels == 1)),
            'test_bonafide': int(np.sum(y_test_labels == 0)),
            'test_attack': int(np.sum(y_test_labels == 1)),
            'train_balance_ratio': float(np.sum(y_train_labels == 1) / len(y_train_labels)),
            'test_balance_ratio': float(np.sum(y_test_labels == 1) / len(y_test_labels)),
            'data_type': 'video' if is_video else 'image',
            'balanced': self.balance_data,
            'quality_enhanced': True,
            'synthetic_data': False,
            'avg_train_quality': float(np.mean([m['composite_score'] for m in train_metrics])),
            'avg_test_quality': float(np.mean([m['composite_score'] for m in test_metrics])),
            'quality_std': float(np.std([m['composite_score'] for m in all_metrics])),
            'min_quality': float(min(m['composite_score'] for m in all_metrics)),
            'max_quality': float(max(m['composite_score'] for m in all_metrics))
        }
        
        self.dataset_stats[dataset_name] = stats
        
        # Save dataset
        dataset = (X_train, X_test, y_train, y_test)
        output_path = f"data/processed/{dataset_name.lower()}.pkl"
        
        # Use more efficient pickle protocol
        with open(output_path, "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save detailed statistics
        with open(f"data/processed/{dataset_name.lower()}_stats.json", "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Save quality metrics for analysis
        quality_report = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'dataset_stats': stats
        }
        
        with open(f"data/quality_reports/{dataset_name.lower()}_quality.json", "w") as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"SUCCESS: {dataset_name} created with enhanced quality!")
        self.logger.info(f"   Total: {len(X_all)} samples")
        self.logger.info(f"   Train: {X_train.shape} (Bonafide: {stats['train_bonafide']}, Attack: {stats['train_attack']})")
        self.logger.info(f"   Test: {X_test.shape} (Bonafide: {stats['test_bonafide']}, Attack: {stats['test_attack']})")
        self.logger.info(f"   Train/Test ratio: {self.train_ratio:.1%}/{1-self.train_ratio:.1%}")
        self.logger.info(f"   Image size: {self.target_size}")
        self.logger.info(f"   Average quality: {stats['avg_train_quality']:.3f} (train), {stats['avg_test_quality']:.3f} (test)")
        
        return True
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive summary report with detailed analytics."""
        if not self.dataset_stats:
            self.logger.warning("No datasets created yet.")
            return
        
        total_datasets = len(self.dataset_stats)
        total_train_samples = sum(stats['train_samples'] for stats in self.dataset_stats.values())
        total_test_samples = sum(stats['test_samples'] for stats in self.dataset_stats.values())
        total_samples = sum(stats['total_samples'] for stats in self.dataset_stats.values())
        
        # Calculate overall quality statistics
        all_qualities = []
        for stats in self.dataset_stats.values():
            if 'avg_train_quality' in stats:
                all_qualities.append(stats['avg_train_quality'])
            if 'avg_test_quality' in stats:
                all_qualities.append(stats['avg_test_quality'])
        
        avg_overall_quality = np.mean(all_qualities) if all_qualities else 0
        
        report = {
            'summary': {
                'total_datasets': total_datasets,
                'total_samples': total_samples,
                'total_train_samples': total_train_samples,
                'total_test_samples': total_test_samples,
                'train_ratio': self.train_ratio,
                'creation_time': datetime.now().isoformat(),
                'target_size': self.target_size,
                'quality_threshold': self.quality_threshold,
                'balanced': self.balance_data,
                'quality_enhanced': True,
                'synthetic_data_used': False,
                'real_data_only': True,
                'avg_quality_score': float(avg_overall_quality),
                'processing_version': '2.0_enhanced'
            },
            'datasets': self.dataset_stats,
            'quality_analysis': {
                'threshold_used': self.quality_threshold,
                'average_quality': float(avg_overall_quality),
                'quality_range': {
                    'min': float(min(stats.get('min_quality', 0) for stats in self.dataset_stats.values())),
                    'max': float(max(stats.get('max_quality', 1) for stats in self.dataset_stats.values()))
                }
            }
        }
        
        # Save comprehensive report
        with open("data/processed/datasets_summary_enhanced.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        # Print enhanced summary
        print("\n" + "="*70)
        print("ENHANCED REAL DATASETS CREATION SUMMARY v2.0")
        print("="*70)
        print(f"Total datasets created: {total_datasets}")
        print(f"Total samples: {total_samples:,}")
        print(f"Training samples: {total_train_samples:,} ({self.train_ratio:.1%})")
        print(f"Test samples: {total_test_samples:,} ({1-self.train_ratio:.1%})")
        print(f"Image size: {self.target_size}")
        print(f"Quality threshold: {self.quality_threshold}")
        print(f"Average quality score: {avg_overall_quality:.3f}")
        print(f"Data balancing: {'Enabled' if self.balance_data else 'Disabled'}")
        print(f"Advanced enhancement: Enabled")
        print(f"Parallel processing: {self.max_workers} workers")
        print(f"Real data only: Yes (no synthetic data)")
        
        print("\nPer-dataset statistics:")
        for name, stats in self.dataset_stats.items():
            print(f"\n{name}:")
            print(f"  Total: {stats['total_samples']:,} samples")
            print(f"  Train: {stats['train_samples']:,} samples ({stats['train_bonafide']} bonafide, {stats['train_attack']} attack)")
            print(f"  Test:  {stats['test_samples']:,} samples ({stats['test_bonafide']} bonafide, {stats['test_attack']} attack)")
            print(f"  Balance: {stats['train_balance_ratio']:.2%} attack ratio")
            print(f"  Type: {stats['data_type']}")
            print(f"  Quality: {stats.get('avg_train_quality', 0):.3f} (train), {stats.get('avg_test_quality', 0):.3f} (test)")
            print(f"  Enhanced: {stats['quality_enhanced']}")
    
    def _generate_markdown_report(self, report: Dict[str, Any]):
        """Generate a detailed markdown report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        markdown_content = f"""# Enhanced Dataset Creation Report v2.0

**Generated:** {timestamp}  
**Processing Version:** 2.0 Enhanced  
**Target Resolution:** {self.target_size[0]}×{self.target_size[1]}  
**Quality Threshold:** {self.quality_threshold}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Datasets | {report['summary']['total_datasets']} |
| Total Samples | {report['summary']['total_samples']:,} |
| Training Samples | {report['summary']['total_train_samples']:,} |
| Test Samples | {report['summary']['total_test_samples']:,} |
| Average Quality Score | {report['summary']['avg_quality_score']:.3f} |
| Quality Threshold | {self.quality_threshold} |

## Dataset Details

"""
        
        for name, stats in report['datasets'].items():
            markdown_content += f"""### {name}

- **Type:** {stats['data_type'].capitalize()}
- **Total Samples:** {stats['total_samples']:,}
- **Training:** {stats['train_samples']:,} ({stats['train_bonafide']} bonafide, {stats['train_attack']} attack)
- **Testing:** {stats['test_samples']:,} ({stats['test_bonafide']} bonafide, {stats['test_attack']} attack)
- **Balance Ratio:** {stats['train_balance_ratio']:.2%} attack
- **Quality Score:** {stats.get('avg_train_quality', 0):.3f} (train), {stats.get('avg_test_quality', 0):.3f} (test)
- **Enhanced Processing:** ✅

"""
        
        markdown_content += f"""## Quality Analysis

- **Threshold Used:** {self.quality_threshold}
- **Average Quality:** {report['quality_analysis']['average_quality']:.3f}
- **Quality Range:** {report['quality_analysis']['quality_range']['min']:.3f} - {report['quality_analysis']['quality_range']['max']:.3f}

## Processing Features

- ✅ **256×256 Resolution** - Enhanced from 128×128
- ✅ **Advanced Quality Filtering** - Multi-metric assessment
- ✅ **Parallel Processing** - {self.max_workers} worker threads
- ✅ **Comprehensive Enhancement** - LAB color space, CLAHE, USM sharpening
- ✅ **Quality-Aware Balancing** - Selects highest quality samples
- ✅ **Video Frame Enhancement** - Specialized processing for video compression artifacts
- ✅ **Detailed Logging** - Thread-safe comprehensive logging
- ✅ **Quality Reports** - Per-dataset quality metrics saved

## Next Steps

1. Review quality reports in `data/quality_reports/`
2. Analyze dataset statistics in `data/processed/datasets_summary_enhanced.json`
3. Run dataset exploration: `jupyter notebook notebooks/01_dataset_exploration.ipynb`
4. Train models: `python scripts/train_all_models.py`

---
*Generated by Enhanced Dataset Creator v2.0*
"""
        
        with open("data/processed/DATASET_REPORT.md", "w", encoding='utf-8') as f:
            f.write(markdown_content)
        
        self.logger.info("Markdown report saved to data/processed/DATASET_REPORT.md")


def main():
    """Main function to create all enhanced datasets with improved processing."""
    print("ENHANCED REAL DATASETS CREATOR v2.0")
    print("Real data only - advanced quality processing")
    print("="*60)
    
    # Enhanced configuration for high quality
    TARGET_SIZE = (256, 256)     # UPGRADED: 256x256 resolution
    TRAIN_RATIO = 0.8            # 80% train, 20% test
    BALANCE_DATA = True          # Enable intelligent class balancing
    QUALITY_THRESHOLD = 0.65     # Higher quality threshold
    MAX_WORKERS = 4              # Parallel processing workers
    
    # Create enhanced dataset creator
    creator = EnhancedDatasetCreator(
        target_size=TARGET_SIZE,
        train_ratio=TRAIN_RATIO,
        balance_data=BALANCE_DATA,
        quality_threshold=QUALITY_THRESHOLD,
        max_workers=MAX_WORKERS
    )
    
    # Dataset configurations - REAL DATA ONLY with enhanced processing
    datasets_config = [
        ("3DMAD", Path("data/3DMAD/images"), False),
        ("CSMAD", Path("data/CSMAD/images"), False),
        ("Replay_Attack", Path("data/Replay_Attack/images"), False),
        ("MSSpoof", Path("data/MSSpoof_dataset/images"), False),
        ("Our_Dataset", Path("data/Our_dataset/videos"), True),  # Video dataset
    ]
    
    # Create datasets with enhanced processing
    successful_datasets = 0
    failed_datasets = []
    
    print(f"\nProcessing {len(datasets_config)} datasets...")
    print(f"Target resolution: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"Quality threshold: {QUALITY_THRESHOLD}")
    print(f"Parallel workers: {MAX_WORKERS}")
    
    for dataset_name, base_dir, is_video in datasets_config:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} {'(VIDEO)' if is_video else '(IMAGES)'}")
        print(f"{'='*60}")
        
        if creator.create_dataset(dataset_name, base_dir, is_video):
            successful_datasets += 1
            print(f"✅ {dataset_name} completed successfully!")
        else:
            failed_datasets.append(dataset_name)
            print(f"❌ {dataset_name} failed!")
    
    # Generate comprehensive report
    creator.generate_comprehensive_report()
    
    # Final summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully created: {successful_datasets}/{len(datasets_config)} datasets")
    
    if failed_datasets:
        print(f"Failed datasets: {', '.join(failed_datasets)}")
    
    print(f"\n📊 Enhanced datasets with 256×256 resolution created!")
    print(f"📈 Quality threshold: {QUALITY_THRESHOLD}")
    print(f"⚡ Parallel processing with {MAX_WORKERS} workers")
    print(f"🔍 Advanced quality filtering applied")
    
    print("\n📁 Files created:")
    print("   - data/processed/{dataset_name}.pkl - Dataset files")
    print("   - data/processed/{dataset_name}_stats.json - Statistics")
    print("   - data/quality_reports/{dataset_name}_quality.json - Quality metrics")
    print("   - data/processed/datasets_summary_enhanced.json - Overall summary")
    print("   - data/processed/DATASET_REPORT.md - Detailed report")
    
    print("\n🚀 Next steps:")
    print("1. Review quality reports in data/quality_reports/")
    print("2. jupyter notebook notebooks/01_dataset_exploration.ipynb")
    print("3. python scripts/train_all_models.py")
    
    if successful_datasets > 0:
        print(f"\n✨ {successful_datasets} enhanced datasets ready for training!")
    
    return successful_datasets > 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logging.exception("Fatal error in main")
        sys.exit(1)
