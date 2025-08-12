"""
Dataset Configuration Settings

This module contains enhanced configuration settings for all datasets used in the 
liveness detection research. Optimized for 256x256 resolution with advanced 
preprocessing parameters and improved dataset-specific settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import json


class DatasetType(Enum):
    """Enumeration of dataset types for different research scenarios."""
    MSSPOOF = "MSSpoof"
    THREEDMAD = "3DMAD" 
    CSMAD = "CSMAD"
    REPLAY_ATTACK = "Replay-Attack"
    OUR_DATASET = "Our"
    COMBINED_ALL = "Combined_All"


class ProcessingMode(Enum):
    """Processing mode for different quality levels."""
    BASIC = "basic"
    ENHANCED = "enhanced" 
    PREMIUM = "premium"


class AugmentationType(Enum):
    """Types of augmentation strategies."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"


@dataclass
class BaseDatasetConfig:
    """Enhanced base configuration class for all datasets - UPGRADED TO 256x256."""
    
    # Data paths
    data_dir: Path = Path("data/processed")
    raw_data_dir: Path = Path("data/raw")
    quality_reports_dir: Path = Path("data/quality_reports")
    
    # Image preprocessing - UPGRADED TO 256x256
    image_size: Tuple[int, int] = (256, 256)  # UPGRADED from 128x128
    channels: int = 3
    normalize: bool = False  
    normalization_method: str = "per_image"  # "per_image", "global", "standardize", "min_max"
    
    # Enhanced quality settings
    quality_threshold: float = 0.65  # Increased from 0.5
    enable_quality_filtering: bool = True
    quality_metrics: List[str] = field(default_factory=lambda: [
        "sharpness", "contrast", "brightness", "blur_detection"
    ])
    
    # Processing configuration
    processing_mode: ProcessingMode = ProcessingMode.ENHANCED
    parallel_workers: int = 4
    memory_efficient: bool = True
    cache_processed: bool = True
    
    # Data splitting - Enhanced with validation split
    train_split: float = 0.7  # Reduced to accommodate validation
    validation_split: float = 0.15  # Added explicit validation split
    test_split: float = 0.15
    random_seed: int = 42
    stratified_split: bool = True  # Ensure balanced splits
    
    # Data loading - Optimized for 256x256
    batch_size: int = 8  # Reduced for larger images
    shuffle_train: bool = True
    shuffle_test: bool = False
    prefetch_buffer: int = 2
    
    # Class balancing - Enhanced options
    balance_classes: bool = True
    balancing_strategy: str = "quality_aware"  # "random", "quality_aware", "temporal"
    undersampling: bool = True  # If False, uses oversampling
    balance_tolerance: float = 0.05  # Acceptable imbalance ratio
    
    # Enhanced augmentation settings
    use_augmentation: bool = True
    augmentation_type: AugmentationType = AugmentationType.ADVANCED
    augmentation_probability: float = 0.6  # Increased for more diversity
    preserve_aspect_ratio: bool = True
    
    # Advanced preprocessing options
    color_space_conversion: Optional[str] = None  # "lab", "hsv", "yuv"
    histogram_equalization: bool = True
    noise_reduction: bool = True
    edge_enhancement: bool = True
    gamma_correction: bool = True
    gamma_value: float = 1.2
    
    # Video-specific settings (when applicable)
    max_frames_per_video: int = 20  # Increased for better sampling
    frame_sampling_strategy: str = "temporal_diverse"  # "uniform", "temporal_diverse", "quality_based"
    temporal_consistency_check: bool = True
    
    # Validation and testing
    cross_validation_folds: int = 5
    enable_cross_dataset_validation: bool = False
    test_time_augmentation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary with proper serialization."""
        result = {}
        for field_info in self.__dataclass_fields__.values():
            value = getattr(self, field_info.name)
            if isinstance(value, (Path, Enum)):
                result[field_info.name] = str(value)
            elif isinstance(value, tuple):
                result[field_info.name] = list(value)
            else:
                result[field_info.name] = value
        return result
    
    def save_config(self, filepath: Path) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, filepath: Path) -> 'BaseDatasetConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert back to proper types
        if 'image_size' in data and isinstance(data['image_size'], list):
            data['image_size'] = tuple(data['image_size'])
        
        if 'data_dir' in data:
            data['data_dir'] = Path(data['data_dir'])
        if 'raw_data_dir' in data:
            data['raw_data_dir'] = Path(data['raw_data_dir'])
        if 'quality_reports_dir' in data:
            data['quality_reports_dir'] = Path(data['quality_reports_dir'])
        
        return cls(**data)


@dataclass
class MSSpoof_Config(BaseDatasetConfig):
    """Enhanced configuration for MSSpoof dataset."""
    
    # Dataset-specific paths
    dataset_name: str = "MSSpoof"
    data_file: str = "msspoof.pkl"
    
    # MSSpoof specific settings - Enhanced
    use_vis_spectrum: bool = True
    use_nir_spectrum: bool = True
    spectrum_fusion: str = "concatenate"  # "concatenate", "average", "separate", "attention"
    nir_enhancement: bool = True  # Enhanced NIR processing
    
    # Image preprocessing specific to MSSpoof - UPGRADED
    image_size: Tuple[int, int] = (256, 256)  # UPGRADED
    channels: int = 3  # RGB for VIS, may adjust for NIR fusion
    normalize: bool = False  
    batch_size: int = 8  # Adjusted for larger images
    quality_threshold: float = 0.7  # Higher threshold for competition dataset
    
    # Enhanced spectrum processing
    vis_preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "contrast_enhancement": True,
        "histogram_equalization": True,
        "noise_reduction": True
    })
    
    nir_preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "intensity_normalization": True,
        "thermal_noise_reduction": True,
        "edge_enhancement": True
    })
    
    # Class distribution (from paper analysis)
    expected_classes: int = 2
    bonafide_label: int = 0
    attack_label: int = 1
    
    # Enhanced quality filtering for competition data
    min_face_size: int = 32  # Increased for 256x256
    face_detection_confidence: float = 0.8
    quality_threshold: float = 0.7
    
    # Protocol settings (following paper methodology)
    protocol: str = "competition"  # "competition", "grandtest"
    cross_validation_folds: int = 5
    
    # Competition-specific settings
    competition_mode: bool = True
    strict_quality_control: bool = True
    multi_spectral_alignment: bool = True


@dataclass  
class ThreeDMAD_Config(BaseDatasetConfig):
    """Enhanced configuration for 3DMAD dataset."""
    
    # Dataset-specific paths
    dataset_name: str = "3DMAD"
    data_file: str = "3dmad.pkl"
    
    # 3DMAD specific settings - Enhanced
    use_rgb: bool = True
    use_depth: bool = False  # Enable for depth-based experiments
    use_eye_positions: bool = False  # Additional modality
    depth_normalization: str = "adaptive"  # "minmax", "standardize", "clip", "adaptive"
    
    # Enhanced Kinect-specific preprocessing
    depth_max_distance: float = 4000.0  # mm
    depth_min_distance: float = 500.0   # mm
    depth_hole_filling: bool = True
    depth_smoothing: bool = True
    depth_edge_preservation: bool = True
    
    # Image preprocessing - UPGRADED
    image_size: Tuple[int, int] = (256, 256)  # UPGRADED
    channels: int = 3  # RGB only, 4 if RGB+D
    normalize: bool = False  
    batch_size: int = 8  # Adjusted
    quality_threshold: float = 0.65
    
    # Enhanced session handling (3DMAD has 3 sessions)
    use_session_split: bool = True
    train_sessions: List[int] = field(default_factory=lambda: [1, 2])
    test_sessions: List[int] = field(default_factory=lambda: [3])
    session_normalization: bool = True  # Normalize across sessions
    
    # Enhanced quality filtering for Kinect data
    min_depth_pixels: int = 1000  # Increased for 256x256
    max_depth_noise: float = 0.08  # Stricter
    rgb_depth_alignment_check: bool = True
    temporal_consistency_threshold: float = 0.9
    
    # Multi-modal fusion settings
    modality_fusion: str = "early"  # "early", "late", "attention"
    depth_weight: float = 0.3  # When fusing RGB+D
    rgb_weight: float = 0.7
    
    def __post_init__(self):
        # Adjust channels based on modalities
        if self.use_depth and self.use_rgb:
            self.channels = 4
        elif self.use_depth and not self.use_rgb:
            self.channels = 1
        elif self.use_eye_positions:
            self.channels += 2  # Add eye position channels


@dataclass
class CSMAD_Config(BaseDatasetConfig):
    """Enhanced configuration for Custom Silicone Mask Attack Dataset."""
    
    # Dataset-specific paths  
    dataset_name: str = "CSMAD"
    data_file: str = "csmad.pkl"
    
    # CSMAD specific settings - Enhanced
    mask_types: List[str] = field(default_factory=lambda: ["WEAR", "STAND"])
    lighting_conditions: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    pose_variations: List[str] = field(default_factory=lambda: ["frontal", "left", "right"])
    
    # Enhanced high-quality mask specific preprocessing
    enhance_texture_details: bool = True
    texture_enhancement_factor: float = 1.3  # Increased
    mask_edge_detection: bool = True
    skin_texture_analysis: bool = True
    
    # Image preprocessing - UPGRADED
    image_size: Tuple[int, int] = (256, 256)  # UPGRADED
    channels: int = 3
    normalize: bool = False  
    batch_size: int = 8  # Adjusted
    quality_threshold: float = 0.75  # Higher for professional masks
    
    # Enhanced alpha channel handling
    remove_alpha_channel: bool = True
    alpha_channel_analysis: bool = True  # Analyze before removal
    background_consistency_check: bool = True
    
    # Enhanced quality filtering for high-resolution masks
    min_mask_coverage: float = 0.75  # Increased
    blur_detection_threshold: float = 120.0  # Stricter
    texture_quality_threshold: float = 0.8
    
    # Professional mask specific settings - Enhanced
    professional_masks_only: bool = True
    exclude_low_quality: bool = True
    mask_material_classification: bool = True
    silicone_specific_processing: bool = True
    
    # Advanced mask detection features
    mask_artifact_detection: bool = True
    edge_quality_assessment: bool = True
    surface_reflection_analysis: bool = True
    
    # Lighting condition specific processing
    lighting_normalization: bool = True
    shadow_reduction: bool = True
    glare_detection: bool = True


@dataclass
class ReplayAttack_Config(BaseDatasetConfig):
    """Enhanced configuration for Replay-Attack dataset."""
    
    # Dataset-specific paths
    dataset_name: str = "Replay-Attack"
    data_file: str = "replay_attack.pkl"
    
    # Replay attack specific settings - Enhanced
    attack_types: List[str] = field(default_factory=lambda: ["print", "mobile", "highdef"])
    lighting_conditions: List[str] = field(default_factory=lambda: ["controlled", "adverse"])
    device_types: List[str] = field(default_factory=lambda: ["laptop", "mobile", "tablet"])
    
    # Enhanced video-based preprocessing
    frame_sampling_rate: int = 3  # More frequent sampling
    max_frames_per_video: int = 25  # Increased
    temporal_consistency: bool = True  # Enable temporal modeling
    optical_flow_analysis: bool = True  # Detect display artifacts
    
    # Image preprocessing - UPGRADED
    image_size: Tuple[int, int] = (256, 256)  # UPGRADED
    channels: int = 3
    normalize: bool = False  
    batch_size: int = 6  # Adjusted for video processing
    quality_threshold: float = 0.6  # Balanced for various qualities
    
    # Enhanced screen reflection handling
    screen_glare_removal: bool = True
    moire_pattern_detection: bool = True
    pixel_grid_detection: bool = True
    display_bezel_detection: bool = True
    
    # Advanced display artifact detection
    refresh_rate_analysis: bool = True
    flicker_detection: bool = True
    color_gamut_analysis: bool = True
    contrast_enhancement: bool = True
    
    # Protocol following paper methodology - Enhanced
    use_predefined_splits: bool = True
    development_set_usage: str = "validation"  # "validation", "training", "ignore"
    cross_device_evaluation: bool = True
    
    # Enhanced quality control for various attack media
    print_quality_assessment: bool = True
    display_quality_metrics: bool = True
    viewing_angle_correction: bool = True


@dataclass
class OurDataset_Config(BaseDatasetConfig):
    """Enhanced configuration for our custom dataset."""
    
    # Dataset-specific paths
    dataset_name: str = "Our"
    data_file: str = "our_dataset.pkl"
    
    # Custom dataset specific settings - Enhanced
    video_sources: List[str] = field(default_factory=lambda: [
        "smartphone", "webcam", "youtube", "social_media"
    ])
    recording_devices: List[str] = field(default_factory=lambda: [
        "laptop_webcam", "phone_camera", "tablet_camera", "external_webcam"
    ])
    
    # Enhanced real-world variation handling
    diverse_lighting: bool = True
    various_angles: bool = True
    multiple_backgrounds: bool = True
    indoor_outdoor_scenes: bool = True
    different_demographics: bool = True
    
    # Image preprocessing - UPGRADED FOR VIDEO DATASET
    image_size: Tuple[int, int] = (256, 256)  # UPGRADED
    channels: int = 3
    normalize: bool = False  
    batch_size: int = 6  # Adjusted for video processing
    quality_threshold: float = 0.6  # Balanced for diverse sources
    
    # Enhanced dataset balancing
    enforce_class_balance: bool = True
    target_balance_ratio: float = 0.5
    demographic_balancing: bool = True
    device_balancing: bool = True
    
    # Enhanced quality control for internet-sourced videos
    min_video_quality: str = "720p"  # Increased from 480p
    max_compression_artifacts: float = 0.25  # Stricter
    face_detection_confidence: float = 0.85  # Increased
    min_face_size_ratio: float = 0.15  # Minimum face size relative to image
    
    # Advanced processing for video frames
    video_enhancement: bool = True
    denoise_frames: bool = True
    sharpen_frames: bool = True
    stabilization: bool = True
    motion_blur_detection: bool = True
    
    # Enhanced metadata tracking
    source_tracking: bool = True
    device_metadata: bool = True
    recording_conditions: Dict[str, Any] = field(default_factory=lambda: {
        "lighting": True,
        "background": True,
        "pose": True,
        "expression": True
    })
    
    # Custom split optimization
    train_split: float = 0.48
    validation_split: float = 0.32
    test_split: float = 0.20  # Added test set
    
    # Advanced video processing
    keyframe_extraction: bool = True
    scene_change_detection: bool = True
    face_tracking_consistency: bool = True


@dataclass
class CombinedDataset_Config(BaseDatasetConfig):
    """Enhanced configuration for combined datasets training."""
    
    # Combined dataset settings
    dataset_name: str = "Combined_All"
    included_datasets: List[DatasetType] = field(default_factory=lambda: [
        DatasetType.MSSPOOF,
        DatasetType.THREEDMAD, 
        DatasetType.CSMAD,
        DatasetType.REPLAY_ATTACK,
        DatasetType.OUR_DATASET
    ])
    
    # Enhanced dataset weighting strategies
    dataset_weights: Dict[str, float] = field(default_factory=lambda: {})
    weighting_strategy: str = "balanced"  # "balanced", "size_based", "quality_based", "custom"
    balance_across_datasets: bool = True
    quality_weighted_sampling: bool = True
    
    # Advanced cross-dataset evaluation settings
    leave_one_out: bool = False
    target_dataset: Optional[DatasetType] = None
    domain_adaptation: bool = False
    
    # Enhanced augmentation for diverse data
    aggressive_augmentation: bool = True
    cross_domain_augmentation: bool = True
    dataset_specific_augmentation: bool = True
    
    # Advanced handling of dataset differences
    normalize_per_dataset: bool = False
    unified_preprocessing: bool = True
    feature_alignment: bool = True
    
    # Large dataset specific settings - UPGRADED
    image_size: Tuple[int, int] = (256, 256)  # UPGRADED
    normalize: bool = False  
    batch_size: int = 4  # Reduced for memory efficiency
    use_data_generators: bool = True
    memory_efficient_loading: bool = True
    cache_preprocessed: bool = True
    distributed_loading: bool = True
    
    # Advanced sampling strategies
    stratified_sampling: bool = True
    temporal_sampling: bool = True
    quality_based_sampling: bool = True
    
    def __post_init__(self):
        if not self.dataset_weights:
            # Initialize weights based on strategy
            if self.weighting_strategy == "balanced":
                self.dataset_weights = {
                    dataset.value: 1.0 for dataset in self.included_datasets
                }
            elif self.weighting_strategy == "quality_based":
                # Higher weights for higher quality datasets
                quality_weights = {
                    DatasetType.MSSPOOF.value: 1.2,
                    DatasetType.CSMAD.value: 1.1,
                    DatasetType.THREEDMAD.value: 1.0,
                    DatasetType.REPLAY_ATTACK.value: 0.9,
                    DatasetType.OUR_DATASET.value: 0.8
                }
                self.dataset_weights = {
                    ds.value: quality_weights.get(ds.value, 1.0) 
                    for ds in self.included_datasets
                }


@dataclass
class CrossDatasetConfig(BaseDatasetConfig):
    """Enhanced configuration for cross-dataset evaluation experiments."""
    
    # Cross-dataset experimental setup
    train_dataset: DatasetType = DatasetType.MSSPOOF
    test_datasets: List[DatasetType] = field(default_factory=lambda: [
        DatasetType.THREEDMAD,
        DatasetType.CSMAD, 
        DatasetType.REPLAY_ATTACK,
        DatasetType.OUR_DATASET
    ])
    
    # Enhanced domain adaptation settings
    use_domain_adaptation: bool = False
    adaptation_method: str = "fine_tuning"  # "fine_tuning", "feature_alignment", "adversarial"
    adaptation_layers: List[str] = field(default_factory=lambda: ["last", "last_two"])
    
    # Advanced evaluation protocol
    report_per_dataset: bool = True
    aggregate_results: bool = True
    confusion_matrix_analysis: bool = True
    error_analysis: bool = True
    
    # Enhanced statistical analysis
    confidence_intervals: bool = True
    statistical_tests: bool = True
    effect_size_analysis: bool = True
    significance_level: float = 0.05
    
    # UPGRADED settings
    image_size: Tuple[int, int] = (256, 256)  # UPGRADED
    normalize: bool = False  
    batch_size: int = 8  # Balanced
    quality_threshold: float = 0.65
    
    # Cross-dataset specific processing
    domain_invariant_features: bool = True
    style_transfer_augmentation: bool = False
    gradient_reversal: bool = False


# Enhanced utility functions

def get_dataset_config(dataset_type: Union[DatasetType, str], 
                      processing_mode: ProcessingMode = ProcessingMode.ENHANCED) -> BaseDatasetConfig:
    """
    Get enhanced configuration for a specific dataset.
    
    Args:
        dataset_type: Type of dataset to get configuration for
        processing_mode: Processing quality mode
        
    Returns:
        Enhanced configuration object for the specified dataset
        
    Raises:
        ValueError: If dataset_type is not recognized
    """
    if isinstance(dataset_type, str):
        try:
            dataset_type = DatasetType(dataset_type)
        except ValueError:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    config_mapping = {
        DatasetType.MSSPOOF: MSSpoof_Config,
        DatasetType.THREEDMAD: ThreeDMAD_Config,
        DatasetType.CSMAD: CSMAD_Config,
        DatasetType.REPLAY_ATTACK: ReplayAttack_Config,
        DatasetType.OUR_DATASET: OurDataset_Config,
        DatasetType.COMBINED_ALL: CombinedDataset_Config,
    }
    
    if dataset_type not in config_mapping:
        available = ", ".join([dt.value for dt in DatasetType])
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {available}")
    
    config = config_mapping[dataset_type]()
    config.processing_mode = processing_mode
    
    # Adjust settings based on processing mode
    if processing_mode == ProcessingMode.PREMIUM:
        config.quality_threshold = min(config.quality_threshold + 0.1, 0.9)
        config.batch_size = max(config.batch_size // 2, 2)
        config.parallel_workers = min(config.parallel_workers + 2, 8)
    elif processing_mode == ProcessingMode.BASIC:
        config.quality_threshold = max(config.quality_threshold - 0.1, 0.4)
        config.enable_quality_filtering = False
        config.noise_reduction = False
        config.edge_enhancement = False
    
    return config


def get_all_dataset_configs(processing_mode: ProcessingMode = ProcessingMode.ENHANCED) -> Dict[str, BaseDatasetConfig]:
    """
    Get enhanced configurations for all available datasets.
    
    Args:
        processing_mode: Processing quality mode
    
    Returns:
        Dictionary mapping dataset names to their enhanced configurations
    """
    return {
        dataset_type.value: get_dataset_config(dataset_type, processing_mode)
        for dataset_type in DatasetType
    }


def create_cross_dataset_configs(processing_mode: ProcessingMode = ProcessingMode.ENHANCED) -> Dict[str, CrossDatasetConfig]:
    """
    Create enhanced configurations for all cross-dataset evaluation scenarios.
    
    Args:
        processing_mode: Processing quality mode
    
    Returns:
        Dictionary of enhanced cross-dataset evaluation configurations
    """
    configs = {}
    
    train_datasets = [
        DatasetType.MSSPOOF,
        DatasetType.THREEDMAD,
        DatasetType.CSMAD,
        DatasetType.REPLAY_ATTACK,
        DatasetType.OUR_DATASET
    ]
    
    for train_ds in train_datasets:
        test_datasets = [ds for ds in train_datasets if ds != train_ds]
        
        config = CrossDatasetConfig(
            train_dataset=train_ds,
            test_datasets=test_datasets,
            processing_mode=processing_mode
        )
        
        configs[f"train_{train_ds.value}"] = config
    
    return configs


def get_optimal_batch_size(dataset_type: DatasetType, image_size: Tuple[int, int], 
                          available_memory_gb: float = 8.0) -> int:
    """
    Calculate optimal batch size based on dataset type and available memory.
    
    Args:
        dataset_type: Type of dataset
        image_size: Target image size
        available_memory_gb: Available GPU/system memory in GB
        
    Returns:
        Recommended batch size
    """
    # Base memory usage per image in MB (for 256x256x3 float32)
    base_memory_per_image = (image_size[0] * image_size[1] * 3 * 4) / (1024 * 1024)
    
    # Dataset-specific multipliers based on complexity
    complexity_multipliers = {
        DatasetType.MSSPOOF: 1.2,  # Multi-spectral data
        DatasetType.THREEDMAD: 1.3,  # RGB + potential depth
        DatasetType.CSMAD: 1.0,  # Standard RGB
        DatasetType.REPLAY_ATTACK: 1.1,  # Video frames
        DatasetType.OUR_DATASET: 1.1,  # Video frames
        DatasetType.COMBINED_ALL: 1.4,  # Mixed complexity
    }
    
    multiplier = complexity_multipliers.get(dataset_type, 1.0)
    memory_per_image = base_memory_per_image * multiplier
    
    # Reserve 70% of memory for batch processing, 30% for model and other operations
    available_batch_memory = available_memory_gb * 1024 * 0.7  # MB
    
    optimal_batch_size = int(available_batch_memory / memory_per_image)
    
    # Ensure reasonable bounds
    return max(2, min(optimal_batch_size, 32))


def get_dataset_statistics() -> Dict[str, Dict[str, Union[int, float, str]]]:
    """
    Get enhanced expected dataset statistics based on research and analysis.
    
    Returns:
        Dictionary containing comprehensive dataset statistics
    """
    return {
        "MSSpoof": {
            "subjects": 21,
            "real_access_per_client": 70,
            "attacks_per_client": 144,
            "spectra": ["VIS", "NIR"],
            "original_resolution": "1280x1024",
            "enhanced_resolution": "256x256",
            "quality_threshold": 0.7,
            "expected_quality_score": 0.75,
            "processing_complexity": "high"
        },
        "3DMAD": {
            "subjects": 17,
            "total_frames": 76500,
            "frames_per_subject": 4500,
            "sessions": 3,
            "data_types": ["RGB", "Depth", "Eye_positions"],
            "kinect_resolution": "640x480",
            "enhanced_resolution": "256x256",
            "quality_threshold": 0.65,
            "expected_quality_score": 0.68,
            "processing_complexity": "high"
        },
        "CSMAD": {
            "subjects": 14,
            "masks": 6,
            "bonafide_videos": 87,
            "attack_videos": 159,
            "lighting_conditions": 4,
            "mask_types": ["silicone", "professional"],
            "enhanced_resolution": "256x256",
            "quality_threshold": 0.75,
            "expected_quality_score": 0.78,
            "processing_complexity": "medium"
        },
        "Replay-Attack": {
            "clients": 50,
            "total_videos": 1300,
            "train_real": 60,
            "train_attacks": 300,
            "test_real": 80,
            "test_attacks": 400,
            "original_resolution": "320x240",
            "enhanced_resolution": "256x256",
            "attack_types": ["print", "mobile", "highdef"],
            "quality_threshold": 0.6,
            "expected_quality_score": 0.62,
            "processing_complexity": "medium"
        },
        "Our": {
            "total_images": 4656,
            "train_images": 2238,
            "validation_images": 2418,
            "class_distribution": "50/50",
            "videos": 84,
            "sources": ["smartphone", "webcam", "youtube"],
            "enhanced_resolution": "256x256",
            "quality_threshold": 0.6,
            "expected_quality_score": 0.65,
            "processing_complexity": "variable"
        }
    }


# Enhanced predefined experiment configurations
ENHANCED_DATASET_CONFIGS = {
    "single_dataset_evaluation": {
        dataset.value: get_dataset_config(dataset, ProcessingMode.ENHANCED) 
        for dataset in [DatasetType.MSSPOOF, DatasetType.THREEDMAD, 
                       DatasetType.CSMAD, DatasetType.REPLAY_ATTACK, 
                       DatasetType.OUR_DATASET]
    },
    
    "premium_quality_evaluation": {
        dataset.value: get_dataset_config(dataset, ProcessingMode.PREMIUM) 
        for dataset in [DatasetType.MSSPOOF, DatasetType.CSMAD]  # High-quality datasets
    },
    
    "cross_dataset_evaluation": create_cross_dataset_configs(ProcessingMode.ENHANCED),
    
    "combined_training": {
        "all_datasets_enhanced": CombinedDataset_Config(),
        "high_quality_only": CombinedDataset_Config(
            included_datasets=[DatasetType.MSSPOOF, DatasetType.CSMAD],
            weighting_strategy="quality_based"
        )
    },
    
    "ablation_studies": {
        "rgb_only_enhanced": {
            dataset.value: get_dataset_config(dataset, ProcessingMode.ENHANCED)
            for dataset in [DatasetType.MSSPOOF, DatasetType.CSMAD, 
                           DatasetType.REPLAY_ATTACK, DatasetType.OUR_DATASET]
        },
        "multimodal_enhanced": {
            DatasetType.THREEDMAD.value: ThreeDMAD_Config(
                use_depth=True, 
                use_rgb=True, 
                processing_mode=ProcessingMode.ENHANCED
            )
        },
        "quality_impact": {
            f"{dataset.value}_basic": get_dataset_config(dataset, ProcessingMode.BASIC)
            for dataset in DatasetType
        }
    }
}


# Configuration validation utilities

def validate_config(config: BaseDatasetConfig) -> List[str]:
    """
    Validate configuration for potential issues.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation warnings/errors
    """
    issues = []
    
    # Check splits sum to 1.0
    total_split = config.train_split + config.validation_split + config.test_split
    if abs(total_split - 1.0) > 0.01:
        issues.append(f"Data splits sum to {total_split:.3f}, should be 1.0")
    
    # Check batch size feasibility
    if config.batch_size < 1:
        issues.append("Batch size must be at least 1")
    
    # Check quality threshold range
    if not 0.0 <= config.quality_threshold <= 1.0:
        issues.append("Quality threshold must be between 0.0 and 1.0")
    
    # Check image size
    if config.image_size[0] < 64 or config.image_size[1] < 64:
        issues.append("Image size should be at least 64x64")
    
    # Check memory requirements
    memory_per_batch = (config.image_size[0] * config.image_size[1] * 
                       config.channels * config.batch_size * 4) / (1024**3)
    if memory_per_batch > 2.0:  # 2GB per batch is quite large
        issues.append(f"Batch memory requirement: {memory_per_batch:.2f}GB may be too large")
    
    return issues


def create_config_summary(config: BaseDatasetConfig) -> str:
    """
    Create a human-readable summary of configuration.
    
    Args:
        config: Configuration to summarize
        
    Returns:
        Formatted configuration summary
    """
    return f"""
Enhanced Dataset Configuration: {config.dataset_name}
{'='*50}
Resolution: {config.image_size[0]}×{config.image_size[1]} (UPGRADED)
Quality Threshold: {config.quality_threshold}
Processing Mode: {config.processing_mode.value}
Batch Size: {config.batch_size}
Workers: {config.parallel_workers}

Data Splits:
  Training: {config.train_split:.1%}
  Validation: {config.validation_split:.1%}
  Testing: {config.test_split:.1%}

Quality Features:
  Quality Filtering: {config.enable_quality_filtering}
  Noise Reduction: {config.noise_reduction}
  Edge Enhancement: {config.edge_enhancement}
  Histogram Equalization: {config.histogram_equalization}
  
Enhancement Features:
  Advanced Augmentation: {config.augmentation_type.value}
  Class Balancing: {config.balancing_strategy}
  Parallel Processing: {config.parallel_workers} workers
  Memory Efficient: {config.memory_efficient}
"""
