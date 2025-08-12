"""
Model Configuration Settings

This module contains optimized hyperparameter configurations for all model architectures.
Configurations are organized by model type and training scenario.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class BaseModelConfig:
    """Base configuration with only parameters that are actually used in training"""
    
    # Training parameters - ACTUALLY USED in TrainingConfig
    optimizer: str = "adam"                    # Used in training_utils.py -> _get_optimizer()
    learning_rate: float = 1e-3               # Used in training_utils.py -> _get_optimizer()
    epochs: int = 10                          # Used in training_utils.py -> train()
    batch_size: int = 8                      # Used in train_all_models.py -> DatasetConfig
    l2_regularization: float = 1e-4
    dropout_rate: float = 0.5
    
    # Callbacks - ACTUALLY USED in training_utils.py
    early_stopping_patience: int = 15        # Used in _get_callbacks() -> EarlyStopping
    reduce_lr_patience: int = 7               # Used in _get_callbacks() -> ReduceLROnPlateau
    reduce_lr_factor: float = 0.5             # Used in _get_callbacks() -> ReduceLROnPlateau
    min_learning_rate: float = 1e-8           # Used in _get_callbacks() -> ReduceLROnPlateau
    
    # Validation split - ACTUALLY USED indirectly
    validation_split: float = 0.2            # Used in dataset_loader.py DatasetConfig
    
    # Model saving - ACTUALLY USED in training_utils.py
    monitor_metric: str = "val_loss"          # Used in _get_callbacks() -> callbacks
    monitor_mode: str = "min"                 # Used in _get_callbacks() -> callbacks

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {field: getattr(self, field) 
                for field in self.__dataclass_fields__}

@dataclass
class LivenessNetConfig(BaseModelConfig):
    """Configuration for baseline LivenessNet model"""
    
    # Conservative settings for baseline model
    epochs: int = 20                           # Quick baseline training
    learning_rate: float = 1e-7               # Very conservative for stability
    early_stopping_patience: int = 20         # Shorter patience for simple model
    batch_size: int = 8                      # Can use larger batch for simple model
    l2_regularization: float = 1e-5
    dropout_rate: float = 0.2

@dataclass  
class AttackNetV1Config(BaseModelConfig):
    """Configuration for AttackNet V1 with concatenation residuals"""
    epochs: int = 20                           
    learning_rate: float = 3e-7              
    early_stopping_patience: int = 20         
    batch_size: int = 8                      
    l2_regularization: float = 1e-5
    dropout_rate: float = 0.5  

@dataclass
class AttackNetV2_1Config(BaseModelConfig):
    """Configuration for AttackNet V2.1 with concatenation residuals"""
    
    epochs: int = 20                           
    learning_rate: float = 4e-7              
    early_stopping_patience: int = 20         
    batch_size: int = 8                      
    l2_regularization: float = 1e-5
    dropout_rate: float = 0.4                      

@dataclass
class AttackNetV2_2Config(BaseModelConfig):
    """Configuration for AttackNet V2.2 with addition residuals"""
    epochs: int = 20                           
    learning_rate: float = 4e-8              
    early_stopping_patience: int = 20         
    batch_size: int = 8                      
    l2_regularization: float = 1e-5
    dropout_rate: float = 0.2  

# Dataset-specific adjustments based on dataset characteristics
@dataclass
class InternetSourcedConfig(BaseModelConfig):
    """Configuration for datasets sourced from internet videos (our_dataset)"""
    reduce_lr_patience: int = 10              # More patience due to quality variation
    learning_rate: float = 1e-5               # More conservative due to noise
    batch_size: int = 8                       # Smaller batch due to variable quality

@dataclass
class SmallDatasetConfig(BaseModelConfig):
    """Configuration for small datasets prone to overfitting (csmad)"""
    learning_rate: float = 1e-5               # Very conservative for small data
    early_stopping_patience: int = 25         # Very patient to avoid early stopping
    epochs: int = 100                         # Allow more epochs with early stopping
    batch_size: int = 8                       # Small batch to avoid overfitting

@dataclass
class HighQualityDatasetConfig(BaseModelConfig):
    """Configuration for high-quality competition datasets (msspoof)"""
    learning_rate: float = 1e-4               # Can be slightly more aggressive
    early_stopping_patience: int = 12         # Standard patience for quality data
    batch_size: int = 32                      # Can use larger batch for quality data

def get_model_config(
    architecture_name: str, 
    dataset_name: Optional[str] = None,
    scenario: str = "default"
) -> BaseModelConfig:
    """
    Get configuration for specified architecture and dataset.
    
    Args:
        architecture_name: Model architecture name
        dataset_name: Optional dataset name for dataset-specific config
        scenario: Training scenario
    
    Returns:
        Configured model instance with ONLY used parameters
    """
    # Architecture mappings
    architectures = {
        "LivenessNet": LivenessNetConfig,
        "AttackNetV1": AttackNetV1Config,
        "AttackNetV2_1": AttackNetV2_1Config,
        "AttackNetV2_2": AttackNetV2_2Config
    }
    
    if architecture_name not in architectures:
        raise ValueError(f"Unknown architecture: {architecture_name}")
    
    # Create base config
    base_config = architectures[architecture_name]()
 
    """ 
    # Apply dataset-specific adjustments based on dataset characteristics
    if dataset_name:
        # Our dataset from internet videos - variable quality, needs conservative approach
        if dataset_name in ["our_dataset"]:
            internet_config = InternetSourcedConfig()
            base_config.reduce_lr_patience = internet_config.reduce_lr_patience
            base_config.learning_rate = internet_config.learning_rate
            base_config.batch_size = internet_config.batch_size
        
        # Small datasets need extra patience to avoid overfitting
        elif dataset_name in ["csmad"]:
            small_config = SmallDatasetConfig()
            base_config.learning_rate = small_config.learning_rate
            base_config.early_stopping_patience = small_config.early_stopping_patience
            base_config.epochs = small_config.epochs
            base_config.batch_size = small_config.batch_size
        
        # High-quality competition datasets can use more aggressive settings
        elif dataset_name in ["msspoof"]:
            hq_config = HighQualityDatasetConfig()
            base_config.learning_rate = hq_config.learning_rate
            base_config.early_stopping_patience = hq_config.early_stopping_patience
            base_config.batch_size = hq_config.batch_size
    """
    return base_config

def get_all_model_configs(scenario: str = "default") -> Dict[str, BaseModelConfig]:
    """Get configurations for all architectures"""
    return {
        "LivenessNet": get_model_config("LivenessNet", scenario=scenario),
        "AttackNetV1": get_model_config("AttackNetV1", scenario=scenario),
        "AttackNetV2_1": get_model_config("AttackNetV2_1", scenario=scenario),
        "AttackNetV2_2": get_model_config("AttackNetV2_2", scenario=scenario)
    }

def create_custom_config(
    base_architecture: str,
    **custom_params
) -> BaseModelConfig:
    """Create custom configuration from base with parameter overrides"""
    config = get_model_config(base_architecture)
    valid_fields = config.__dataclass_fields__
    
    for param, value in custom_params.items():
        if param in valid_fields:
            setattr(config, param, value)
        else:
            raise ValueError(f"Invalid parameter: {param}")
    
    return config

# Predefined experiment configurations - ONLY used parameters
OPTIMIZED_CONFIGS = {
    "conservative_training": {
        "learning_rate": 1e-5,
        "early_stopping_patience": 20,
        "epochs": 100
    },
    "fast_training": {
        "learning_rate": 5e-4,
        "epochs": 30,
        "early_stopping_patience": 5
    },
    "patient_training": {
        "early_stopping_patience": 25,
        "reduce_lr_patience": 12,
        "epochs": 150
    }
}
