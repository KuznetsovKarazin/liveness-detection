"""
Configuration Package

This package contains all configuration files for the liveness detection project,
including model hyperparameters, dataset settings, and training configurations.
"""

# No imports here to avoid circular dependencies
# Users should import directly from submodules:
# from config.model_configs import get_model_config
# from config.dataset_configs import get_dataset_config
# etc.

__all__ = [
    "model_configs",
    "dataset_configs"
]