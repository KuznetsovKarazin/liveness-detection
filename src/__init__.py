"""
Source Package for Liveness Detection

This package contains the main source code for the liveness detection project:
- Model architectures
- Dataset loading utilities
- Training and evaluation utilities
- Performance analysis tools
"""

# No imports here to avoid circular dependencies
# Users should import directly from submodules:
# from src.architectures import LivenessNet
# from src.dataset_loader import DatasetLoader
# etc.

__all__ = [
    # Architectures
    "architectures",
    
    # Dataset loading
    "dataset_loader",
    
    # Evaluation
    "evaluation_utils",
    
    # Training
    "training_utils"
]