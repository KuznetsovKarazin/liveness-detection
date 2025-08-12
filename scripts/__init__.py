"""
Enhanced Training and Evaluation Scripts

This package contains optimized automation scripts for:
- Multi-threaded model training with advanced configurations
- Comprehensive cross-dataset evaluation  
- Performance analysis and benchmarking
- Hyperparameter optimization with Optuna
- Production deployment utilities

Optimized for AMD Ryzen 7 7840HS with 64GB RAM.
"""

__version__ = "2.0.0"
__description__ = "Enhanced automation scripts for liveness detection"

# Export script utilities when available
__all__ = [
    "train_all_models",
    "evaluate_all_models", 
    "run_hyperparameter_optimization",
    "benchmark_performance"
]

print(f"📜 Enhanced Scripts Package v{__version__} - Ready for automation")