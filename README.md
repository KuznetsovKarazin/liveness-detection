# Face Liveness Detection Research 🔬

A comprehensive deep learning framework for **face liveness detection** and **presentation attack detection (PAD)**, featuring multiple CNN architectures, cross-dataset evaluation, and advanced preprocessing with 256×256 resolution enhancement.

## 🏆 Key Features

- **4 Neural Network Architectures**: LivenessNet (baseline), AttackNetV1, AttackNetV2.1, AttackNetV2.2
- **5 Benchmark Datasets**: MSSpoof, 3DMAD, CSMAD, Replay-Attack, Custom Dataset
- **Enhanced Resolution**: Upgraded from 128×128 to 256×256 pixels (+300% resolution)
- **Advanced Preprocessing**: CLAHE, bilateral filtering, gamma correction, USM sharpening
- **Cross-Dataset Evaluation**: Comprehensive generalization testing across all datasets
- **Biometric Metrics**: APCER, BPCER, ACER, EER, FAR, FRR calculations
- **Multi-threaded Training**: Parallel training with GPU optimization support
- **Hyperparameter Optimization**: Optuna integration for automated tuning

## 📁 Project Structure

```
LivenessDetection/
├── src/
│   ├── architectures.py          # Neural network models
│   ├── dataset_loader.py         # Enhanced data loading with augmentation
│   ├── training_utils.py         # Advanced training utilities
│   └── evaluation_utils.py       # Comprehensive evaluation metrics
├── config/
│   ├── dataset_configs.py        # Dataset-specific configurations
│   └── model_configs.py          # Model hyperparameters
├── scripts/
│   ├── create_datasets.py        # Dataset creation with quality filtering
│   ├── create_combined_dataset.py # Multi-dataset combination
│   ├── train_all_models.py       # Multi-model training orchestrator
│   ├── train_combined_models.py  # Combined dataset training
│   └── evaluate_all_models.py    # Cross-dataset evaluation
├── data/
│   ├── raw/                      # Original dataset files
│   ├── processed/                 # Preprocessed .pkl files
│   └── quality_reports/          # Quality assessment reports
├── results/
│   ├── training/                 # Trained model checkpoints
│   ├── evaluation/               # Evaluation metrics & visualizations
│   └── logs/                     # Training logs
├── setup_environment.py          # Environment setup script
├── test_environment.py           # Environment verification
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11 (recommended)
- CUDA-capable GPU (optional but recommended)
- 16GB+ RAM
- 50GB+ free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/KuznetsovKarazin/liveness-detection.git
cd liveness-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
python setup_environment.py
# or manually:
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python test_environment.py
```

### Dataset Preparation

1. **Download datasets** and place in `data/raw/`:
  - MSSpoof: https://www.idiap.ch/en/dataset/msspoof 
  - 3DMAD: https://www.idiap.ch/en/dataset/3dmad  
  - CSMAD: https://www.idiap.ch/en/dataset/csmad
  - Replay-Attack: https://www.idiap.ch/en/dataset/replayattack

2. **Create preprocessed datasets**
```bash
python scripts/create_datasets.py
```

3. **Create combined dataset** (optional)
```bash
python scripts/create_combined_dataset.py
```
### ⚠️ Important Notes

- **Dataset Size**: The complete preprocessed datasets require ~15GB of storage
- **Training Time**: Full training on all datasets takes 8-12 hours on GPU
- **Memory Requirements**: Minimum 8GB RAM, 16GB recommended
- **Our Dataset**: Custom dataset from YouTube videos (not redistributable due to privacy)

### Training

**Train all models on all datasets:**
```bash
python scripts/train_all_models.py --threads 2
```

**Train specific model on specific dataset:**
```bash
python scripts/train_all_models.py --models AttackNetV2_1 --datasets msspoof
```

**Train with hyperparameter optimization:**
```bash
python scripts/train_all_models.py --optimize --optuna-trials 20
```

**Train on combined dataset:**
```bash
python scripts/train_combined_models.py
```

### Evaluation

**Evaluate all trained models:**
```bash
python scripts/evaluate_all_models.py
```

**Cross-dataset evaluation:**
```bash
python scripts/evaluate_all_models.py --cross-dataset
```

**Statistical analysis:**
```bash
python scripts/evaluate_all_models.py --statistical-analysis
```

## 📊 Model Architectures

### LivenessNet (Baseline)
- Simple CNN with 2 convolutional blocks
- BatchNormalization and Dropout regularization
- 8,406,098 parameters

### AttackNetV1
- CNN with residual connections using concatenation
- LeakyReLU activations
- Tanh in dense layers
- 33,588,738 parameters

### AttackNetV2.1
- Enhanced version with optimized activation functions
- Concatenation-based skip connections
- Advanced regularization
- 33,588,738 parameters

### AttackNetV2.2
- Addition-based skip connections (vs concatenation)
- Improved gradient flow
- Better performance on cross-dataset evaluation
- 16,806,722 parameters

## 📈 Performance Metrics

### Standard Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix
- Matthews Correlation Coefficient

### Biometric Metrics
- **APCER**: Attack Presentation Classification Error Rate
- **BPCER**: Bonafide Presentation Classification Error Rate
- **ACER**: Average Classification Error Rate
- **EER**: Equal Error Rate
- **FAR/FRR**: False Accept/Reject Rates
- **BPCER@APCER**: BPCER at specific APCER thresholds

## 🔬 Key Technologies

- **Deep Learning**: TensorFlow 2.15.0, Keras
- **Data Processing**: NumPy, OpenCV, Albumentations
- **Visualization**: Matplotlib, Seaborn, Plotly
- **ML Tools**: Scikit-learn, Optuna
- **Parallel Computing**: ThreadPoolExecutor, GPU optimization
- **Quality Assurance**: Comprehensive logging, unit tests

## 📊 Dataset Statistics

| Dataset | Total Samples | Resolution | Data Type | Quality Threshold |
|---------|--------------|------------|-----------|---------------|
| MSSpoof | 2,620 | 256×256 | Image | 0.65 |
| 3DMAD | 3,440 | 256×256 | Image | 0.65 |
| CSMAD | 1,568 | 256×256 | Image | 0.65 |
| Replay-Attack | 5,000 | 256×256 | Image | 0.65 |
| Our Dataset | 2,040 | 256×256 | Image | 0.65 |
| Combined | 13,100 | 256×256 | Image | 0.65 |

## 🎯 Results Summary

### Within-Dataset Performance (Best Models)
- **MSSpoof**: AttackNetV2.2 (99.8% accuracy, 0.002 ACER)
- **3DMAD**: LivenessNet (99.7% accuracy, 0.003 ACER)
- **CSMAD**: LivenessNet (99.7% accuracy, 0.003 ACER)
- **Replay-Attack**: AttackNetV2.2 (99.9% accuracy, 0.001 ACER)
- **Our Dataset**: LivenessNet (99.8% accuracy, 0.002 ACER)

### Combined Dataset Performance
**AttackNetV2.2** achieved best results:
- Test Accuracy: 99.8%
- ACER: 0.002
- EER: 0.000
- Zero false positives across 2,620 test samples

### Cross-Dataset Generalization
- Single-dataset training: ~52% average cross-dataset accuracy
- Combined-dataset training (AttackNetV2.2): 99.9% average accuracy
- Improvement: +91% relative performance gain

## 🛠️ Configuration

### Dataset Configuration
Edit `config/dataset_configs.py`:
```python
# Adjust quality threshold
quality_threshold = 0.65  # Range: 0.0-1.0

# Modify augmentation
augmentation_level = "medium"  # Options: light, medium, heavy

# Change batch size
batch_size = 32  # Adjust based on GPU memory
```

### Model Configuration
Edit `config/model_configs.py`:
```python
# Learning rate
learning_rate = 1e-6

# Training epochs
epochs = 20

# Early stopping
early_stopping_patience = 10
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{kuznetsov2025liveness,
  title={Deep Learning Models for Robust Facial Liveness Detection},
  author={Kuznetsov, Oleksandr and Frontoni, Emanuele and Romeo, Luca and 
          Rosati, Riccardo and Maranesi, Andrea and Muscatello, Alessandro},
  year={2025},
  journal={arXiv preprint},
  url={https://github.com/KuznetsovKarazin/liveness-detection}
}
```
## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## 🙏 Acknowledgments

- MSSpoof dataset providers
- 3DMAD dataset creators
- CSMAD dataset contributors
- Replay-Attack dataset team
- TensorFlow and Keras communities
- Open-source contributors

## 📧 Contact

- **Author**: Oleksandr Kuznetsov
- **Email**: oleksandr.o.kuznetsov@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/oleksandr-kuznetsov/

## 🐛 Known Issues

- TensorFlow 2.15.0 compatibility with Python 3.12+
- GPU memory limitations with batch sizes > 64
- Cross-dataset evaluation requires significant memory

