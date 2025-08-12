\# Face Liveness Detection Research 🔬



A comprehensive deep learning framework for \*\*face liveness detection\*\* and \*\*presentation attack detection (PAD)\*\*, featuring multiple CNN architectures, cross-dataset evaluation, and advanced preprocessing with 256×256 resolution enhancement.



\## 🏆 Key Features



\- \*\*4 Neural Network Architectures\*\*: LivenessNet (baseline), AttackNetV1, AttackNetV2.1, AttackNetV2.2

\- \*\*5 Benchmark Datasets\*\*: MSSpoof, 3DMAD, CSMAD, Replay-Attack, Custom Dataset

\- \*\*Enhanced Resolution\*\*: Upgraded from 128×128 to 256×256 pixels (+300% resolution)

\- \*\*Advanced Preprocessing\*\*: CLAHE, bilateral filtering, gamma correction, USM sharpening

\- \*\*Cross-Dataset Evaluation\*\*: Comprehensive generalization testing across all datasets

\- \*\*Biometric Metrics\*\*: APCER, BPCER, ACER, EER, FAR, FRR calculations

\- \*\*Multi-threaded Training\*\*: Parallel training with GPU optimization support

\- \*\*Hyperparameter Optimization\*\*: Optuna integration for automated tuning



\## 📁 Project Structure



```

LivenessDetection/

├── src/

│   ├── architectures.py          # Neural network models

│   ├── dataset\_loader.py         # Enhanced data loading with augmentation

│   ├── training\_utils.py         # Advanced training utilities

│   └── evaluation\_utils.py       # Comprehensive evaluation metrics

├── config/

│   ├── dataset\_configs.py        # Dataset-specific configurations

│   └── model\_configs.py          # Model hyperparameters

├── scripts/

│   ├── create\_datasets.py        # Dataset creation with quality filtering

│   ├── create\_combined\_dataset.py # Multi-dataset combination

│   ├── train\_all\_models.py       # Multi-model training orchestrator

│   ├── train\_combined\_models.py  # Combined dataset training

│   └── evaluate\_all\_models.py    # Cross-dataset evaluation

├── notebooks/

│   └── dataset\_exploration.ipynb # Interactive data analysis

├── data/

│   ├── raw/                      # Original dataset files

│   ├── processed/                 # Preprocessed .pkl files

│   └── quality\_reports/          # Quality assessment reports

├── results/

│   ├── training/                 # Trained model checkpoints

│   ├── evaluation/               # Evaluation metrics \& visualizations

│   └── logs/                     # Training logs

├── setup\_environment.py          # Environment setup script

├── test\_environment.py           # Environment verification

├── requirements.txt              # Python dependencies

└── README.md                     # This file

```



\## 🚀 Quick Start



\### Prerequisites



\- Python 3.10 or 3.11 (recommended)

\- CUDA-capable GPU (optional but recommended)

\- 16GB+ RAM

\- 50GB+ free disk space



\### Installation



1\. \*\*Clone the repository\*\*

```bash

git clone https://github.com/yourusername/liveness-detection.git

cd liveness-detection

```



2\. \*\*Create virtual environment\*\*

```bash

python -m venv venv

source venv/bin/activate  # On Windows: venv\\Scripts\\activate

```



3\. \*\*Install dependencies\*\*

```bash

python setup\_environment.py

\# or manually:

pip install -r requirements.txt

```



4\. \*\*Verify installation\*\*

```bash

python test\_environment.py

```



\### Dataset Preparation



1\. \*\*Download datasets\*\* and place in `data/raw/`:

&nbsp;  - MSSpoof: \[https://www.idiap.ch:/en/dataset/msspoof/index\_html]

&nbsp;  - 3DMAD: \[https://www.idiap.ch:/en/dataset/3dmad/index\_html]

&nbsp;  - CSMAD: \[https://www.idiap.ch:/en/dataset/csmad/index\_html]

&nbsp;  - Replay-Attack: \[https://www.idiap.ch:/en/dataset/replayattack/index\_html]



2\. \*\*Create preprocessed datasets\*\*

```bash

python scripts/create\_datasets.py

```



3\. \*\*Create combined dataset\*\* (optional)

```bash

python scripts/create\_combined\_dataset.py

```



\### Training



\*\*Train all models on all datasets:\*\*

```bash

python scripts/train\_all\_models.py --threads 2

```



\*\*Train specific model on specific dataset:\*\*

```bash

python scripts/train\_all\_models.py --models AttackNetV2\_1 --datasets msspoof

```



\*\*Train with hyperparameter optimization:\*\*

```bash

python scripts/train\_all\_models.py --optimize --optuna-trials 20

```



\*\*Train on combined dataset:\*\*

```bash

python scripts/train\_combined\_models.py

```



\### Evaluation



\*\*Evaluate all trained models:\*\*

```bash

python scripts/evaluate\_all\_models.py

```



\*\*Cross-dataset evaluation:\*\*

```bash

python scripts/evaluate\_all\_models.py --cross-dataset

```



\*\*Statistical analysis:\*\*

```bash

python scripts/evaluate\_all\_models.py --statistical-analysis

```



\## 📊 Model Architectures



\### LivenessNet (Baseline)

\- Simple CNN with 2 convolutional blocks

\- BatchNormalization and Dropout regularization

\- 8,406,098 parameters



\### AttackNetV1

\- CNN with residual connections using concatenation

\- LeakyReLU activations

\- Tanh in dense layers

\- 33,588,738 parameters



\### AttackNetV2.1

\- Enhanced version with optimized activation functions

\- Concatenation-based skip connections

\- Advanced regularization

\- 33,588,738 parameters



\### AttackNetV2.2

\- Addition-based skip connections (vs concatenation)

\- Improved gradient flow

\- Better performance on cross-dataset evaluation

\- 16,806,722 parameters



\## 📈 Performance Metrics



\### Standard Metrics

\- Accuracy, Precision, Recall, F1-Score

\- ROC-AUC, PR-AUC

\- Confusion Matrix

\- Matthews Correlation Coefficient



\### Biometric Metrics

\- \*\*APCER\*\*: Attack Presentation Classification Error Rate

\- \*\*BPCER\*\*: Bonafide Presentation Classification Error Rate

\- \*\*ACER\*\*: Average Classification Error Rate

\- \*\*EER\*\*: Equal Error Rate

\- \*\*FAR/FRR\*\*: False Accept/Reject Rates

\- \*\*BPCER@APCER\*\*: BPCER at specific APCER thresholds



\## 🔬 Key Technologies



\- \*\*Deep Learning\*\*: TensorFlow 2.15.0, Keras

\- \*\*Data Processing\*\*: NumPy, OpenCV, Albumentations

\- \*\*Visualization\*\*: Matplotlib, Seaborn, Plotly

\- \*\*ML Tools\*\*: Scikit-learn, Optuna

\- \*\*Parallel Computing\*\*: ThreadPoolExecutor, GPU optimization

\- \*\*Quality Assurance\*\*: Comprehensive logging, unit tests



\## 📊 Dataset Statistics



| Dataset | Total Samples | Resolution | Data Type | Quality Threshold |

|---------|--------------|------------|-----------|---------------|

| MSSpoof | 2,620 | 256×256 | Image | 0.65 |

| 3DMAD | 3,440 | 256×256 | Image | 0.65 |

| CSMAD | 1,568 | 256×256 | Image | 0.65 |

| Replay-Attack | 5,000 | 256×256 | Image | 0.65 |

| Our Dataset | 2,040 | 256×256 | Image | 0.65 |

| Combined | 13,100 | 256×256 | Image | 0.65 |



\## 🎯 Results Summary



\### Best Models per Dataset

\- \*\*MSSpoof\*\*: AttackNetV2.1 (99.8% accuracy, 0.001 HTER)

\- \*\*3DMAD\*\*: LivenessNet (99.7% accuracy, 0.002 HTER) 

\- \*\*CSMAD\*\*: AttackNetV1 (99.8% accuracy, 0.001 HTER)

\- \*\*Replay-Attack\*\*: AttackNetV2.2 (99.9% accuracy, 0.003 HTER)

\- \*\*Combined\*\*: AttackNetV2.2 (99.8% accuracy, 0.002 HTER)



\### Cross-Dataset Performance

Best generalization: \*\*AttackNetV2.2\*\* trained on combined dataset

\- Average accuracy: 99.9%

\- Average ACER: 0.001



\## 🛠️ Configuration



\### Dataset Configuration

Edit `config/dataset\_configs.py`:

```python

\# Adjust quality threshold

quality\_threshold = 0.65  # Range: 0.0-1.0



\# Modify augmentation

augmentation\_level = "medium"  # Options: light, medium, heavy



\# Change batch size

batch\_size = 32  # Adjust based on GPU memory

```



\### Model Configuration

Edit `config/model\_configs.py`:

```python

\# Learning rate

learning\_rate = 1e-6



\# Training epochs

epochs = 20



\# Early stopping

early\_stopping\_patience = 10

```



\## 📝 Citation



If you use this code in your research, please cite:



```bibtex

@article{liveness\_detection\_2024,

&nbsp; title={Deep Learning Models for Robust Facial Liveness Detection},

&nbsp; author={Oleksandr Kuznetsov},

&nbsp; year={2025},

&nbsp; journal={GitHub Repository},

&nbsp; url={https://github.com/KuznetsovKarazin/liveness-detection}

}

```



\## 🤝 Contributing



Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.



1\. Fork the repository

2\. Create your feature branch (`git checkout -b feature/AmazingFeature`)

3\. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

4\. Push to the branch (`git push origin feature/AmazingFeature`)

5\. Open a Pull Request



\## 📄 License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## 🙏 Acknowledgments



\- MSSpoof dataset providers

\- 3DMAD dataset creators

\- CSMAD dataset contributors

\- Replay-Attack dataset team

\- TensorFlow and Keras communities

\- Open-source contributors



\## 📧 Contact



\- \*\*Author\*\*: \[Oleksandr Kuznetsov]

\- \*\*Email\*\*: \[oleksandr.o.kuznetsov@gmail.com]

\- \*\*LinkedIn\*\*: \[https://www.linkedin.com/in/oleksandr-kuznetsov/]



\## 🐛 Known Issues



\- TensorFlow 2.15.0 compatibility with Python 3.12+

\- GPU memory limitations with batch sizes > 64

\- Cross-dataset evaluation requires significant memory





