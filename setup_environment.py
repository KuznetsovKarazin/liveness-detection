#!/usr/bin/env python3
"""
Environment Setup Script 

This script helps set up the complete environment for the liveness detection project.
Run this after creating your virtual environment and installing Python 3.10/3.11.
"""

import sys
import subprocess
import platform
from pathlib import Path
import os

# Fix for Windows Unicode issues
if platform.system() == 'Windows':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("ERROR: Python 3 is required!")
        return False
    
    if version.minor < 8:
        print("ERROR: Python 3.8+ is required!")
        return False
    
    if version.minor > 11:
        print("WARNING: Python 3.12+ may have TensorFlow compatibility issues")
        print("   Recommended: Python 3.10 or 3.11")
    
    print("OK: Python version is compatible")
    return True


def check_virtual_environment():
    """Check if running in a virtual environment."""
    in_venv = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    
    if in_venv:
        print("OK: Running in virtual environment")
        print(f"   Environment path: {sys.prefix}")
        return True
    else:
        print("WARNING: Not running in virtual environment")
        print("   It's recommended to use a virtual environment")
        return False


def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n[RUNNING] {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"[SUCCESS] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def install_packages():
    """Install all required packages."""
    print("\n" + "="*50)
    print("INSTALLING PACKAGES")
    print("="*50)
    
    # Upgrade pip first
    if not run_command(
        f"{sys.executable} -m pip install --upgrade pip setuptools wheel",
        "Upgrading pip and setuptools"
    ):
        return False
    
    # Install TensorFlow first (most critical)
    if not run_command(
        f"{sys.executable} -m pip install tensorflow==2.15.0",
        "Installing TensorFlow 2.15.0"
    ):
        print("Trying alternative TensorFlow installation...")
        if not run_command(
            f"{sys.executable} -m pip install tensorflow==2.15.0 --no-cache-dir",
            "Installing TensorFlow (no cache)"
        ):
            return False
    
    # Install other core packages
    core_packages = [
        "numpy==1.24.3",
        "opencv-python==4.8.1.78", 
        "matplotlib==3.8.2",
        "scikit-learn==1.3.2",
        "pandas==2.1.4",
        "seaborn==0.13.0",
        "tqdm==4.66.1",
        "jupyter==1.0.0",
    ]
    
    for package in core_packages:
        if not run_command(
            f"{sys.executable} -m pip install {package}",
            f"Installing {package.split('==')[0]}"
        ):
            print(f"WARNING: Failed to install {package}, continuing...")
    
    # Install remaining packages from requirements.txt if it exists
    requirements_path = Path("requirements.txt")
    if requirements_path.exists():
        run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing remaining packages from requirements.txt"
        )
    
    return True


def test_installations():
    """Test if critical packages are working."""
    print("\n" + "="*50)
    print("TESTING INSTALLATIONS")
    print("="*50)
    
    tests = [
        ("TensorFlow", "import tensorflow as tf; print('TF version:', tf.__version__)"),
        ("NumPy", "import numpy as np; print('NumPy version:', np.__version__)"),
        ("OpenCV", "import cv2; print('OpenCV version:', cv2.__version__)"),
        ("Matplotlib", "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"),
        ("Scikit-learn", "import sklearn; print('Sklearn version:', sklearn.__version__)"),
        ("Pandas", "import pandas as pd; print('Pandas version:', pd.__version__)"),
    ]
    
    all_passed = True
    
    for name, test_code in tests:
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_code],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            )
            print(f"[OK] {name}: {result.stdout.strip()}")
        except subprocess.CalledProcessError:
            print(f"[ERROR] {name}: Import failed")
            all_passed = False
    
    return all_passed


def test_tensorflow_functionality():
    """Test TensorFlow functionality specifically."""
    print("\n" + "="*50)
    print("TESTING TENSORFLOW FUNCTIONALITY")
    print("="*50)
    
    tf_test = """
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

try:
    # Test GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f'GPUs available: {len(gpus)}')
    for gpu in gpus:
        print(f'  - {gpu}')

    # Test basic model creation
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Test prediction
    test_data = np.random.random((10, 5))
    predictions = model.predict(test_data, verbose=0)
    print(f'Model prediction shape: {predictions.shape}')
    print('TensorFlow functionality test passed!')
except Exception as e:
    print(f'TensorFlow test failed: {e}')
    raise
"""
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", tf_test],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'TF_CPP_MIN_LOG_LEVEL': '2'}
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] TensorFlow functionality test failed:")
        print(e.stderr)
        return False


def create_project_structure():
    """Create the project directory structure."""
    print("\n" + "="*50)
    print("CREATING PROJECT STRUCTURE")
    print("="*50)
    
    directories = [
        "src",
        "config", 
        "notebooks",
        "scripts",
        "data/raw",
        "data/processed",
        "models",
        "results/confusion_matrices",
        "results/performance_metrics",
        "results/training_curves",
        "results/comparison_tables",
        "logs"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created: {directory}/")
    
    return True


def create_quick_test():
    """Create a quick test script."""
    test_script = '''#!/usr/bin/env python3
"""Quick test to verify the environment is working."""

def test_environment():
    print("ENVIRONMENT TEST")
    print("="*30)
    
    try:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        import tensorflow as tf
        print(f"[OK] TensorFlow {tf.__version__}")
        
        import numpy as np
        print(f"[OK] NumPy {np.__version__}")
        
        import cv2
        print(f"[OK] OpenCV {cv2.__version__}")
        
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, activation='softmax', input_shape=(5,))
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Test prediction
        test_data = np.random.random((1, 5))
        pred = model.predict(test_data, verbose=0)
        
        print(f"[OK] Model test passed: {pred.shape}")
        print("\\nEnvironment is ready!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

if __name__ == "__main__":
    test_environment()
'''
    
    try:
        with open("test_environment.py", "w", encoding='utf-8') as f:
            f.write(test_script)
        print("[OK] Created test_environment.py")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create test_environment.py: {e}")
        return False


def main():
    """Main setup function."""
    print("="*60)
    print("LIVENESS DETECTION PROJECT SETUP")
    print("="*60)
    
    # Check system
    if not check_python_version():
        return False
    
    check_virtual_environment()
    
    # Install packages
    if not install_packages():
        print("\n[ERROR] Package installation failed!")
        return False
    
    # Test installations
    if not test_installations():
        print("\n[WARNING] Some packages failed to install properly")
    
    # Test TensorFlow specifically
    if not test_tensorflow_functionality():
        print("\n[WARNING] TensorFlow functionality test failed")
    
    # Create project structure
    create_project_structure()
    
    # Create test script
    create_quick_test()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test environment: python test_environment.py")
    print("2. Copy your project files to this directory")
    print("3. Start with: jupyter notebook")
    
    print(f"\nProject directory: {Path.cwd().absolute()}")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n[ERROR] Setup failed. Please check the errors above.")
        sys.exit(1)
