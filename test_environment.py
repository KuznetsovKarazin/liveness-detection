#!/usr/bin/env python3
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
        print("\nEnvironment is ready!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

if __name__ == "__main__":
    test_environment()
