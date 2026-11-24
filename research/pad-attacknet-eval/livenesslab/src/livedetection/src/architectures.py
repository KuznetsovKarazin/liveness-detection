"""
Neural Network Architectures for Liveness Detection

This module contains all the model architectures used in the research:
- LivenessNet: Simple baseline CNN
- AttackNetV1: CNN with residual connections (concatenation)
- AttackNetV2_1: Enhanced with LeakyReLU and Tanh activations
- AttackNetV2_2: Uses addition instead of concatenation in skip connections
"""


from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import tensorflow as tf
from tensorflow.keras import layers, Model

import numpy as np

from tensorflow.keras import regularizers

class BaseArchitecture(ABC):
    """Abstract base class for all model architectures."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3), 
                 dropout_rate: float = 0.5, l2_reg: float = 0.01) -> None:
        """
        Initialize the base architecture.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            dropout_rate: Dropout rate to use in the model
        """
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self._model = None
        
    @abstractmethod
    def build_model(self) -> Model:
        """Build and return the Keras model."""
        pass
    
    def get_model(self) -> Model:
        """Get the built model, building it if necessary."""
        if self._model is None:
            self._model = self.build_model()
        return self._model
    
    def get_parameter_count(self) -> int:
        """Get the total number of trainable parameters."""
        model = self.get_model()
        return model.count_params()
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        model = self.get_model()
        return model.summary()


class LivenessNet(BaseArchitecture):
    """LivenessNet: Simple baseline CNN architecture."""
    
    def build_model(self) -> Model:
        """Build the LivenessNet model with configurable dropout."""
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name="input_layer")
        
        # First convolutional block
        x = layers.Conv2D(16, (3, 3), padding="same", name="conv1_1")(inputs)
        x = layers.Activation("relu", name="relu1_1")(x)
        x = layers.BatchNormalization(name="bn1_1")(x)
        x = layers.Conv2D(16, (3, 3), padding="same", name="conv1_2")(x)
        x = layers.Activation("relu", name="relu1_2")(x)
        x = layers.BatchNormalization(name="bn1_2")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool1")(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name="dropout1")(x)  # Use half dropout in conv layers
        
        # Second convolutional block
        x = layers.Conv2D(32, (3, 3), padding="same", name="conv2_1")(x)
        x = layers.Activation("relu", name="relu2_1")(x)
        x = layers.BatchNormalization(name="bn2_1")(x)
        x = layers.Conv2D(32, (3, 3), padding="same", name="conv2_2")(x)
        x = layers.Activation("relu", name="relu2_2")(x)
        x = layers.BatchNormalization(name="bn2_2")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2")(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name="dropout2")(x)
        
        # Dense layers
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.BatchNormalization(name="bn_dense")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_dense")(x)  # Full dropout in dense layer
        
        # Output layer
        outputs = layers.Dense(2, activation="softmax", name="output")(x)
        
        # Create model
        model = Model(inputs, outputs, name="LivenessNet")
        
        return model


class AttackNetV1(BaseArchitecture):
    """AttackNet V1: CNN with residual connections using concatenation."""
    
    def build_model(self) -> Model:
        """Build the AttackNetV1 model with configurable dropout."""
        reg = regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name="input_layer")
        
        # First convolutional phase with residual connection
        y1 = layers.Conv2D(16, (3, 3), padding="same", name="conv1_1",
                     kernel_regularizer=reg)(inputs)
        x = layers.LeakyReLU(alpha=0.2, name="leaky_relu1_1")(y1)
        x = layers.BatchNormalization(name="bn1_1")(x)
        x = layers.Conv2D(16, (3, 3), padding="same", name="conv1_2",
                     kernel_regularizer=reg)(x)
        x = layers.LeakyReLU(alpha=0.2, name="leaky_relu1_2")(x)
        x = layers.Conv2D(16, (3, 3), padding="same", name="conv1_3",
                     kernel_regularizer=reg)(x)
        z1 = layers.LeakyReLU(alpha=0.2, name="leaky_relu1_3")(x)
        
        # Concatenate residual connection
        x = layers.Concatenate(name="concat1")([y1, z1])
        x = layers.BatchNormalization(name="bn1_final")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool1")(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name="dropout1")(x)
        
        # Second convolutional phase with residual connection
        y2 = layers.Conv2D(32, (3, 3), padding="same", name="conv2_1",
                     kernel_regularizer=reg)(x)
        x = layers.LeakyReLU(alpha=0.2, name="leaky_relu2_1")(y2)
        x = layers.BatchNormalization(name="bn2_1")(x)
        x = layers.Conv2D(32, (3, 3), padding="same", name="conv2_2",
                     kernel_regularizer=reg)(x)
        x = layers.LeakyReLU(alpha=0.2, name="leaky_relu2_2")(x)
        x = layers.Conv2D(32, (3, 3), padding="same", name="conv2_3",
                     kernel_regularizer=reg)(x)
        z2 = layers.LeakyReLU(alpha=0.2, name="leaky_relu2_3")(x)
        
        # Concatenate residual connection
        x = layers.Concatenate(name="concat2")([y2, z2])
        x = layers.BatchNormalization(name="bn2_final")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2")(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name="dropout2")(x)
        
        # Dense phase
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(128, activation="tanh", name="dense1",
                     kernel_regularizer=reg)(x)
        x = layers.BatchNormalization(name="bn_dense")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_dense")(x)
        
        # Output layer
        outputs = layers.Dense(2, activation="softmax", name="output",
                     kernel_regularizer=reg)(x)
        
        # Create model
        model = Model(inputs, outputs, name="AttackNetV1")
        
        return model


class AttackNetV2_1(BaseArchitecture):
    """AttackNet V2.1: Enhanced with LeakyReLU and Tanh activations."""
    
    def build_model(self) -> Model:
        """Build the AttackNetV2_1 model with configurable dropout."""
        reg = regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name="input_layer")
        
        # First convolutional phase
        y1 = layers.Conv2D(16, (3, 3), padding="same", name="conv1_1",
                     kernel_regularizer=reg)(inputs)
        x = layers.LeakyReLU(alpha=0.2, name="leaky_relu1_1")(y1)
        x = layers.BatchNormalization(name="bn1_1")(x)
        x = layers.Conv2D(16, (3, 3), padding="same", name="conv1_2",
                     kernel_regularizer=reg)(x)
        x = layers.LeakyReLU(alpha=0.2, name="leaky_relu1_2")(x)
        x = layers.Conv2D(16, (3, 3), padding="same", name="conv1_3",
                     kernel_regularizer=reg)(x)
        z1 = layers.LeakyReLU(alpha=0.2, name="leaky_relu1_3")(x)
        
        # Concatenate residual connection
        x = layers.Concatenate(name="concat1")([y1, z1])
        x = layers.BatchNormalization(name="bn1_final")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool1")(x)
        # x = layers.Dropout(self.dropout_rate * 0.5, name="dropout1")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout1")(x)
        
        # Second convolutional phase
        y2 = layers.Conv2D(32, (3, 3), padding="same", name="conv2_1",
                     kernel_regularizer=reg)(x)
        x = layers.LeakyReLU(alpha=0.2, name="leaky_relu2_1")(y2)
        x = layers.BatchNormalization(name="bn2_1")(x)
        x = layers.Conv2D(32, (3, 3), padding="same", name="conv2_2",
                     kernel_regularizer=reg)(x)
        x = layers.LeakyReLU(alpha=0.2, name="leaky_relu2_2")(x)
        x = layers.Conv2D(32, (3, 3), padding="same", name="conv2_3",
                     kernel_regularizer=reg)(x)
        z2 = layers.LeakyReLU(alpha=0.2, name="leaky_relu2_3")(x)
        
        # Concatenate residual connection
        x = layers.Concatenate(name="concat2")([y2, z2])
        x = layers.BatchNormalization(name="bn2_final")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2")(x)
        # x = layers.Dropout(self.dropout_rate * 0.5, name="dropout2")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout2")(x)
        
        # Dense phase with Tanh activation
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(128, name="dense1_linear")(x)
        x = layers.Activation("tanh", name="tanh_activation")(x)
        x = layers.BatchNormalization(name="bn_dense")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_dense")(x)
        
        # Output layer
        outputs = layers.Dense(2, activation="softmax", name="output",
                     kernel_regularizer=reg)(x)
        
        # Create model
        model = Model(inputs, outputs, name="AttackNetV2_1")
        
        return model


class AttackNetV2_2(BaseArchitecture):
    """AttackNet V2.2: Uses addition instead of concatenation in skip connections."""
    
    def build_model(self) -> Model:
        """Build the AttackNetV2_2 model with configurable dropout."""
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name="input_layer")
        
        # First convolutional phase with addition-based residual
        y1 = layers.Conv2D(16, (3, 3), padding="same", name="conv1_1")(inputs)
        x = layers.LeakyReLU(alpha=0.2, name="leaky_relu1_1")(y1)
        x = layers.BatchNormalization(name="bn1_1")(x)
        x = layers.Conv2D(16, (3, 3), padding="same", name="conv1_2")(x)
        x = layers.LeakyReLU(alpha=0.2, name="leaky_relu1_2")(x)
        x = layers.Conv2D(16, (3, 3), padding="same", name="conv1_3")(x)
        z1 = layers.LeakyReLU(alpha=0.2, name="leaky_relu1_3")(x)
        
        # Add residual connection (element-wise addition)
        x = layers.Add(name="add1")([y1, z1])
        x = layers.BatchNormalization(name="bn1_final")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool1")(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name="dropout1")(x)
        
        # Second convolutional phase with addition-based residual
        y2 = layers.Conv2D(32, (3, 3), padding="same", name="conv2_1")(x)
        x = layers.LeakyReLU(alpha=0.2, name="leaky_relu2_1")(y2)
        x = layers.BatchNormalization(name="bn2_1")(x)
        x = layers.Conv2D(32, (3, 3), padding="same", name="conv2_2")(x)
        x = layers.LeakyReLU(alpha=0.2, name="leaky_relu2_2")(x)
        x = layers.Conv2D(32, (3, 3), padding="same", name="conv2_3")(x)
        z2 = layers.LeakyReLU(alpha=0.2, name="leaky_relu2_3")(x)
        
        # Add residual connection (element-wise addition)
        x = layers.Add(name="add2")([y2, z2])
        x = layers.BatchNormalization(name="bn2_final")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2")(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name="dropout2")(x)
        
        # Dense phase
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(128, activation="tanh", name="dense1")(x)
        x = layers.BatchNormalization(name="bn_dense")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_dense")(x)
        
        # Output layer
        outputs = layers.Dense(2, activation="softmax", name="output")(x)
        
        # Create model
        model = Model(inputs, outputs, name="AttackNetV2_2")
        
        return model


def create_model(architecture_name: str, 
                input_shape: Tuple[int, int, int] = (256, 256, 3),
                dropout_rate: float = 0.5,
                l2_reg: float = 0.0) -> BaseArchitecture:
    """
    Factory function to create model instances by name.
    
    Args:
        architecture_name: Name of the architecture
        input_shape: Shape of input images
        dropout_rate: Dropout rate to use
        
    Returns:
        Instance of the requested architecture
    """
    architectures = {
        "LivenessNet": LivenessNet,
        "AttackNetV1": AttackNetV1,
        "AttackNetV2_1": AttackNetV2_1,
        "AttackNetV2_2": AttackNetV2_2,
    }
    
    if architecture_name not in architectures:
        available = ", ".join(architectures.keys())
        raise ValueError(f"Unknown architecture: {architecture_name}. Available: {available}")
    
    return architectures[architecture_name](input_shape, dropout_rate, l2_reg)


def get_all_architectures(input_shape: Tuple[int, int, int] = (256, 256, 3),
                         dropout_rate: float = 0.5) -> Dict[str, BaseArchitecture]:
    """Get instances of all available architectures."""
    return {
        "LivenessNet": LivenessNet(input_shape, dropout_rate),
        "AttackNetV1": AttackNetV1(input_shape, dropout_rate),
        "AttackNetV2_1": AttackNetV2_1(input_shape, dropout_rate),
        "AttackNetV2_2": AttackNetV2_2(input_shape, dropout_rate),
    }
