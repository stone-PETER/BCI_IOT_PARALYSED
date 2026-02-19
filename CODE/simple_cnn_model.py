"""
Example: Simple CNN Model for BCI Classification
This demonstrates how to create and integrate a new model architecture

This is a simpler baseline model that can be useful for:
- Quick prototyping
- Establishing baseline performance
- Educational purposes
- Debugging data pipelines
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dropout, Flatten, Dense
)
import numpy as np
import yaml
import logging
from typing import Tuple, Optional


class SimpleCNN:
    """
    Simple CNN baseline model for BCI classification.
    
    Architecture:
    1. Convolutional block 1: Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout
    2. Convolutional block 2: Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout
    3. Flatten -> Dense -> Output
    
    This model is simpler than EEGNet and can serve as a baseline.
    """
    
    def __init__(self, config_path: str = "config_2b.yaml"):
        """
        Initialize SimpleCNN with configuration parameters.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_config = self.config['model']
        
        # Model architecture parameters
        self.nb_classes = self.model_config['nb_classes']
        self.chans = self.model_config['chans']
        self.samples = self.model_config['samples']
        self.dropoutRate = self.model_config.get('dropoutRate', 0.5)
        
        # SimpleCNN specific parameters (with defaults)
        self.filters1 = self.model_config.get('filters1', 16)
        self.filters2 = self.model_config.get('filters2', 32)
        self.kernel_size = self.model_config.get('kernel_size', (1, 32))
        self.pool_size = self.model_config.get('pool_size', (1, 4))
        self.dense_units = self.model_config.get('dense_units', 128)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        self.model = None
    
    def build_model(self) -> Model:
        """
        Build the SimpleCNN model architecture.
        
        Returns:
            Compiled Keras model ready for training
        """
        self.logger.info("Building SimpleCNN model...")
        
        # Input layer: (channels, samples, 1)
        input_layer = Input(shape=(self.chans, self.samples, 1))
        
        ##################################################################
        # Convolutional Block 1
        ##################################################################
        
        x = Conv2D(
            filters=self.filters1,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False
        )(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=self.pool_size)(x)
        x = Dropout(self.dropoutRate)(x)
        
        ##################################################################
        # Convolutional Block 2
        ##################################################################
        
        x = Conv2D(
            filters=self.filters2,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=self.pool_size)(x)
        x = Dropout(self.dropoutRate)(x)
        
        ##################################################################
        # Classification Block
        ##################################################################
        
        # Flatten
        x = Flatten()(x)
        
        # Dense layer
        x = Dense(self.dense_units, activation='relu')(x)
        x = Dropout(self.dropoutRate)(x)
        
        # Output layer
        output = Dense(self.nb_classes, activation='softmax', name='classification')(x)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output)
        
        # Log model info
        self._log_model_info()
        
        return self.model
    
    def _log_model_info(self):
        """Log model architecture information."""
        if self.model:
            total_params = self.model.count_params()
            
            self.logger.info("SimpleCNN model built successfully:")
            self.logger.info(f"  - Input shape: {self.model.input_shape}")
            self.logger.info(f"  - Output classes: {self.nb_classes}")
            self.logger.info(f"  - Total parameters: {total_params:,}")
            
            # Calculate trainable params
            trainable_count = sum([
                tf.size(w).numpy() for w in self.model.trainable_weights
            ])
            
            self.logger.info(f"  - Trainable parameters: {trainable_count:,}")
    
    def compile_model(self, optimizer: str = 'adam', learning_rate: float = 0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            self.build_model()
        
        # Create optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
            self.logger.warning(f"Unknown optimizer '{optimizer}', using Adam")
        
        # Compile
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info(f"Model compiled with {optimizer} optimizer (lr={learning_rate})")
    
    def get_model_summary(self) -> str:
        """
        Get formatted model architecture summary.
        
        Returns:
            String representation of model architecture
        """
        if self.model is None:
            return "Model not built yet. Call build_model() first."
        
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()


def create_simple_cnn(config_path: str = "config_2b.yaml") -> SimpleCNN:
    """
    Convenience function to create a SimpleCNN model.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SimpleCNN instance with built model
    """
    model = SimpleCNN(config_path)
    model.build_model()
    return model


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("SimpleCNN Model Demo")
    print("=" * 60)
    
    try:
        # Create model
        print("\n1. Creating SimpleCNN model...")
        simple_cnn = create_simple_cnn('config_2b.yaml')
        
        # Show summary
        print("\n2. Model Architecture:")
        print(simple_cnn.get_model_summary())
        
        # Compile model
        print("\n3. Compiling model...")
        simple_cnn.compile_model(optimizer='adam', learning_rate=0.001)
        
        # Test with dummy data
        print("\n4. Testing with dummy data...")
        dummy_input = np.random.randn(16, 3, 1000, 1).astype(np.float32)
        dummy_output = simple_cnn.model.predict(dummy_input, verbose=0)
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {dummy_output.shape}")
        print(f"   Output probabilities (first sample): {dummy_output[0]}")
        
        print("\n✓ SimpleCNN model is working correctly!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
