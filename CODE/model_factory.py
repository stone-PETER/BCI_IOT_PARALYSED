"""
Model Factory for BCI Classification Models
Centralized model creation and configuration management

This module allows easy switching between different model architectures
via configuration files without changing code.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from tensorflow.keras.models import Model
import tensorflow as tf

# Import available model architectures
from eegnet_model import EEGNet
from simple_cnn_model import SimpleCNN

# Dictionary of available models
AVAILABLE_MODELS = {
    'eegnet': EEGNet,
    'simplecnn': SimpleCNN,
    # Add more models here as you create them:
    # 'deepconvnet': DeepConvNet,
    # 'shallowconvnet': ShallowConvNet,
    # 'eegnet_attention': EEGNetAttention,
}


class ModelFactory:
    """
    Factory class for creating BCI classification models.
    
    This factory pattern allows you to:
    1. Switch models by changing config files only
    2. Add new models without modifying existing code
    3. Maintain consistent interfaces across different architectures
    """
    
    def __init__(self, config_path: str = "config_2b.yaml"):
        """
        Initialize the model factory.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Resolve path relative to script location
        self.config_path = config_path
        if not Path(config_path).is_absolute():
            script_dir = Path(__file__).parent.resolve()
            self.config_path = str(script_dir / config_path)
        
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Get model configuration
        self.model_config = self.config.get('model', {})
        self.model_type = self.model_config.get('architecture', 'eegnet').lower()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Model Factory initialized with architecture: {self.model_type}")
    
    def create_model(self, compile_model: bool = True) -> Model:
        """
        Create and return a model based on configuration.
        
        Args:
            compile_model: Whether to compile the model with optimizer
            
        Returns:
            Keras model instance
            
        Raises:
            ValueError: If specified model type is not available
        """
        if self.model_type not in AVAILABLE_MODELS:
            raise ValueError(
                f"Model type '{self.model_type}' not found. "
                f"Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        
        self.logger.info(f"Creating {self.model_type} model...")
        
        # Get the model class
        model_class = AVAILABLE_MODELS[self.model_type]
        
        # Instantiate the model with the config path
        model_instance = model_class(config_path=self.config_path)
        
        # Build the model
        model = model_instance.build_model()
        
        # Optionally compile the model
        if compile_model:
            self.logger.info("Compiling model...")
            training_config = self.config.get('training', {})
            learning_rate = training_config.get('learning_rate', 0.001)
            optimizer_type = training_config.get('optimizer', 'adam')
            
            model_instance.compile_model(
                optimizer=optimizer_type,
                learning_rate=learning_rate
            )
        
        self.logger.info(f"Model created successfully: {self.model_type}")
        self._log_model_info(model)
        
        return model_instance
    
    def _log_model_info(self, model_instance):
        """Log model architecture information."""
        if hasattr(model_instance, 'model') and model_instance.model:
            model = model_instance.model
            total_params = model.count_params()
            
            # Count trainable params
            trainable_params = sum(
                [tf.size(w).numpy() for w in model.trainable_weights]
            )
            
            self.logger.info(f"Model Parameters:")
            self.logger.info(f"  - Total: {total_params:,}")
            self.logger.info(f"  - Trainable: {trainable_params:,}")
            self.logger.info(f"  - Input shape: {model.input_shape}")
            self.logger.info(f"  - Output shape: {model.output_shape}")
    
    @staticmethod
    def get_available_models():
        """Return list of available model architectures."""
        return list(AVAILABLE_MODELS.keys())
    
    @staticmethod
    def add_model(name: str, model_class):
        """
        Register a new model architecture.
        
        Args:
            name: Name identifier for the model
            model_class: Class that implements the model
        """
        if name in AVAILABLE_MODELS:
            logging.warning(f"Overwriting existing model: {name}")
        
        AVAILABLE_MODELS[name] = model_class
        logging.info(f"Registered model: {name}")


def create_model_from_config(config_path: str, compile_model: bool = True) -> Any:
    """
    Convenience function to create a model from a config file.
    
    Args:
        config_path: Path to configuration YAML file
        compile_model: Whether to compile the model
        
    Returns:
        Model instance ready for training
        
    Example:
        >>> model = create_model_from_config('config_opt2.yaml')
        >>> model.model.summary()
    """
    factory = ModelFactory(config_path)
    return factory.create_model(compile_model=compile_model)


def list_available_models():
    """Print available model architectures."""
    models = ModelFactory.get_available_models()
    print("\nAvailable BCI Model Architectures:")
    print("-" * 40)
    for i, model_name in enumerate(models, 1):
        print(f"{i}. {model_name}")
    print("-" * 40)
    print(f"Total: {len(models)} models available\n")
    return models


if __name__ == "__main__":
    # Demo usage
    print("BCI Model Factory Demo")
    print("=" * 50)
    
    # List available models
    list_available_models()
    
    # Example: Create default EEGNet model
    print("\nCreating default model from config_2b.yaml...")
    try:
        factory = ModelFactory('config_2b.yaml')
        model_instance = factory.create_model(compile_model=False)
        print(f"✓ Successfully created {factory.model_type} model")
        
        if hasattr(model_instance, 'model'):
            model_instance.model.summary()
    except Exception as e:
        print(f"✗ Error: {e}")
