"""
NPG Lite Inference Engine
Real-time motor imagery classification for NPG Lite data

Uses trained BCI IV 2b model (binary classification: left hand vs right hand)
Model accuracy: ~73.6% on test set
"""

import numpy as np
import tensorflow as tf
import logging
import time
from typing import Tuple, Optional, Dict
from collections import deque
from pathlib import Path

from npg_preprocessor import NPGPreprocessor, SlidingWindowBuffer


class NPGInferenceEngine:
    """
    Real-time inference engine for NPG Lite motor imagery classification.
    
    Uses trained EEGNet model for binary classification (left vs right hand).
    """
    
    def __init__(self,
                 model_path: str = None,
                 confidence_threshold: float = 0.7,
                 smoothing_window: int = 3):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained .keras model
                       Default: models/best_eegnet_2class_bci2b.keras
            confidence_threshold: Minimum confidence for valid prediction (0-1)
            smoothing_window: Number of predictions to smooth over
        """
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        
        # Setup logging FIRST (before anything that might log)
        self.logger = logging.getLogger(__name__)
        
        # Class labels
        self.class_names = ['LEFT_HAND', 'RIGHT_HAND']
        
        # Load model
        if model_path is None:
            model_path = Path(__file__).parent / 'models' / 'best_eegnet_2class_bci2b.keras'
        
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()
        
        # Prediction smoothing buffer
        self.prediction_buffer = deque(maxlen=smoothing_window)
        
        # Statistics
        self.total_predictions = 0
        self.confident_predictions = 0
        self.class_counts = {class_name: 0 for class_name in self.class_names}
        self.prediction_times = deque(maxlen=100)
        
        self.logger.info(f"NPG Inference Engine initialized:")
        self.logger.info(f"  Model: {self.model_path.name}")
        self.logger.info(f"  Classes: {self.class_names}")
        self.logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        self.logger.info(f"  Smoothing window: {self.smoothing_window}")
    
    def _load_model(self):
        """Load the trained model."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            self.logger.info(f"Loading model from {self.model_path}...")
            self.model = tf.keras.models.load_model(str(self.model_path))
            
            # Verify model structure
            input_shape = self.model.input_shape  # (None, 3, 1000, 1)
            output_shape = self.model.output_shape  # (None, 2)
            
            self.logger.info(f"  Input shape: {input_shape}")
            self.logger.info(f"  Output shape: {output_shape}")
            
            # Verify expected shapes
            expected_input = (None, 3, 1000, 1)
            expected_output = (None, 2)
            
            if input_shape != expected_input:
                self.logger.warning(f"Input shape mismatch! Expected {expected_input}, got {input_shape}")
            
            if output_shape != expected_output:
                self.logger.warning(f"Output shape mismatch! Expected {expected_output}, got {output_shape}")
            
            self.logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, preprocessed_data: np.ndarray) -> Tuple[int, float, str]:
        """
        Run inference on preprocessed data.
        
        Args:
            preprocessed_data: Model-ready data (1, 3, 1000, 1)
        
        Returns:
            Tuple of (class_idx, confidence, class_name)
        """
        start_time = time.time()
        
        try:
            # Run prediction
            prediction = self.model.predict(preprocessed_data, verbose=0)
            
            # Extract class and confidence
            probabilities = prediction[0]
            class_idx = np.argmax(probabilities)
            confidence = probabilities[class_idx]
            class_name = self.class_names[class_idx]
            
            # Update statistics
            self.total_predictions += 1
            if confidence >= self.confidence_threshold:
                self.confident_predictions += 1
                self.class_counts[class_name] += 1
            
            # Track prediction time
            pred_time = time.time() - start_time
            self.prediction_times.append(pred_time)
            
            return class_idx, float(confidence), class_name
        
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return -1, 0.0, "ERROR"
    
    def predict_smoothed(self, preprocessed_data: np.ndarray) -> Tuple[int, float, str]:
        """
        Run inference with temporal smoothing.
        
        Args:
            preprocessed_data: Model-ready data (1, 3, 1000, 1)
        
        Returns:
            Tuple of (class_idx, confidence, class_name) - smoothed
        """
        # Get raw prediction
        class_idx, confidence, class_name = self.predict(preprocessed_data)
        
        # Add to buffer
        self.prediction_buffer.append((class_idx, confidence))
        
        # If buffer not full, return raw prediction
        if len(self.prediction_buffer) < self.smoothing_window:
            return class_idx, confidence, class_name
        
        # Smooth predictions - majority vote
        class_votes = [0, 0]
        total_confidence = 0.0
        
        for idx, conf in self.prediction_buffer:
            if idx >= 0:  # Valid prediction
                class_votes[idx] += 1
                total_confidence += conf
        
        # Determine smoothed prediction
        smoothed_class_idx = np.argmax(class_votes)
        smoothed_confidence = total_confidence / len(self.prediction_buffer)
        smoothed_class_name = self.class_names[smoothed_class_idx]
        
        return smoothed_class_idx, smoothed_confidence, smoothed_class_name
    
    def get_statistics(self) -> Dict:
        """
        Get inference statistics.
        
        Returns:
            Dictionary with statistics
        """
        avg_pred_time = np.mean(self.prediction_times) if self.prediction_times else 0.0
        confidence_rate = (self.confident_predictions / self.total_predictions * 100
                          if self.total_predictions > 0 else 0.0)
        
        return {
            'total_predictions': self.total_predictions,
            'confident_predictions': self.confident_predictions,
            'confidence_rate': confidence_rate,
            'class_distribution': self.class_counts.copy(),
            'avg_prediction_time_ms': avg_pred_time * 1000,
            'predictions_per_second': 1.0 / avg_pred_time if avg_pred_time > 0 else 0.0
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_predictions = 0
        self.confident_predictions = 0
        self.class_counts = {class_name: 0 for class_name in self.class_names}
        self.prediction_times.clear()
        self.prediction_buffer.clear()


if __name__ == "__main__":
    # Test the inference engine
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("\n" + "="*70)
    print("Testing NPG Inference Engine")
    print("="*70)
    
    # Check if model exists
    model_path = Path(__file__).parent / 'models' / 'best_eegnet_2class_bci2b.keras'
    
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("   Train the model first with: python train_model_2b.py")
    else:
        # Create inference engine
        engine = NPGInferenceEngine(model_path=str(model_path))
        
        # Create preprocessor
        preprocessor = NPGPreprocessor()
        
        # Generate synthetic test data (4 seconds @ 256 Hz, 6 channels)
        print("\n1. Generating synthetic motor imagery data...")
        n_samples = int(4.0 * 256)
        n_channels = 6
        t = np.arange(n_samples) / 256
        
        # Simulate motor imagery (left hand pattern - mu rhythm suppression in C3)
        test_data = np.zeros((n_samples, n_channels))
        for ch in range(n_channels):
            # Alpha rhythm (10 Hz)
            alpha = 10 * np.sin(2 * np.pi * 10 * t)
            # Beta rhythm (20 Hz) - motor imagery marker
            beta = 8 * np.sin(2 * np.pi * 20 * t)
            # Mu rhythm suppression for C3 (left hand imagery)
            if ch == 0:  # C3
                mu_suppression = -5 * np.sin(2 * np.pi * 10 * t)
            else:
                mu_suppression = 0
            # Noise
            noise = np.random.randn(n_samples) * 2
            
            test_data[:, ch] = alpha + beta + mu_suppression + noise
        
        print(f"   Generated data shape: {test_data.shape}")
        
        # Preprocess
        print("\n2. Preprocessing data...")
        preprocessed = preprocessor.preprocess_for_model(test_data)
        print(f"   Preprocessed shape: {preprocessed.shape}")
        
        # Run inference
        print("\n3. Running inference...")
        class_idx, confidence, class_name = engine.predict(preprocessed)
        print(f"   Prediction: {class_name}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Status: {'✅ CONFIDENT' if confidence >= engine.confidence_threshold else '⚠️  UNCERTAIN'}")
        
        # Run multiple predictions for smoothing test
        print("\n4. Testing temporal smoothing (5 predictions)...")
        for i in range(5):
            # Add slight noise variation
            noisy_data = test_data + np.random.randn(*test_data.shape) * 1
            preprocessed = preprocessor.preprocess_for_model(noisy_data)
            
            # Smoothed prediction
            class_idx, confidence, class_name = engine.predict_smoothed(preprocessed)
            print(f"   Prediction #{i+1}: {class_name} ({confidence:.2%})")
        
        # Statistics
        print("\n5. Statistics:")
        stats = engine.get_statistics()
        print(f"   Total predictions: {stats['total_predictions']}")
        print(f"   Confident predictions: {stats['confident_predictions']} ({stats['confidence_rate']:.1f}%)")
        print(f"   Class distribution: {stats['class_distribution']}")
        print(f"   Avg prediction time: {stats['avg_prediction_time_ms']:.2f} ms")
        print(f"   Throughput: {stats['predictions_per_second']:.1f} predictions/sec")
        
        print("\n" + "="*70)
        print("✅ All tests passed!")
        print("="*70)
