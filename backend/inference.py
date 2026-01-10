#!/usr/bin/env python3
"""
Real-time BCI Model Inference Engine
Handles model loading and real-time EEG classification
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Callable
import sys

# Add CODE directory to path
code_dir = Path(__file__).parent.parent / 'CODE'
sys.path.append(str(code_dir))

from preProcessing import EEGPreprocessor, RealTimeBuffer, MOTOR_IMAGERY_CLASSES

class BCIModelInference:
    """
    Real-time BCI model inference engine
    """
    
    def __init__(self, 
                 model_path: str = None,
                 sampling_rate: int = 250,
                 window_size_seconds: float = 4.0,
                 confidence_threshold: float = 0.5):
        """
        Initialize BCI model inference engine
        
        Args:
            model_path: Path to trained model file
            sampling_rate: EEG sampling rate in Hz
            window_size_seconds: Classification window size in seconds
            confidence_threshold: Minimum confidence for predictions
        """
        self.sampling_rate = sampling_rate
        self.window_size = int(window_size_seconds * sampling_rate)
        self.confidence_threshold = confidence_threshold
        
        # Model and preprocessing
        self.model = None
        self.preprocessor = EEGPreprocessor(sampling_rate=sampling_rate)
        self.buffer = RealTimeBuffer(
            channels=22,
            buffer_size=int(10 * sampling_rate),  # 10 second buffer
            window_size=1000  # 4 seconds at 250Hz = 1000 samples
        )
        
        # Performance tracking
        self.classification_history = []
        self.processing_times = []
        self.confidence_scores = []
        
        # Real-time processing
        self.is_processing = False
        self.processing_thread = None
        self.prediction_callback = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load model
        if model_path is None:
            model_path = code_dir / 'models' / 'eegnet_4class_motor_imagery.keras'
        
        self._load_model(model_path)
        
        self.logger.info(f"BCI Inference Engine initialized: {sampling_rate}Hz, "
                        f"Window: {window_size_seconds}s, Model: {Path(model_path).name}")
    
    def _load_model(self, model_path: str):
        """Load the trained BCI model"""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model with custom objects if needed
            self.model = tf.keras.models.load_model(str(model_path))
            
            # Verify model input/output shapes
            input_shape = self.model.input_shape  # (None, 22, 1000, 1)
            output_shape = self.model.output_shape  # (None, 4)
            
            expected_input = (None, 22, 1000, 1)
            expected_output = (None, 4)
            
            if input_shape != expected_input:
                self.logger.warning(f"Model input shape {input_shape} != expected {expected_input}")
            
            if output_shape != expected_output:
                self.logger.warning(f"Model output shape {output_shape} != expected {expected_output}")
            
            # Test model with dummy data
            dummy_input = np.random.randn(1, 22, 1000, 1).astype(np.float32)
            test_output = self.model.predict(dummy_input, verbose=0)
            
            self.logger.info(f"Model loaded successfully: {model_path.name}")
            self.logger.info(f"Input shape: {input_shape}, Output shape: {output_shape}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
    
    def add_eeg_chunk(self, chunk: np.ndarray):
        """
        Add new EEG data chunk to processing buffer
        
        Args:
            chunk: EEG data chunk (channels x samples)
        """
        try:
            # Validate chunk shape
            if chunk.ndim != 2 or chunk.shape[0] != 22:
                self.logger.error(f"Invalid chunk shape: {chunk.shape}, expected (22, samples)")
                return
            
            # Add to buffer
            self.buffer.add_data(chunk)
            
        except Exception as e:
            self.logger.error(f"Error adding EEG chunk: {e}")
    
    def classify_latest_window(self) -> Optional[Dict]:
        """
        Classify the latest window of EEG data
        
        Returns:
            Classification results dictionary or None if insufficient data
        """
        try:
            # Get latest window
            window = self.buffer.get_latest_window()
            if window is None:
                return None
            
            start_time = time.time()
            
            # Preprocess window
            preprocessed = self.preprocessor.preprocess_epoch(window)
            
            # Format for model
            model_input = self.preprocessor.format_for_model(preprocessed)
            
            # Get prediction
            prediction = self.model.predict(model_input, verbose=0)
            probabilities = prediction[0]  # Remove batch dimension
            
            # Get predicted class and confidence
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Create result
            result = {
                'predicted_class': predicted_class,
                'predicted_class_name': MOTOR_IMAGERY_CLASSES[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    MOTOR_IMAGERY_CLASSES[i]: float(probabilities[i]) 
                    for i in range(len(probabilities))
                },
                'processing_time_ms': processing_time,
                'timestamp': time.time(),
                'is_confident': confidence >= self.confidence_threshold
            }
            
            # Update tracking
            self.classification_history.append(result)
            self.processing_times.append(processing_time)
            self.confidence_scores.append(confidence)
            
            # Keep history manageable
            if len(self.classification_history) > 1000:
                self.classification_history = self.classification_history[-500:]
                self.processing_times = self.processing_times[-500:]
                self.confidence_scores = self.confidence_scores[-500:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return None
    
    def start_realtime_processing(self, prediction_callback: Callable[[Dict], None]):
        """
        Start real-time processing of incoming EEG data
        
        Args:
            prediction_callback: Callback function for prediction results
        """
        if self.is_processing:
            self.logger.warning("Real-time processing already active")
            return
        
        self.prediction_callback = prediction_callback
        self.is_processing = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Real-time processing started")
    
    def stop_realtime_processing(self):
        """Stop real-time processing"""
        self.is_processing = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        self.logger.info("Real-time processing stopped")
    
    def _processing_worker(self):
        """Worker thread for real-time classification"""
        try:
            while self.is_processing:
                # Try to classify latest window
                result = self.classify_latest_window()
                
                if result and self.prediction_callback:
                    self.prediction_callback(result)
                
                # Wait a bit before next classification
                time.sleep(0.1)  # 10 Hz classification rate
                
        except Exception as e:
            self.logger.error(f"Processing worker error: {e}")
            self.is_processing = False
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.processing_times:
            return {
                'avg_processing_time_ms': 0,
                'min_processing_time_ms': 0,
                'max_processing_time_ms': 0,
                'avg_confidence': 0,
                'min_confidence': 0,
                'max_confidence': 0,
                'total_classifications': 0,
                'confident_classifications': 0
            }
        
        confident_count = sum(1 for score in self.confidence_scores 
                            if score >= self.confidence_threshold)
        
        return {
            'avg_processing_time_ms': np.mean(self.processing_times),
            'min_processing_time_ms': np.min(self.processing_times),
            'max_processing_time_ms': np.max(self.processing_times),
            'avg_confidence': np.mean(self.confidence_scores),
            'min_confidence': np.min(self.confidence_scores),
            'max_confidence': np.max(self.confidence_scores),
            'total_classifications': len(self.classification_history),
            'confident_classifications': confident_count,
            'confidence_rate': confident_count / len(self.confidence_scores) if self.confidence_scores else 0
        }
    
    def get_session_accuracy(self, true_labels: List[int]) -> Dict:
        """
        Calculate session accuracy compared to true labels
        
        Args:
            true_labels: List of true class labels for recent predictions
            
        Returns:
            Accuracy metrics dictionary
        """
        if not self.classification_history or not true_labels:
            return {'accuracy': 0, 'total_predictions': 0}
        
        # Get recent predictions (same length as true labels)
        recent_predictions = self.classification_history[-len(true_labels):]
        predicted_labels = [pred['predicted_class'] for pred in recent_predictions]
        
        if len(predicted_labels) != len(true_labels):
            return {'accuracy': 0, 'total_predictions': len(predicted_labels)}
        
        # Calculate accuracy
        correct = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)
        accuracy = correct / len(true_labels)
        
        # Per-class accuracy
        per_class_accuracy = {}
        for class_idx in range(4):
            class_mask = [true == class_idx for true in true_labels]
            if any(class_mask):
                class_correct = sum(1 for i, (pred, true) in enumerate(zip(predicted_labels, true_labels))
                                  if class_mask[i] and pred == true)
                class_total = sum(class_mask)
                per_class_accuracy[MOTOR_IMAGERY_CLASSES[class_idx]] = class_correct / class_total
            else:
                per_class_accuracy[MOTOR_IMAGERY_CLASSES[class_idx]] = 0
        
        return {
            'accuracy': accuracy,
            'total_predictions': len(predicted_labels),
            'correct_predictions': correct,
            'per_class_accuracy': per_class_accuracy
        }
    
    def reset_session(self):
        """Reset session data"""
        self.buffer.clear()
        self.classification_history.clear()
        self.processing_times.clear()
        self.confidence_scores.clear()
        self.logger.info("Session data reset")

# Convenience function for testing
def test_inference_engine():
    """Test the BCI inference engine"""
    logging.basicConfig(level=logging.INFO)
    
    def prediction_handler(result):
        print(f"Prediction: {result['predicted_class_name']} "
              f"(confidence: {result['confidence']:.3f}, "
              f"time: {result['processing_time_ms']:.1f}ms)")
    
    try:
        # Initialize inference engine
        engine = BCIModelInference()
        
        # Generate test data
        test_chunk = np.random.randn(22, 32).astype(np.float32)
        
        # Add some chunks to build up buffer
        for _ in range(100):  # Should give us enough for classification
            engine.add_eeg_chunk(test_chunk)
        
        # Test single classification
        result = engine.classify_latest_window()
        if result:
            print("Single classification result:", result)
        
        # Test real-time processing
        engine.start_realtime_processing(prediction_handler)
        
        # Add more data
        for _ in range(50):
            engine.add_eeg_chunk(test_chunk)
            time.sleep(0.02)  # 50 Hz data rate
        
        engine.stop_realtime_processing()
        
        # Get performance stats
        stats = engine.get_performance_stats()
        print("Performance stats:", stats)
        
        print("Test completed successfully")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_inference_engine()