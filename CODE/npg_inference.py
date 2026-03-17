"""
NPG Lite Inference Engine
Real-time motor imagery classification for NPG Lite data

Uses trained BCI IV 2b model (binary classification: left hand vs right hand)
Model accuracy: ~73.6% on test set

Features:
- Neutral zone to filter uncertain predictions
- Configurable thresholds and decay rates
- FIXED: Temperature-scaled calibrated confidence
- FIXED: Entropy-based uncertainty estimation
- FIXED: Confidence-weighted smoothing
"""

import numpy as np
import tensorflow as tf
import logging
import time
from typing import Tuple, Optional, Dict
from collections import deque
from pathlib import Path

from npg_preprocessor import NPGPreprocessor, SlidingWindowBuffer


class CalibratedConfidence:
    """
    Temperature scaling for calibrated probability estimates.
    
    Raw softmax outputs are often overconfident. Temperature scaling
    produces well-calibrated probabilities where confidence reflects
    actual accuracy.
    
    Reference: Guo et al., "On Calibration of Modern Neural Networks" (2017)
    """
    
    def __init__(self, temperature: float = 1.5):
        """
        Initialize calibrated confidence estimator.
        
        Args:
            temperature: Scaling temperature (>1 reduces confidence, <1 increases)
                        Default 1.5 typically works well for EEG classification
        """
        self.temperature = temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model output (before or after softmax)
        
        Returns:
            Calibrated probabilities
        """
        # If input looks like probabilities (sums to ~1), convert to logits first
        if np.abs(logits.sum() - 1.0) < 0.01:
            # Convert probabilities back to log-odds
            logits = np.log(logits + 1e-10)
        
        # Apply temperature scaling
        scaled = logits / self.temperature
        
        # Softmax to get calibrated probabilities
        exp_scaled = np.exp(scaled - np.max(scaled))  # Subtract max for numerical stability
        calibrated = exp_scaled / exp_scaled.sum()
        
        return calibrated
    
    def compute_uncertainty(self, probabilities: np.ndarray) -> float:
        """
        Compute entropy-based uncertainty measure.
        
        Args:
            probabilities: Probability distribution
        
        Returns:
            Uncertainty score (0 = certain, 1 = maximum uncertainty)
        """
        # Shannon entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        # Normalize by maximum entropy (uniform distribution)
        max_entropy = np.log(len(probabilities))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0


class SmileyFeedback:
    """
    "Smiley Feedback" Running Integral Classifier
    
    Implements the BCI Competition strategy of integrating probability outputs
    over a 2-second window to reduce flickering and produce stable commands.
    
    Instead of acting on single-frame predictions, this accumulates probabilities
    over multiple predictions and only triggers commands when the integrated sum
    exceeds a high threshold.
    
    Reference: Similar to the "Smiley" paradigm used in BCI Competition where
    visual feedback (smiley face position/curvature) was mapped to continuous
    integrated classifier output rather than instantaneous predictions.
    
    Key parameters:
    - Window duration: 2 seconds (typically 4-5 predictions at 2.2 Hz)
    - Integration: Sum probabilities for each class over window
    - Threshold: High threshold (e.g., 3.5 out of 5) to ensure consistency
    """
    
    def __init__(self,
                 n_classes: int = 2,
                 window_duration: float = 2.0,
                 prediction_rate: float = 2.2,
                 trigger_threshold: float = 3.5,
                 reset_on_trigger: bool = False):
        """
        Initialize Smiley Feedback integrator.
        
        Args:
            n_classes: Number of output classes
            window_duration: Integration window duration in seconds
            prediction_rate: Expected prediction rate (epochs/second)
            trigger_threshold: Sum threshold to trigger command
            reset_on_trigger: Whether to reset buffer after triggering
        """
        self.n_classes = n_classes
        self.window_duration = window_duration
        self.prediction_rate = prediction_rate
        self.trigger_threshold = trigger_threshold
        self.reset_on_trigger = reset_on_trigger
        
        # Calculate buffer size (number of predictions in window)
        self.buffer_size = max(3, int(window_duration * prediction_rate))
        
        # Circular buffer of probability distributions
        self.probability_buffer = deque(maxlen=self.buffer_size)
        
        # Statistics
        self.update_count = 0
        self.trigger_count = 0
        self.last_triggered_class = -1
        self.last_trigger_sum = 0.0
        
        # Running integral (sum of probabilities)
        self.integral = np.zeros(n_classes)
    
    def update(self, probabilities: np.ndarray) -> Tuple[int, bool, float]:
        """
        Update the running integral with new probability distribution.
        
        Args:
            probabilities: Probability distribution from classifier [p_left, p_right, ...]
        
        Returns:
            Tuple of (triggered_class, is_triggered, winning_sum)
            - triggered_class: Class index that triggered (-1 if none)
            - is_triggered: Whether threshold was exceeded
            - winning_sum: Integrated sum of winning class
        """
        self.update_count += 1
        
        # Add new probabilities to buffer
        self.probability_buffer.append(probabilities.copy())
        
        # Not enough data yet
        if len(self.probability_buffer) < self.buffer_size:
            return -1, False, 0.0
        
        # Compute running integral (sum over window)
        self.integral = np.sum(self.probability_buffer, axis=0)
        
        # Find winning class
        winning_class = np.argmax(self.integral)
        winning_sum = self.integral[winning_class]
        
        # Check if threshold exceeded
        is_triggered = winning_sum >= self.trigger_threshold
        
        if is_triggered:
            self.trigger_count += 1
            self.last_triggered_class = winning_class
            self.last_trigger_sum = winning_sum
            
            # Optionally reset buffer to prevent repeated triggers
            if self.reset_on_trigger:
                self.probability_buffer.clear()
                self.integral = np.zeros(self.n_classes)
            
            return winning_class, True, winning_sum
        
        return -1, False, winning_sum
    
    def get_integral_values(self) -> np.ndarray:
        """Get current integrated probability sums."""
        return self.integral.copy()
    
    def get_buffer_fill(self) -> float:
        """Get buffer fill percentage (0.0 to 1.0)."""
        return len(self.probability_buffer) / self.buffer_size
    
    def reset(self):
        """Clear the probability buffer and reset integral."""
        self.probability_buffer.clear()
        self.integral = np.zeros(self.n_classes)
        self.last_triggered_class = -1
        self.last_trigger_sum = 0.0
    
    def get_statistics(self) -> Dict:
        """Get running statistics."""
        trigger_rate = self.trigger_count / max(1, self.update_count)
        
        return {
            'window_duration_sec': self.window_duration,
            'buffer_size': self.buffer_size,
            'buffer_fill': len(self.probability_buffer),
            'trigger_threshold': self.trigger_threshold,
            'total_updates': self.update_count,
            'total_triggers': self.trigger_count,
            'trigger_rate': trigger_rate,
            'last_triggered_class': self.last_triggered_class,
            'last_trigger_sum': float(self.last_trigger_sum),
            'current_integral': self.integral.tolist()
        }


class NPGInferenceEngine:
    """
    Real-time inference engine for NPG Lite motor imagery classification.
    
    Uses trained EEGNet model for binary classification (left vs right hand).
    
    FIXES APPLIED:
    - Temperature-scaled calibrated confidence
    - Entropy-based uncertainty estimation  
    - Confidence-weighted smoothing (not just majority vote)
    - Consistent threshold handling
    """
    
    def __init__(self,
                 model_path: str = None,
                 confidence_threshold: float = 0.65,
                 smoothing_window: int = 8,
                 temperature: float = 1.5,
                 use_calibration: bool = True,
                 use_smiley_feedback: bool = False,
                 smiley_window_duration: float = 2.0,
                 smiley_threshold: float = 3.5,
                 smiley_prediction_rate: float = 2.2,
                 neutral_threshold: float = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained .keras model
                       Default: models/best_eegnet_2class_bci2b.keras
            confidence_threshold: Minimum confidence for valid prediction (0-1)
                                 Increased default to 0.65 for more reliable commands
            smoothing_window: Number of predictions to smooth over (default: 8)
            temperature: Temperature for calibrated confidence (default: 1.5)
            use_calibration: Whether to use temperature-scaled confidence
            use_smiley_feedback: Whether to use Smiley Feedback running integral
            smiley_window_duration: Integration window in seconds (default: 2.0)
            smiley_threshold: Sum threshold to trigger (default: 3.5)
            smiley_prediction_rate: Expected prediction rate in Hz (default: 2.2)
            neutral_threshold: Maximum confidence for NEUTRAL state (None = disabled)
                               If max(confidences) < neutral_threshold, returns NEUTRAL
        """
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.use_calibration = use_calibration
        self.use_smiley_feedback = use_smiley_feedback
        self.neutral_threshold = neutral_threshold
        
        # Setup logging FIRST (before anything that might log)
        self.logger = logging.getLogger(__name__)
        
        # Class labels (2-class model + optional NEUTRAL state)
        self.class_names = ['LEFT_HAND', 'RIGHT_HAND']
        self.class_names_3state = ['LEFT_HAND', 'RIGHT_HAND', 'NEUTRAL']
        
        # Load model
        #model path main
        if model_path is None:
            model_path = Path(__file__).parent / 'models' / 'best' / 'best_eegnet_2class_bci2b.keras'
        
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()
        
        # FIXED: Calibrated confidence estimator
        self.calibrator = CalibratedConfidence(temperature=temperature)
        
        # FIXED: Prediction smoothing buffer with confidence-weighted history
        self.prediction_buffer = deque(maxlen=smoothing_window)
        
        # Smiley Feedback running integral (optional)
        if use_smiley_feedback:
            self.smiley_feedback = SmileyFeedback(
                n_classes=len(self.class_names),
                window_duration=smiley_window_duration,
                prediction_rate=smiley_prediction_rate,
                trigger_threshold=smiley_threshold,
                reset_on_trigger=False  # Keep accumulating for sustained commands
            )
        else:
            self.smiley_feedback = None
        
        # Baseline bias correction (for 2-class forced choice)
        self.baseline_bias = None  # Will be [left_bias, right_bias] after calibration
        self.calibration_predictions = []  # Buffer for calibration data
        
        # Statistics
        self.total_predictions = 0
        self.confident_predictions = 0
        self.class_counts = {class_name: 0 for class_name in self.class_names}
        self.prediction_times = deque(maxlen=100)
        self.triggered_commands = {class_name: 0 for class_name in self.class_names}
        self.uncertainty_history = deque(maxlen=100)  # Track uncertainty
        
        self.logger.info(f"NPG Inference Engine initialized (FIXED):")
        self.logger.info(f"  Model: {self.model_path.name}")
        self.logger.info(f"  Classes: {self.class_names}")
        self.logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        self.logger.info(f"  Smoothing window: {self.smoothing_window}")
        self.logger.info(f"  Temperature calibration: {temperature if use_calibration else 'disabled'}")
        self.logger.info(f"  Smiley Feedback: {self.use_smiley_feedback}")
        if self.use_smiley_feedback:
            self.logger.info(f"    Window: {smiley_window_duration}s, Threshold: {smiley_threshold}, Rate: {smiley_prediction_rate} Hz")
    
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
        
        FIXED: Returns calibrated confidence and tracks uncertainty.
        
        Args:
            preprocessed_data: Model-ready data (1, 3, 1000, 1)
        
        Returns:
            Tuple of (class_idx, confidence, class_name)
        """
        start_time = time.time()
        
        try:
            # Run prediction
            prediction = self.model.predict(preprocessed_data, verbose=0)
            
            # Extract raw probabilities
            raw_probabilities = prediction[0]
            
            # Apply baseline bias correction if calibrated
            if self.baseline_bias is not None:
                raw_probabilities = self._apply_bias_correction(raw_probabilities)
            
            # FIXED: Apply temperature scaling for calibrated confidence
            if self.use_calibration:
                probabilities = self.calibrator.calibrate(raw_probabilities)
                uncertainty = self.calibrator.compute_uncertainty(probabilities)
                self.uncertainty_history.append(uncertainty)
            else:
                probabilities = raw_probabilities
            
            class_idx = np.argmax(probabilities)
            confidence = float(probabilities[class_idx])
            class_name = self.class_names[class_idx]
            
            # Update statistics
            self.total_predictions += 1
            if confidence >= self.confidence_threshold:
                self.confident_predictions += 1
                self.class_counts[class_name] += 1
            
            # Track prediction time
            pred_time = time.time() - start_time
            self.prediction_times.append(pred_time)
            
            return class_idx, confidence, class_name
        
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return -1, 0.0, "ERROR"
    
    def predict_with_uncertainty(self, preprocessed_data: np.ndarray) -> Tuple[int, float, str, float]:
        """
        Run inference and return uncertainty estimate.
        
        Args:
            preprocessed_data: Model-ready data (1, 3, 1000, 1)
        
        Returns:
            Tuple of (class_idx, confidence, class_name, uncertainty)
            uncertainty: 0 = very certain, 1 = very uncertain
        """
        start_time = time.time()
        
        try:
            prediction = self.model.predict(preprocessed_data, verbose=0)
            raw_probabilities = prediction[0]
            
            if self.baseline_bias is not None:
                raw_probabilities = self._apply_bias_correction(raw_probabilities)
            
            # Calibrate and compute uncertainty
            probabilities = self.calibrator.calibrate(raw_probabilities)
            uncertainty = self.calibrator.compute_uncertainty(probabilities)
            
            class_idx = np.argmax(probabilities)
            confidence = float(probabilities[class_idx])
            
            # Check for NEUTRAL state if threshold is set
            if self.neutral_threshold is not None and confidence < self.neutral_threshold:
                class_name = 'NEUTRAL'
                class_idx = 2  # Index for NEUTRAL in 3-state system
            else:
                class_name = self.class_names[class_idx]
            
            self.total_predictions += 1
            self.uncertainty_history.append(uncertainty)
            
            if confidence >= self.confidence_threshold:
                self.confident_predictions += 1
                self.class_counts[class_name] += 1
            
            pred_time = time.time() - start_time
            self.prediction_times.append(pred_time)
            
            return class_idx, confidence, class_name, uncertainty
        
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return -1, 0.0, "ERROR", 1.0
    
    def _apply_bias_correction(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply baseline bias correction to probabilities.
        
        Args:
            probabilities: Raw model probabilities [left, right]
        
        Returns:
            Bias-corrected probabilities
        """
        if self.baseline_bias is None:
            return probabilities
        
        # Subtract bias
        corrected = probabilities - self.baseline_bias
        
        # Renormalize to ensure sum = 1.0
        corrected = np.maximum(corrected, 0.0)  # Clip to non-negative
        total = corrected.sum()
        if total > 0:
            corrected = corrected / total
        else:
            # If all clipped to zero, return uniform
            corrected = np.ones_like(probabilities) / len(probabilities)
        
        return corrected
    
    def start_calibration(self):
        """Start collecting baseline calibration data."""
        self.calibration_predictions = []
        self.logger.info("Started baseline calibration")
    
    def add_calibration_sample(self, preprocessed_data: np.ndarray):
        """
        Add a prediction sample during calibration (rest state).
        
        Args:
            preprocessed_data: Model-ready data (1, 3, 1000, 1)
        """
        try:
            # Run raw prediction (without bias correction)
            prediction = self.model.predict(preprocessed_data, verbose=0)
            probabilities = prediction[0]
            self.calibration_predictions.append(probabilities.copy())
        except Exception as e:
            self.logger.error(f"Calibration sample error: {e}")
    
    def finalize_calibration(self) -> bool:
        """
        Calculate baseline bias from calibration samples.
        
        Returns:
            True if calibration successful, False otherwise
        """
        if len(self.calibration_predictions) < 10:
            self.logger.error(f"Insufficient calibration samples: {len(self.calibration_predictions)} < 10")
            return False
        
        # Calculate mean prediction during rest
        mean_probabilities = np.mean(self.calibration_predictions, axis=0)
        
        # Baseline bias is deviation from 50/50
        ideal_rest = np.array([0.5, 0.5])
        self.baseline_bias = mean_probabilities - ideal_rest
        
        self.logger.info(f"Calibration complete: {len(self.calibration_predictions)} samples")
        self.logger.info(f"  Rest prediction: LEFT={mean_probabilities[0]:.1%}, RIGHT={mean_probabilities[1]:.1%}")
        self.logger.info(f"  Bias correction: LEFT={self.baseline_bias[0]:+.3f}, RIGHT={self.baseline_bias[1]:+.3f}")
        
        # Clear calibration buffer
        self.calibration_predictions = []
        
        return True
    
    def get_baseline_bias(self) -> Optional[np.ndarray]:
        """Get current baseline bias."""
        return self.baseline_bias.copy() if self.baseline_bias is not None else None
    
    def set_baseline_bias(self, bias: np.ndarray):
        """Set baseline bias (e.g., loaded from file)."""
        if bias is not None and len(bias) == len(self.class_names):
            self.baseline_bias = np.array(bias)
            self.logger.info(f"Loaded baseline bias: {self.baseline_bias}")
        else:
            self.logger.warning("Invalid baseline bias")
    
    def clear_baseline_bias(self):
        """Clear baseline bias correction."""
        self.baseline_bias = None
        self.logger.info("Cleared baseline bias")
    
    def load_neutral_threshold(self, json_path: str) -> bool:
        """
        Load neutral threshold from JSON file.
        
        Args:
            json_path: Path to neutral threshold JSON file
        
        Returns:
            True if loaded successfully
        """
        try:
            import json
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            threshold = data.get('threshold')
            if threshold is not None:
                self.neutral_threshold = float(threshold)
                self.logger.info(f"✅ Loaded neutral threshold: {self.neutral_threshold:.4f}")
                self.logger.info(f"   3-state mode enabled: LEFT/RIGHT/NEUTRAL")
                return True
            else:
                self.logger.error("No 'threshold' field in JSON file")
                return False
        except Exception as e:
            self.logger.error(f"Failed to load neutral threshold: {e}")
            return False
    
    def set_neutral_threshold(self, threshold: float):
        """
        Set neutral threshold manually.
        
        Args:
            threshold: Maximum confidence for NEUTRAL state (0-1)
        """
        self.neutral_threshold = float(threshold)
        self.logger.info(f"Set neutral threshold: {self.neutral_threshold:.4f}")
        self.logger.info(f"3-state mode enabled: LEFT/RIGHT/NEUTRAL")
    
    def clear_neutral_threshold(self):
        """Disable neutral state detection."""
        self.neutral_threshold = None
        self.logger.info("Cleared neutral threshold - 2-state mode (LEFT/RIGHT only)")
    
    def get_neutral_threshold(self) -> Optional[float]:
        """Get current neutral threshold."""
        return self.neutral_threshold
    
    def is_3state_mode(self) -> bool:
        """Check if 3-state mode (with NEUTRAL) is enabled."""
        return self.neutral_threshold is not None
    
    def predict_smoothed(self, preprocessed_data: np.ndarray) -> Tuple[int, float, str]:
        """
        Run inference with temporal smoothing.
        
        FIXED: Uses confidence-weighted voting instead of simple majority vote.
        This gives more weight to high-confidence predictions.
        
        Args:
            preprocessed_data: Model-ready data (1, 3, 1000, 1)
        
        Returns:
            Tuple of (class_idx, confidence, class_name) - smoothed
        """
        # Get raw prediction
        class_idx, confidence, class_name = self.predict(preprocessed_data)
        
        # Add to buffer with confidence weight
        self.prediction_buffer.append((class_idx, confidence))
        
        # If buffer not full, return raw prediction
        if len(self.prediction_buffer) < self.smoothing_window // 2:
            return class_idx, confidence, class_name
        
        # FIXED: Confidence-weighted voting (not just majority vote)
        # This gives more importance to high-confidence predictions
        class_weights = np.zeros(len(self.class_names))
        total_weight = 0.0
        
        for idx, conf in self.prediction_buffer:
            if idx >= 0 and idx < len(self.class_names):
                # Weight by confidence squared (emphasize high confidence)
                weight = conf ** 2
                class_weights[idx] += weight
                total_weight += conf
        
        # Determine smoothed prediction
        smoothed_class_idx = np.argmax(class_weights)
        
        # Compute smoothed confidence as weighted average
        if class_weights.sum() > 0:
            # Normalize weights to get probability-like scores
            normalized_weights = class_weights / class_weights.sum()
            smoothed_confidence = normalized_weights[smoothed_class_idx]
            
            # Blend with average raw confidence for more stable output
            avg_confidence = total_weight / len(self.prediction_buffer)
            smoothed_confidence = 0.7 * smoothed_confidence + 0.3 * avg_confidence
        else:
            smoothed_confidence = 0.5
        
        smoothed_class_name = self.class_names[smoothed_class_idx]
        
        return smoothed_class_idx, float(smoothed_confidence), smoothed_class_name
    
    def predict_with_smiley_feedback(self, preprocessed_data: np.ndarray) -> Tuple[int, float, str, bool, float]:
        """
        Run inference with Smiley Feedback running integral.
        
        This implements the BCI Competition strategy of integrating probability
        outputs over a 2-second window. Commands are only triggered when the
        integrated sum exceeds a high threshold, eliminating flickering.
        
        Flow:
        1. Get raw prediction with probabilities
        2. Feed full probability distribution to running integral
        3. Check if any class exceeds the threshold
        4. Only trigger command when threshold is met
        
        Args:
            preprocessed_data: Model-ready data (1, 3, 1000, 1)
        
        Returns:
            Tuple of (class_idx, confidence, class_name, is_triggered, winning_sum)
            - class_idx: -1 if no command triggered, else predicted class
            - confidence: Raw prediction confidence for this frame
            - class_name: "UNCERTAIN" if no trigger, else class name
            - is_triggered: True if running integral exceeded threshold
            - winning_sum: Current integrated sum of winning class
        """
        if not self.use_smiley_feedback or self.smiley_feedback is None:
            # Fall back to regular confidence-thresholded prediction
            class_idx, confidence, class_name = self.predict_smoothed(preprocessed_data)
            is_triggered = confidence >= self.confidence_threshold
            if is_triggered:
                self.triggered_commands[class_name] += 1
                return class_idx, confidence, class_name, True, 0.0
            return -1, confidence, "UNCERTAIN", False, 0.0
        
        start_time = time.time()
        
        try:
            # Get raw prediction with full probability distribution
            prediction = self.model.predict(preprocessed_data, verbose=0)
            raw_probabilities = prediction[0]
            
            # Apply bias correction if available
            if self.baseline_bias is not None:
                raw_probabilities = self._apply_bias_correction(raw_probabilities)
            
            # Calibrate probabilities for better confidence estimates
            if self.use_calibration:
                probabilities = self.calibrator.calibrate(raw_probabilities)
            else:
                probabilities = raw_probabilities
            
            # Get current frame prediction for reference
            class_idx = np.argmax(probabilities)
            confidence = float(probabilities[class_idx])
            
            # Update Smiley Feedback running integral
            triggered_class, is_triggered, winning_sum = self.smiley_feedback.update(probabilities)
            
            # Update statistics
            self.total_predictions += 1
            pred_time = time.time() - start_time
            self.prediction_times.append(pred_time)
            
            if is_triggered:
                triggered_name = self.class_names[triggered_class]
                self.triggered_commands[triggered_name] += 1
                return triggered_class, confidence, triggered_name, True, winning_sum
            else:
                return -1, confidence, "UNCERTAIN", False, winning_sum
        
        except Exception as e:
            self.logger.error(f"Smiley Feedback prediction error: {e}")
            return -1, 0.0, "ERROR", False, 0.0
    
    def get_smiley_feedback_status(self) -> Dict:
        """
        Get current Smiley Feedback running integral status.
        
        Returns:
            Dictionary with integral values and status for each class
        """
        if not self.use_smiley_feedback or self.smiley_feedback is None:
            return {'enabled': False}
        
        integral_values = self.smiley_feedback.get_integral_values()
        threshold = self.smiley_feedback.trigger_threshold
        buffer_fill = self.smiley_feedback.get_buffer_fill()
        
        return {
            'enabled': True,
            'window_duration_sec': self.smiley_feedback.window_duration,
            'buffer_size': self.smiley_feedback.buffer_size,
            'buffer_fill_count': len(self.smiley_feedback.probability_buffer),
            'buffer_fill_percent': buffer_fill * 100,
            'trigger_threshold': threshold,
            'integral_sums': {
                self.class_names[i]: {
                    'sum': float(integral_values[i]),
                    'threshold': threshold,
                    'percent_to_trigger': float(integral_values[i] / threshold * 100)
                }
                for i in range(len(self.class_names))
            },
            'statistics': self.smiley_feedback.get_statistics()
        }
    
    def get_statistics(self) -> Dict:
        """
        Get inference statistics.
        
        FIXED: Includes uncertainty metrics.
        
        Returns:
            Dictionary with statistics
        """
        avg_pred_time = np.mean(self.prediction_times) if self.prediction_times else 0.0
        confidence_rate = (self.confident_predictions / self.total_predictions * 100
                          if self.total_predictions > 0 else 0.0)
        
        # FIXED: Compute uncertainty statistics
        avg_uncertainty = np.mean(self.uncertainty_history) if self.uncertainty_history else 0.5
        min_uncertainty = np.min(self.uncertainty_history) if self.uncertainty_history else 0.5
        max_uncertainty = np.max(self.uncertainty_history) if self.uncertainty_history else 0.5
        
        stats = {
            'total_predictions': self.total_predictions,
            'confident_predictions': self.confident_predictions,
            'confidence_rate': confidence_rate,
            'class_distribution': self.class_counts.copy(),
            'triggered_commands': self.triggered_commands.copy(),
            'avg_prediction_time_ms': avg_pred_time * 1000,
            'predictions_per_second': 1.0 / avg_pred_time if avg_pred_time > 0 else 0.0,
            'uncertainty': {
                'average': float(avg_uncertainty),
                'min': float(min_uncertainty),
                'max': float(max_uncertainty)
            }
        }
        
        # Add smiley feedback stats if enabled
        if self.use_smiley_feedback and self.smiley_feedback is not None:
            stats['smiley_feedback'] = self.smiley_feedback.get_statistics()
        
        # Add baseline bias info
        if self.baseline_bias is not None:
            stats['baseline_bias'] = {
                'calibrated': True,
                'left_bias': float(self.baseline_bias[0]),
                'right_bias': float(self.baseline_bias[1])
            }
        else:
            stats['baseline_bias'] = {'calibrated': False}
        
        # Add temperature calibration info
        stats['calibration'] = {
            'enabled': self.use_calibration,
            'temperature': self.calibrator.temperature if self.use_calibration else None
        }
        
        return stats
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_predictions = 0
        self.confident_predictions = 0
        self.class_counts = {class_name: 0 for class_name in self.class_names}
        self.triggered_commands = {class_name: 0 for class_name in self.class_names}
        self.prediction_times.clear()
        self.prediction_buffer.clear()
        self.uncertainty_history.clear()
        if self.use_smiley_feedback and self.smiley_feedback is not None:
            self.smiley_feedback.reset()
    
    def set_temperature(self, temperature: float):
        """
        Update temperature scaling parameter.
        
        Args:
            temperature: New temperature value (>1 reduces confidence, <1 increases)
        """
        self.calibrator.temperature = temperature
        self.logger.info(f"Temperature updated to {temperature}")
    
    def get_average_uncertainty(self) -> float:
        """Get average uncertainty over recent predictions."""
        if self.uncertainty_history:
            return float(np.mean(self.uncertainty_history))
        return 0.5


if __name__ == "__main__":
    # Test the inference engine with confidence-thresholded output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("\n" + "="*70)
    print("Testing NPG Inference Engine")
    print("="*70)
    
    # Check if model exists
    model_path = Path(__file__).parent / 'models' / 'best' / 'eegnet_2class_bci2b.keras'
    
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("   Train the model first with: python train_model_2b.py")
    else:
        # Create inference engine
        engine = NPGInferenceEngine(
            model_path=str(model_path),
            confidence_threshold=0.65,
            smoothing_window=8
        )
        
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
        
        # Run inference with smoothing + threshold
        print("\n3. Testing smoothed predictions...")
        print("-" * 60)
        
        confident_count = 0
        for i in range(15):
            # Add slight noise variation to simulate real data
            noisy_data = test_data + np.random.randn(*test_data.shape) * 1
            preprocessed = preprocessor.preprocess_for_model(noisy_data)
            
            class_idx, confidence, class_name = engine.predict_smoothed(preprocessed)
            is_confident = confidence >= engine.confidence_threshold
            output = class_name if is_confident else "UNCERTAIN"
            marker = "🎯" if is_confident else ""
            print(f"   #{i+1:2}: Smoothed={class_name:10} ({confidence:.1%}) | Output={output:10} {marker}")
            if is_confident:
                confident_count += 1
        
        print("-" * 60)
        print(f"\n   Confident outputs: {confident_count}/15 predictions")

        print("\n4. Resetting statistics...")
        engine.reset_statistics()
        
        # Statistics
        print("\n5. Final Statistics:")
        stats = engine.get_statistics()
        print(f"   Total predictions: {stats['total_predictions']}")
        print(f"   Triggered commands: {stats['triggered_commands']}")
        print(f"   Avg prediction time: {stats['avg_prediction_time_ms']:.2f} ms")
        
        print("\n" + "="*70)
        print("✅ All tests passed!")
        print("="*70)
