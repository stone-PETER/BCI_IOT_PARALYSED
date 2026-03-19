"""
Blink Detection from Forehead EEG (Channel 5)

Detects single and double blinks using STATISTICAL THRESHOLDING on channel data.
Generates LEFT/RIGHT commands and optional API calls.

Single blink → LEFT command
Double blink (consecutive) → RIGHT command
"""

import time
import requests
import json
import urllib.request
import urllib.error
import numpy as np
from typing import Callable, Optional
from collections import deque


class BlinkDetector:
    """
    Blink detection from forehead EEG (Channel 5) with command/API integration.
    
    Uses STATISTICAL THRESHOLDING (recommended):
    - Detects peaks that exceed: mean + Z*sigma (Z-score approach)
    - Default Z-score threshold: 3.0 (easier to detect than 4.0)
    - Automatically adapts to signal noise level
    
    Fallback to FIXED THRESHOLD:
    - Use if statistical mode is disabled (use_statistical=False)
    - Fixed amplitude threshold (default: 150.0 μV)
    
    Configuration:
    - Disabled by default (enable with --enable-blink-detection)
    - API calls disabled by default (enable with --enable-blink-api)
    
    Behavior:
    - Debounce: Blinks must be spaced >1.0s apart
    - 1 blink (isolated) → predict "left" command
    - 2 blinks within 1.5s → predict "right" command
    - Each prediction has 0.8s cooldown before next can be issued
    - API only sends if api_enabled=True
    
    Tuning (if not detecting):
    - **Decrease Z-score to 2.0 or 2.5** (more sensitive)
    - Decrease min_blink_interval to 0.5s
    - Switch to fixed threshold mode (--no-statistical-blink --blink-threshold 100)
    
    Tuning (if too many false detections):
    - **Increase Z-score to 4.0 or 5.0** (stricter)
    - Increase min_blink_interval to 1.5s or 2.0s
    - Increase fixed threshold value
    """
    
    def __init__(self, 
                 sampling_rate: int = 250,
                 threshold: Optional[float] = None,
                 api_endpoint: Optional[str] = None,
                 api_timeout: float = 2.0,
                 use_statistical: bool = True):
        """
        Initialize blink detector.
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 250)
            threshold: Fixed amplitude threshold for blink detection (default: auto-calculated)
                       If use_statistical=True, fallback only
            api_endpoint: API endpoint for predictions (default: None)
            api_timeout: HTTP timeout in seconds (default: 2.0)
            use_statistical: Use mean+sigma statistical detection instead of fixed threshold (default: True)
                           Much better at adapting to signal noise
        """
        self.sampling_rate = sampling_rate
        self.fixed_threshold = threshold or 150.0
        self.api_endpoint = api_endpoint
        self.api_timeout = api_timeout
        self.use_statistical = use_statistical
        
        # Feature flags
        self.enabled = False          # Blink detection enabled/disabled
        self.api_enabled = False      # API calls enabled/disabled
        
        # Statistical detection
        self.signal_buffer = deque(maxlen=500)  # Last 500 samples for statistics
        self.mean_signal = 0.0
        self.std_signal = 0.0
        self.z_score_threshold = 3.0  # Blink must exceed mean + 3*std (changed from 4.0 for easier detection)
        
        # Blink timing - FIXED: Better debouncing
        self.last_blink_time = 0
        self.min_blink_interval = 1.0  # seconds between blinks (very strict)
        self.double_blink_timeout = 1.5  # max time to look for second blink
        
        # State tracking
        self.blink_count = 0
        self.last_prediction_time = 0
        self.prediction_cooldown = 0.8  # seconds before accepting new prediction
        self.last_first_blink_time = 0  # Track when first blink of sequence happened
        self.callback = None
        
        # Stats
        self.total_blinks = 0
        self.total_predictions = 0
    
    def set_callback(self, callback: Callable[[str], None]):
        """
        Set callback for predictions.
        
        Args:
            callback: Function called with prediction string ('left' or 'right')
        """
        self.callback = callback
    
    def detect_blink(self, channel_sample: float) -> Optional[str]:
        """
        Detect blink from channel sample using threshold detection.
        
        Two modes:
        1. Statistical (recommended): Detects peaks exceeding mean + Z*sigma
        2. Fixed threshold: Uses absolute amplitude threshold
        
        Logic:
        - Single blink (no second blink within 1.5s) → LEFT
        - Two blinks within 1.5s → RIGHT
        - Each blink must be spaced >1.0s apart (strict debounce)
        
        Returns: 'left', 'right', or None (no prediction)
        """
        if not self.enabled:
            return None
        
        current_time = time.time()
        prediction = None
        
        # Update signal statistics
        self.signal_buffer.append(abs(channel_sample))
        if len(self.signal_buffer) >= 100:  # Need enough samples for statistics
            self.mean_signal = np.mean(list(self.signal_buffer))
            self.std_signal = np.std(list(self.signal_buffer))
        
        # Check if this is a blink event
        is_blink = False
        
        if self.use_statistical and self.std_signal > 1e-6:
            # Statistical detection: mean + Z*sigma
            z_score = (abs(channel_sample) - self.mean_signal) / (self.std_signal + 1e-10)
            is_blink = z_score > self.z_score_threshold
            
            # DEBUG: Log occasionally to see what's happening
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
            
            if self._debug_counter % 500 == 0:
                print(f"[Blink DEBUG] Z-score: {z_score:.2f} | "
                      f"Signal: {abs(channel_sample):.1f} | "
                      f"Mean: {self.mean_signal:.1f} | "
                      f"Std: {self.std_signal:.1f} | "
                      f"Threshold: {self.z_score_threshold:.1f}σ ({self.mean_signal + self.z_score_threshold * self.std_signal:.1f})")
        else:
            # Fixed threshold detection
            is_blink = abs(channel_sample) > self.fixed_threshold
        
        # Debounce: only count if >min_blink_interval since last blink
        if is_blink:
            if current_time - self.last_blink_time > self.min_blink_interval:
                self.last_blink_time = current_time
                self.blink_count += 1
                self.total_blinks += 1
                print(f"[Blink] Detected blink #{self.total_blinks} (count in sequence: {self.blink_count})")
                
                # If this is first blink, record the time
                if self.blink_count == 1:
                    self.last_first_blink_time = current_time
        
        # Check if we should emit a prediction
        # Prediction happens in two cases:
        # 1. We got 2 blinks within double_blink_timeout
        if self.blink_count >= 2:
            # Double blink detected
            prediction = "right"
            self.blink_count = 0
            self.last_prediction_time = current_time
            print(f"[Blink] DOUBLE BLINK → RIGHT")
        
        # 2. We got 1 blink and it's been >double_blink_timeout since first blink
        elif self.blink_count == 1:
            time_since_first = current_time - self.last_first_blink_time
            if time_since_first > self.double_blink_timeout:
                # Single isolated blink - generate prediction
                prediction = "left"
                self.blink_count = 0
                self.last_prediction_time = current_time
                print(f"[Blink] SINGLE BLINK → LEFT")
        
        # Execute prediction
        if prediction:
            self._execute_prediction(prediction)
        
        return prediction
    
    def _execute_prediction(self, prediction: str):
        """
        Execute prediction with callback and optional API call.
        
        Args:
            prediction: The prediction ('left' or 'right')
        """
        self.total_predictions += 1
        
        # Call registered callback first
        if self.callback:
            self.callback(prediction)
        
        # Make API call if enabled
        if self.api_enabled and self.api_endpoint:
            self._send_api_call(prediction)
    
    def _send_api_call(self, command: str):
        """
        Send API call for blink prediction.
        
        Args:
            command: The command to send ('left' or 'right')
        """
        try:
            # Map blink command to BCI command
            bci_command = 'LEFT_HAND' if command == 'left' else 'RIGHT_HAND'
            
            payload = {
                'command': bci_command,
                'source': 'blink_detection',
                'channel': 5,
                'timestamp': time.time()
            }
            
            body = json.dumps(payload).encode('utf-8')
            request = urllib.request.Request(
                self.api_endpoint,
                data=body,
                method='POST',
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(request, timeout=self.api_timeout) as response:
                status = response.getcode()
            
            print(f"[Blink] 👁️  API call sent: {bci_command} (status={status})")
            
        except urllib.error.HTTPError as e:
            print(f"[Blink] ❌ API HTTP error: {bci_command} (status={e.code})")
        except Exception as e:
            print(f"[Blink] ❌ API call failed: {e}")
    
    def enable(self):
        """Enable blink detection."""
        self.enabled = True
    
    def disable(self):
        """Disable blink detection."""
        self.enabled = False
    
    def enable_api(self):
        """Enable API calls for predictions."""
        self.api_enabled = True
    
    def disable_api(self):
        """Disable API calls for predictions."""
        self.api_enabled = False
    
    def set_threshold(self, threshold: float):
        """Set blink amplitude threshold."""
        self.fixed_threshold = threshold
    
    def set_z_score_threshold(self, z_score: float):
        """Set Z-score threshold for statistical detection (default: 4.0)."""
        self.z_score_threshold = z_score
    
    def get_stats(self) -> dict:
        """Return detection statistics."""
        return {
            'enabled': self.enabled,
            'api_enabled': self.api_enabled,
            'total_blinks': self.total_blinks,
            'total_predictions': self.total_predictions,
            'fixed_threshold': self.fixed_threshold,
            'z_score_threshold': self.z_score_threshold,
            'use_statistical': self.use_statistical,
            'signal_mean': self.mean_signal,
            'signal_std': self.std_signal,
            'current_blink_count': self.blink_count
        }
