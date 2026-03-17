"""
NPG Lite Real-time BCI System
Complete integration: NPG Lite → Preprocessing → Inference → Commands

Real-time motor imagery classification for paralysis assistance

FIXES APPLIED:
- Correct window buffer size (2000 samples @ 500 Hz = 4 seconds)
- Consistent sampling rate handling (500 Hz input, 250 Hz after preprocessing)
- Added baseline calibration routine
- Synced parameters across all components
- Added uncertainty monitoring
"""

import numpy as np
import logging
import time
import argparse
import sys
import threading
import json
import urllib.request
import urllib.error
from typing import Optional
from pathlib import Path
from collections import deque

from npg_lite_adapter import NPGLiteAdapter, NPGLiteSimulator, NPGLiteDirectSerial
from npg_preprocessor import NPGPreprocessor, SlidingWindowBuffer
from npg_inference import NPGInferenceEngine


# === FIXED: Consistent parameters across all components ===
INPUT_SAMPLING_RATE = 500   # NPG Lite via Chords-Python
OUTPUT_SAMPLING_RATE = 250  # Model trained at this rate
EPOCH_DURATION = 4.0        # 4 seconds per classification
N_CHANNELS = 3              # C3, Cz, C4

# Calculated values
INPUT_EPOCH_SAMPLES = int(EPOCH_DURATION * INPUT_SAMPLING_RATE)   # 2000 @ 500 Hz
OUTPUT_EPOCH_SAMPLES = int(EPOCH_DURATION * OUTPUT_SAMPLING_RATE) # 1000 @ 250 Hz


class NPGRealtimeBCI:
    """
    Complete real-time BCI system for NPG Lite.
    
    Pipeline:
    NPG Lite (500 Hz, 3 ch) → Anti-alias → Resample (250 Hz) → 
    Notch → Small Laplacian → Bandpass (8-30 Hz) → Z-score → Model → Command
    
    FIXES:
    - Correct window sizes matching input rate
    - Consistent parameter handling
    - Baseline calibration support
    - Uncertainty tracking
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.65,
                 smoothing_window: int = 8,
                 window_overlap: float = 0.5,
                 simulate: bool = False,
                 use_accumulator: bool = True,
                 accumulator_threshold: float = 2.0,
                 accumulator_decay: float = 0.15,
                 neutral_zone: tuple = (0.45, 0.55),
                 temperature: float = 1.5,
                 use_calibration: bool = True,
                 use_smiley_feedback: bool = False,
                 smiley_window_duration: float = 2.0,
                 smiley_threshold: float = 3.5,
                 smiley_prediction_rate: float = 2.2,
                 neutral_threshold: float = None,
                 api_base_url: Optional[str] = None,
                 api_light_path: str = '/light',
                 api_fan_path: str = '/fan',
                 api_timeout: float = 2.0,
                 api_confidence_threshold: float = 0.75,
                 api_neutral_hold: float = 0.8,
                 api_cooldown: float = 2.5,
                 api_rearm_timeout: float = 4.0):
        """
        Initialize real-time BCI system.
        
        Args:
            model_path: Path to trained model
            confidence_threshold: Minimum confidence for command execution (default: 0.65)
            smoothing_window: Number of predictions to smooth (default: 8)
            window_overlap: Overlap ratio for sliding windows (0-1)
            simulate: Use simulator instead of real device
            use_accumulator: Use Leaky Accumulator for stable commands (default: True)
            accumulator_threshold: Bucket level to trigger command (default: 2.0)
            accumulator_decay: Decay rate per update (default: 0.15)
            neutral_zone: Confidence range treated as uncertain (default: 0.45-0.55)
            temperature: Temperature for calibrated confidence (default: 1.5)
            use_calibration: Whether to use temperature-scaled confidence
            use_smiley_feedback: Use Smiley Feedback running integral (default: False)
            smiley_window_duration: Integration window in seconds (default: 2.0)
            smiley_threshold: Sum threshold to trigger (default: 3.5)
            smiley_prediction_rate: Expected prediction rate in Hz (default: 2.2)
            neutral_threshold: Threshold for NEUTRAL state (None = 2-state, float = 3-state)
        """
        self.simulate = simulate
        self.confidence_threshold = confidence_threshold
        self.use_accumulator = use_accumulator
        self.use_smiley_feedback = use_smiley_feedback
        self.is_running = False
        self.neutral_threshold = neutral_threshold
        self.api_base_url = api_base_url.rstrip('/') if api_base_url else None
        self.api_light_path = api_light_path
        self.api_fan_path = api_fan_path
        self.api_timeout = api_timeout
        self.api_confidence_threshold = api_confidence_threshold
        self.api_neutral_hold = api_neutral_hold
        self.api_cooldown = api_cooldown
        self.api_rearm_timeout = api_rearm_timeout
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("="*70)
        self.logger.info("NPG Lite Real-time BCI System (FIXED)")
        self.logger.info("="*70)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Input rate: {INPUT_SAMPLING_RATE} Hz (NPG Lite via Chords)")
        self.logger.info(f"  Output rate: {OUTPUT_SAMPLING_RATE} Hz (model)")
        self.logger.info(f"  Epoch duration: {EPOCH_DURATION}s")
        self.logger.info(f"  Input epoch: {INPUT_EPOCH_SAMPLES} samples")
        self.logger.info(f"  Output epoch: {OUTPUT_EPOCH_SAMPLES} samples")
        if self.api_base_url:
            self.logger.info("  API mode: ENABLED")
            self.logger.info(f"  API base URL: {self.api_base_url}")
            self.logger.info(f"  API cooldown: {self.api_cooldown:.1f}s")
            self.logger.info(f"  API neutral hold: {self.api_neutral_hold:.1f}s")
        else:
            self.logger.info("  API mode: DISABLED")
        
        # Initialize components
        self.logger.info("\n1. Initializing components...")
        
        # NPG adapter
        if simulate:
            self.logger.info("   Using SIMULATOR mode")
            self.adapter = NPGLiteSimulator(sampling_rate=INPUT_SAMPLING_RATE)
        else:
            self.logger.info("   Using HARDWARE mode")
            self.adapter = NPGLiteAdapter(sampling_rate=INPUT_SAMPLING_RATE)
        
        # FIXED: Preprocessor with correct parameters
        self.preprocessor = NPGPreprocessor(
            input_rate=INPUT_SAMPLING_RATE,
            output_rate=OUTPUT_SAMPLING_RATE,
            epoch_duration=EPOCH_DURATION,
            use_car=False,  # Model trained on raw data, no CAR
            apply_bandpass=False,  # Model trained on raw data
            apply_zscore=False,  # Model trained on raw data
            realtime_mode=True
        )
        self.logger.info("   ✅ Preprocessor ready (raw data preprocessing matching training)")
        
        # Inference engine with calibration
        self.inference = NPGInferenceEngine(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            smoothing_window=smoothing_window,
            use_accumulator=use_accumulator,
            accumulator_threshold=accumulator_threshold,
            accumulator_decay=accumulator_decay,
            neutral_zone=neutral_zone,
            temperature=temperature,
            use_calibration=use_calibration,
            use_smiley_feedback=use_smiley_feedback,
            smiley_window_duration=smiley_window_duration,
            smiley_threshold=smiley_threshold,
            smiley_prediction_rate=smiley_prediction_rate,
            neutral_threshold=neutral_threshold  # Add neutral threshold for 3-state
        )
        mode_str = "Smiley Feedback" if use_smiley_feedback else "temperature-calibrated"
        state_mode = "3-state (LEFT/RIGHT/NEUTRAL)" if neutral_threshold else "2-state (LEFT/RIGHT)"
        self.logger.info(f"   ✅ Inference engine ready ({mode_str} confidence, {state_mode})")
        
        # FIXED: Sliding window buffer with correct size for input rate
        stride = int(INPUT_EPOCH_SAMPLES * (1 - window_overlap))
        self.window_buffer = SlidingWindowBuffer(
            window_size=INPUT_EPOCH_SAMPLES,  # 2000 samples @ 500 Hz
            overlap=window_overlap,
            sampling_rate=INPUT_SAMPLING_RATE
        )
        self.logger.info(f"   ✅ Window buffer ready ({INPUT_EPOCH_SAMPLES} samples, {window_overlap:.0%} overlap)")
        
        # Command tracking
        self.last_command = None
        self.last_command_time = 0
        self.command_counts = {'LEFT_HAND': 0, 'RIGHT_HAND': 0, 'NEUTRAL': 0, 'UNCERTAIN': 0}
        self.device_state = {'light': False, 'fan': False}
        self.api_armed = {'LEFT_HAND': True, 'RIGHT_HAND': True}
        self.api_last_sent_time = {'LEFT_HAND': 0.0, 'RIGHT_HAND': 0.0}
        self.api_blocked_since = {'LEFT_HAND': None, 'RIGHT_HAND': None}
        self.api_neutral_start = None
        self.api_prev_command = 'NEUTRAL'
        
        # Performance metrics
        self.processing_times = deque(maxlen=100)
        self.total_epochs_processed = 0
        self.start_time = None
        
        # Try to load saved baseline bias
        self._load_baseline_bias()
        
        self.logger.info("="*70)
    
    def connect(self, device_id: Optional[str] = None):
        """
        Connect to NPG Lite device.
        
        Args:
            device_id: Serial port (e.g., 'COM3'). Auto-detect if None
        """
        if self.simulate:
            self.logger.info("Simulated connection - no hardware needed")
            return
        
        self.logger.info("Connecting to NPG Lite...")
        success = self.adapter.connect(device_id)
        
        if not success:
            raise RuntimeError("Failed to connect to NPG Lite")
        
        # Check signal quality
        self.logger.info("Checking signal quality...")
        quality = self.adapter.check_signal_quality()
        
        for ch, q in quality.items():
            status = "✅" if q > 0.5 else "⚠️"
            self.logger.info(f"  {status} {ch}: {q:.2f}")
        
        self.logger.info("✅ Connected successfully")
    
    def start(self):
        """Start real-time BCI processing."""
        if self.is_running:
            self.logger.warning("BCI already running")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        self.logger.info("\n" + "="*70)
        self.logger.info("Starting real-time BCI processing...")
        self.logger.info("="*70)
        self.logger.info("Commands: LEFT_HAND, RIGHT_HAND")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold:.0%}")
        self.logger.info("="*70 + "\n")
        
        # Start streaming
        self.adapter.start_streaming()
        
        try:
            self._processing_loop()
        except KeyboardInterrupt:
            self.logger.info("\n\nKeyboard interrupt detected")
        finally:
            self.stop()
    
    def _processing_loop(self):
        """Main processing loop."""
        # FIXED: Use correct sample counts for 500 Hz input
        warmup_samples = INPUT_EPOCH_SAMPLES  # 2000 samples = 4 seconds @ 500 Hz
        chunk_size = int(INPUT_SAMPLING_RATE * 0.25)  # 125 samples = 0.25 seconds
        
        self.logger.info(f"Warming up... collecting {warmup_samples} samples ({EPOCH_DURATION}s)")
        
        # Initial warmup phase
        warmup_complete = False
        loop_count = 0
        warmup_start = time.time()
        
        while self.is_running:
            try:
                loop_count += 1
                
                # Get latest data from adapter
                if not warmup_complete:
                    # First time: get full warmup
                    data = self.adapter.get_latest_data(n_samples=warmup_samples)
                    if data is not None and len(data) >= warmup_samples:
                        warmup_complete = True
                        self.logger.info(f"✅ Warmup complete! Starting continuous processing...")
                        # Reset filter states for clean start
                        self.preprocessor.reset_filters()
                    else:
                        # Check for timeout and show progress
                        elapsed = time.time() - warmup_start
                        if elapsed > 10 and loop_count % 100 == 0:
                            # Get buffer size safely
                            buffer_size = 0
                            if hasattr(self.adapter, 'buffer_lock') and hasattr(self.adapter, 'data_buffer'):
                                with self.adapter.buffer_lock:
                                    buffer_size = len(self.adapter.data_buffer)
                            self.logger.warning(f"Still warming up... ({elapsed:.1f}s elapsed, {buffer_size}/{warmup_samples} samples)")
                        
                        time.sleep(0.1)  # Wait for more data
                        continue
                else:
                    # After warmup: get smaller chunks
                    data = self.adapter.get_latest_data(n_samples=chunk_size)
                
                if data is None or len(data) < 10:  # At least 10 samples
                    if loop_count % 50 == 0:  # Log every 5 seconds
                        buffer_size = len(self.adapter.data_buffer) if hasattr(self.adapter, 'data_buffer') else 0
                        self.logger.debug(f"Waiting for data... buffer size: {buffer_size}")
                    time.sleep(0.1)
                    continue
                
                # Verify channel count
                if data.shape[1] != N_CHANNELS:
                    self.logger.warning(f"Expected {N_CHANNELS} channels, got {data.shape[1]}")
                    data = data[:, :N_CHANNELS]  # Trim to expected channels
                
                # Add to window buffer
                self.window_buffer.add_samples(data)
                
                # Extract windows
                windows = self.window_buffer.get_windows()
                
                if not windows:
                    if loop_count % 50 == 0:  # Log every 5 seconds
                        self.logger.debug(f"No windows ready. Buffer size: {self.window_buffer.get_buffer_size()}")
                    time.sleep(0.1)
                    continue
                
                # Process each window
                for window in windows:
                    self._process_epoch(window)
                
                # Small delay to prevent CPU overload
                time.sleep(0.05)
            
            except Exception as e:
                self.logger.error(f"Processing error: {e}", exc_info=True)
                time.sleep(0.5)
    
    def _process_epoch(self, epoch_data: np.ndarray):
        """
        Process a single 4-second epoch.
        
        Args:
            epoch_data: Raw data (2000 samples @ 500 Hz, 3 channels)
        """
        start_time = time.time()
        
        try:
            # Verify input shape
            expected_samples = INPUT_EPOCH_SAMPLES
            if epoch_data.shape[0] < expected_samples * 0.9:  # Allow 10% tolerance
                self.logger.warning(f"Epoch too short: {epoch_data.shape[0]} samples, expected ~{expected_samples}")
                return
            
            # Preprocess (handles resampling, filtering, normalization)
            preprocessed = self.preprocessor.preprocess_for_model(epoch_data)
            
            # Verify output shape
            if preprocessed.shape != (1, N_CHANNELS, OUTPUT_EPOCH_SAMPLES, 1):
                self.logger.warning(f"Preprocessed shape mismatch: {preprocessed.shape}")
            
            # Inference with accumulator or smiley feedback
            if self.use_smiley_feedback:
                # Use Smiley Feedback running integral
                class_idx, confidence, class_name, is_triggered, winning_sum = \
                    self.inference.predict_with_smiley_feedback(preprocessed)
                
                if is_triggered:
                    command = class_name
                    self.command_counts[command] += 1
                else:
                    command = "UNCERTAIN"
                    self.command_counts["UNCERTAIN"] += 1
                    
            elif self.use_accumulator:
                # Use Leaky Accumulator
                class_idx, confidence, class_name, is_triggered = \
                    self.inference.predict_with_accumulator(preprocessed)
                
                # Only count as command if accumulator triggered
                if is_triggered:
                    command = class_name
                    self.command_counts[command] += 1
                else:
                    command = "UNCERTAIN"
                    self.command_counts["UNCERTAIN"] += 1
            else:
                # Fallback: simple smoothed prediction
                class_idx, confidence, class_name = self.inference.predict_smoothed(preprocessed)
                
                if confidence >= self.confidence_threshold:
                    command = class_name
                    self.command_counts[command] += 1
                else:
                    command = "UNCERTAIN"
                    self.command_counts["UNCERTAIN"] += 1
            
            # Check if command changed (only for actual triggers, not UNCERTAIN)
            command_changed = (command != self.last_command and 
                             command != "UNCERTAIN")

            # API dispatch uses edge-trigger + neutral re-arm + cooldown
            self._handle_api_command(command, confidence)
            
            # Update tracking
            if command != "UNCERTAIN":
                self.last_command = command
                self.last_command_time = time.time()
            self.total_epochs_processed += 1
            
            # Track processing time
            proc_time = time.time() - start_time
            self.processing_times.append(proc_time)
            
            # Log result
            if command_changed:
                self._log_command(command, confidence, proc_time)
            elif self.total_epochs_processed % 10 == 0:
                self._log_status()
        
        except Exception as e:
            self.logger.error(f"Epoch processing error: {e}", exc_info=True)

    def _build_api_url(self, device: str) -> Optional[str]:
        """Build API URL for mapped IoT device endpoint."""
        if not self.api_base_url:
            return None

        path = self.api_light_path if device == 'light' else self.api_fan_path
        if not path.startswith('/'):
            path = f'/{path}'
        return f"{self.api_base_url}{path}"

    def _post_toggle(self, device: str, state: bool, command: str, confidence: float):
        """Send POST request for a device toggle event."""
        url = self._build_api_url(device)
        if not url:
            return

        payload = {
            'device': device,
            'on': state,
            'source': 'bci',
            'command': command,
            'confidence': float(confidence),
            'timestamp': time.time()
        }

        body = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            url,
            data=body,
            method='POST',
            headers={'Content-Type': 'application/json'}
        )

        try:
            with urllib.request.urlopen(request, timeout=self.api_timeout) as response:
                status = response.getcode()
            self.logger.info(f"🌐 API CALL OK | {device}={'ON' if state else 'OFF'} | status={status} | {url}")
        except urllib.error.HTTPError as e:
            self.logger.error(f"🌐 API CALL HTTP ERROR | {device}={'ON' if state else 'OFF'} | status={e.code} | {url}")
        except Exception as e:
            self.logger.error(f"🌐 API CALL FAILED | {device}={'ON' if state else 'OFF'} | {url} | error={e}")

    def _toggle_device_for_command(self, command: str, confidence: float):
        """Toggle mapped IoT device for LEFT/RIGHT commands."""
        if command == 'LEFT_HAND':
            self.device_state['light'] = not self.device_state['light']
            self._post_toggle('light', self.device_state['light'], command, confidence)
            self.logger.info(f"💡 TOGGLE | LEFT_HAND -> light={'ON' if self.device_state['light'] else 'OFF'}")
        elif command == 'RIGHT_HAND':
            self.device_state['fan'] = not self.device_state['fan']
            self._post_toggle('fan', self.device_state['fan'], command, confidence)
            self.logger.info(f"🌀 TOGGLE | RIGHT_HAND -> fan={'ON' if self.device_state['fan'] else 'OFF'}")

    def _handle_api_command(self, command: str, confidence: float):
        """Prevent repeated toggles using edge trigger, neutral re-arm, and cooldown."""
        if not self.api_base_url:
            return

        now = time.time()

        if command in ('LEFT_HAND', 'RIGHT_HAND') and confidence < self.api_confidence_threshold:
            self.logger.info(
                f"🚫 API BLOCK | low confidence | cmd={command} conf={confidence:.1%} < {self.api_confidence_threshold:.1%}"
            )
            self.api_prev_command = command
            return

        if command == 'NEUTRAL':
            if self.api_neutral_start is None:
                self.api_neutral_start = now

            neutral_hold = now - self.api_neutral_start
            if neutral_hold >= self.api_neutral_hold:
                self.api_armed['LEFT_HAND'] = True
                self.api_armed['RIGHT_HAND'] = True
                self.api_blocked_since['LEFT_HAND'] = None
                self.api_blocked_since['RIGHT_HAND'] = None
                self.logger.info(f"✅ API RE-ARM | neutral hold={neutral_hold:.2f}s")

            self.api_prev_command = 'NEUTRAL'
            return

        # Any non-neutral command breaks neutral window immediately
        self.api_neutral_start = None

        if command not in ('LEFT_HAND', 'RIGHT_HAND'):
            self.api_prev_command = command
            return

        is_edge = self.api_prev_command != command
        if not is_edge:
            self.logger.info(f"🚫 API BLOCK | no-edge repeat | cmd={command}")
            self.api_prev_command = command
            return

        since_last = now - self.api_last_sent_time[command]
        if since_last < self.api_cooldown:
            self.logger.info(f"🚫 API BLOCK | cooldown | cmd={command} remaining={self.api_cooldown - since_last:.2f}s")
            self.api_prev_command = command
            return

        if not self.api_armed[command]:
            if self.api_blocked_since[command] is None:
                self.api_blocked_since[command] = now

            blocked_for = now - self.api_blocked_since[command]
            if blocked_for >= self.api_rearm_timeout:
                self.api_armed[command] = True
                self.api_blocked_since[command] = None
                self.logger.info(f"⏱️ API TIMEOUT RE-ARM | cmd={command} blocked_for={blocked_for:.2f}s")
            else:
                self.logger.info(
                    f"🚫 API BLOCK | not armed | cmd={command} (need NEUTRAL {self.api_neutral_hold:.1f}s or timeout {self.api_rearm_timeout:.1f}s)"
                )
                self.api_prev_command = command
                return

        self._toggle_device_for_command(command, confidence)
        self.api_last_sent_time[command] = now
        self.api_armed[command] = False
        self.api_blocked_since[command] = now
        self.api_prev_command = command
    
    def _log_command(self, command: str, confidence: float, proc_time: float):
        """Log a new command."""
        if command == "LEFT_HAND":
            icon = "👈"
        elif command == "RIGHT_HAND":
            icon = "👉"
        elif command == "NEUTRAL":
            icon = "⚪"
        else:
            icon = "❓"
        
        # Add accumulator or smiley feedback info
        feedback_info = ""
        if self.use_smiley_feedback:
            status = self.inference.get_smiley_feedback_status()
            if status.get('enabled'):
                sums = status['integral_sums']
                left_pct = sums['LEFT_HAND']['percent_to_trigger']
                right_pct = sums['RIGHT_HAND']['percent_to_trigger']
                feedback_info = f" | Integral: L={left_pct:3.0f}% R={right_pct:3.0f}%"
        elif self.use_accumulator:
            status = self.inference.get_accumulator_status()
            if status.get('enabled'):
                left_pct = status['buckets']['LEFT_HAND']['percent_full']
                right_pct = status['buckets']['RIGHT_HAND']['percent_full']
                feedback_info = f" | Buckets: L={left_pct:3.0f}% R={right_pct:3.0f}%"
        
        self.logger.info(
            f"{icon} {command:12} | "
            f"Confidence: {confidence:5.1%} | "
            f"Processing: {proc_time*1000:5.1f}ms{feedback_info}"
        )
    
    def _log_status(self):
        """Log periodic status update."""
        runtime = time.time() - self.start_time
        epochs_per_sec = self.total_epochs_processed / runtime
        avg_proc_time = np.mean(self.processing_times) if self.processing_times else 0
        
        # Add accumulator or smiley feedback info
        feedback_info = ""
        if self.use_smiley_feedback:
            status = self.inference.get_smiley_feedback_status()
            if status.get('enabled'):
                sums = status['integral_sums']
                left_pct = sums['LEFT_HAND']['percent_to_trigger']
                right_pct = sums['RIGHT_HAND']['percent_to_trigger']
                feedback_info = f" | Integral: L={left_pct:3.0f}% R={right_pct:3.0f}%"
        elif self.use_accumulator:
            status = self.inference.get_accumulator_status()
            if status.get('enabled'):
                left_pct = status['buckets']['LEFT_HAND']['percent_full']
                right_pct = status['buckets']['RIGHT_HAND']['percent_full']
                feedback_info = f" | Buckets: L={left_pct:3.0f}% R={right_pct:3.0f}%"
        
        # Add uncertainty info
        avg_uncertainty = self.inference.get_average_uncertainty()
        uncertainty_info = f" | Uncertainty: {avg_uncertainty:.1%}"
        
        self.logger.info(
            f"📊 Status | "
            f"Epochs: {self.total_epochs_processed:4} | "
            f"Rate: {epochs_per_sec:4.1f}/s | "
            f"Avg time: {avg_proc_time*1000:5.1f}ms{feedback_info}{uncertainty_info}"
        )
    
    def stop(self):
        """Stop BCI processing and show statistics."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        self.logger.info("\n" + "="*70)
        self.logger.info("Stopping BCI system...")
        self.logger.info("="*70)
        
        # Stop streaming
        self.adapter.stop_streaming()
        
        # Show statistics
        self._show_statistics()
        
        self.logger.info("="*70)
        self.logger.info("BCI system stopped")
        self.logger.info("="*70)
    
    def run_calibration(self, duration_seconds: int = 60):
        """
        Run baseline calibration to correct 2-class model bias.
        User should sit still and look at a fixed point during this time.
        
        Args:
            duration_seconds: Calibration duration (default: 60s)
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("🎯 BASELINE CALIBRATION MODE")
        self.logger.info("="*70)
        self.logger.info("Instructions:")
        self.logger.info("  • Sit completely still")
        self.logger.info("  • Look at a fixed point")
        self.logger.info("  • Do NOT imagine any movement")
        self.logger.info(f"  • Calibration will run for {duration_seconds} seconds")
        self.logger.info("="*70)
        
        input("\nPress ENTER when ready to start calibration...")
        
        # Start calibration
        self.inference.start_calibration()
        self.adapter.start_streaming()
        
        # Collect data
        start_time = time.time()
        warmup_samples = 1024
        warmup_complete = False
        samples_collected = 0
        
        self.logger.info(f"\n⏱️  Calibrating... ({duration_seconds}s)")
        self.logger.info("   Stay relaxed and still!\n")
        
        try:
            while time.time() - start_time < duration_seconds:
                # Get data
                if not warmup_complete:
                    data = self.adapter.get_latest_data(n_samples=warmup_samples)
                    if data is not None and len(data) >= warmup_samples:
                        warmup_complete = True
                    else:
                        time.sleep(0.1)
                        continue
                else:
                    data = self.adapter.get_latest_data(n_samples=128)
                
                if data is None or len(data) < 10:
                    time.sleep(0.1)
                    continue
                
                # Add to window buffer
                self.window_buffer.add_samples(data)
                
                # Process windows
                windows = self.window_buffer.get_windows()
                for window in windows:
                    # Preprocess
                    preprocessed = self.preprocessor.preprocess_for_model(window)
                    
                    # Add calibration sample
                    self.inference.add_calibration_sample(preprocessed)
                    samples_collected += 1
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    remaining = duration_seconds - elapsed
                    if samples_collected % 5 == 0:
                        self.logger.info(f"   Progress: {elapsed:.0f}s / {duration_seconds}s "
                                       f"({samples_collected} samples) - {remaining:.0f}s remaining")
                
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            self.logger.warning("\nCalibration interrupted!")
        
        finally:
            self.adapter.stop_streaming()
        
        # Finalize calibration
        self.logger.info(f"\n✅ Collected {samples_collected} samples")
        
        if self.inference.finalize_calibration():
            self.logger.info("✅ Baseline calibration successful!")
            
            # Save bias to file
            bias = self.inference.get_baseline_bias()
            self._save_baseline_bias(bias)
            
            self.logger.info("\n" + "="*70)
            self.logger.info("You can now run the BCI system with bias correction")
            self.logger.info("="*70)
            return True
        else:
            self.logger.error("❌ Calibration failed")
            return False
    
    def _save_baseline_bias(self, bias: np.ndarray):
        """Save baseline bias to JSON file."""
        import json
        bias_file = Path(__file__).parent / 'baseline_bias.json'
        
        try:
            data = {
                'bias': bias.tolist(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'class_names': self.inference.class_names
            }
            
            with open(bias_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"💾 Saved baseline bias to: {bias_file.name}")
        except Exception as e:
            self.logger.error(f"Failed to save baseline bias: {e}")
    
    def _load_baseline_bias(self) -> bool:
        """Load baseline bias from JSON file."""
        import json
        bias_file = Path(__file__).parent / 'baseline_bias.json'
        
        if not bias_file.exists():
            return False
        
        try:
            with open(bias_file, 'r') as f:
                data = json.load(f)
            
            bias = np.array(data['bias'])
            self.inference.set_baseline_bias(bias)
            
            self.logger.info(f"✅ Loaded baseline bias from: {bias_file.name}")
            self.logger.info(f"   Calibrated: {data.get('timestamp', 'unknown')}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load baseline bias: {e}")
            return False
    
    def _show_statistics(self):
        """Show comprehensive statistics."""
        runtime = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info("\n📊 Session Statistics:")
        self.logger.info(f"   Runtime: {runtime:.1f}s")
        self.logger.info(f"   Epochs processed: {self.total_epochs_processed}")
        
        if runtime > 0:
            self.logger.info(f"   Processing rate: {self.total_epochs_processed/runtime:.2f} epochs/s")
        
        # Command distribution
        total_commands = sum(self.command_counts.values())
        self.logger.info("\n   Command Distribution:")
        for cmd, count in self.command_counts.items():
            pct = (count / total_commands * 100) if total_commands > 0 else 0
            icon = "👈" if cmd == "LEFT_HAND" else ("👉" if cmd == "RIGHT_HAND" else "❓")
            self.logger.info(f"     {icon} {cmd:12}: {count:4} ({pct:5.1f}%)")
        
        # Processing performance
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            min_time = np.min(self.processing_times)
            max_time = np.max(self.processing_times)
            
            self.logger.info("\n   Processing Performance:")
            self.logger.info(f"     Avg: {avg_time*1000:.2f}ms")
            self.logger.info(f"     Min: {min_time*1000:.2f}ms")
            self.logger.info(f"     Max: {max_time*1000:.2f}ms")
        
        # Inference statistics
        inf_stats = self.inference.get_statistics()
        self.logger.info("\n   Inference Statistics:")
        self.logger.info(f"     Confident predictions: {inf_stats['confident_predictions']} "
                        f"({inf_stats['confidence_rate']:.1f}%)")
        self.logger.info(f"     Class distribution: {inf_stats['class_distribution']}")
        self.logger.info(f"     Throughput: {inf_stats['predictions_per_second']:.1f} pred/s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='NPG Lite Real-time BCI System (Upside Down Labs)')
    parser.add_argument('--user-id', type=str, default=None,
                       help='User ID to load personalized model from registry')
    parser.add_argument('--simulate', action='store_true',
                       help='Use simulator instead of real NPG Lite device')
    parser.add_argument('--direct', action='store_true',
                       help='Direct serial connection (no Chords-Python/LSL needed)')
    parser.add_argument('--port', type=str, default='COM6',
                       help='Serial port for direct connection (default: COM6)')
    parser.add_argument('--baudrate', type=int, default=230400,
                       help='Serial baud rate for direct connection (default: 230400)')
    parser.add_argument('--model', type=str,
                       help='Path to trained model (default: models/best_eegnet_2class_bci2b.keras)')
    parser.add_argument('--confidence', type=float, default=0.65,
                       help='Confidence threshold for classification (0-1, default: 0.65)')
    parser.add_argument('--smoothing', type=int, default=8,
                       help='Smoothing window size for predictions (default: 8)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap ratio (0-1, default: 0.5)')
    parser.add_argument('--no-accumulator', action='store_true',
                       help='Disable Leaky Accumulator (use simple threshold instead)')
    parser.add_argument('--acc-threshold', type=float, default=2.0,
                       help='Accumulator trigger threshold (default: 2.0)')
    parser.add_argument('--acc-decay', type=float, default=0.15,
                       help='Accumulator decay rate per update (default: 0.15)')
    parser.add_argument('--neutral-low', type=float, default=0.45,
                       help='Neutral zone lower bound (default: 0.45)')
    parser.add_argument('--neutral-high', type=float, default=0.55,
                       help='Neutral zone upper bound (default: 0.55)')
    parser.add_argument('--smiley-feedback', action='store_true',
                       help='Use Smiley Feedback running integral (BCI Competition strategy)')
    parser.add_argument('--smiley-window', type=float, default=2.0,
                       help='Smiley Feedback integration window in seconds (default: 2.0)')
    parser.add_argument('--smiley-threshold', type=float, default=3.5,
                       help='Smiley Feedback trigger threshold for sum (default: 3.5)')
    parser.add_argument('--smiley-rate', type=float, default=2.2,
                       help='Expected prediction rate in Hz (default: 2.2)')
    parser.add_argument('--calibrate', action='store_true',
                       help='Run baseline calibration (60s rest state recording)')
    parser.add_argument('--calibrate-duration', type=int, default=60,
                       help='Calibration duration in seconds (default: 60)')
    parser.add_argument('--no-bias-correction', action='store_true',
                       help='Disable baseline bias correction (even if calibrated)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--api-base-url', type=str, default=None,
                       help='Base URL for IoT API (example: http://127.0.0.1:8000/devices)')
    parser.add_argument('--api-light-path', type=str, default='/light',
                       help='Endpoint path for light toggle (default: /light)')
    parser.add_argument('--api-fan-path', type=str, default='/fan',
                       help='Endpoint path for fan toggle (default: /fan)')
    parser.add_argument('--api-timeout', type=float, default=2.0,
                       help='HTTP timeout in seconds (default: 2.0)')
    parser.add_argument('--api-conf-threshold', type=float, default=0.75,
                       help='Minimum confidence for API calls (default: 0.75)')
    parser.add_argument('--api-neutral-hold', type=float, default=0.8,
                       help='Neutral hold duration to re-arm commands (default: 0.8s)')
    parser.add_argument('--api-cooldown', type=float, default=2.5,
                       help='Per-command cooldown in seconds (default: 2.5s)')
    parser.add_argument('--api-rearm-timeout', type=float, default=4.0,
                       help='Timeout fallback to re-arm same command (default: 4.0s)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Load model from registry if user_id provided
    neutral_threshold = None
    if args.user_id:
        from model_factory import ModelFactory
        try:
            model_info = ModelFactory.load_from_registry(args.user_id)
            model_path = Path(model_info['path'])
            neutral_threshold = model_info.get('neutral_threshold')
            
            print(f"\n✅ Loading personalized model for user: {args.user_id}")
            print(f"   Model: {model_path}")
            if neutral_threshold:
                print(f"   Neutral threshold: {neutral_threshold:.4f}")
                print(f"   3-state mode: LEFT/RIGHT/NEUTRAL")
            else:
                print(f"   2-state mode: LEFT/RIGHT only")
        except Exception as e:
            print(f"⚠️  Failed to load personalized model: {e}")
            print(f"   Falling back to base model")
            args.user_id = None
    
    # Check model
    if args.user_id:
        # Already loaded from registry
        pass
    elif args.model:
        model_path = Path(args.model)
    else:
        # Try to find the latest timestamped model
        models_dir = Path(__file__).parent / 'models'
        best_models = list(models_dir.glob('best_eegnet_2class_bci2b_*.keras'))
        
        if best_models:
            # Sort by filename (timestamp) and get the latest
            latest_model = sorted(best_models, key=lambda x: x.name)[-1]
            model_path = latest_model
            print(f"-> Using latest model: {latest_model.name}")
        else:
            # Fall back to best model
            model_path = models_dir / 'best' / 'eegnet_2class_bci2b.keras'
    
    if not model_path.exists():
        print(f"❌ Error: Model not found: {model_path}")
        print("   Train the model first with: python train_model_2b.py")
        sys.exit(1)
    
    # Create neutral zone tuple
    neutral_zone = (args.neutral_low, args.neutral_high)
    use_accumulator = not args.no_accumulator
    
    # Create BCI system based on mode
    if args.simulate:
        # Simulator mode
        bci = NPGRealtimeBCI(
            model_path=str(model_path),
            confidence_threshold=args.confidence,
            smoothing_window=args.smoothing,
            window_overlap=args.overlap,
            simulate=True,
            use_accumulator=use_accumulator,
            accumulator_threshold=args.acc_threshold,
            accumulator_decay=args.acc_decay,
            neutral_zone=neutral_zone,
            use_smiley_feedback=args.smiley_feedback,
            smiley_window_duration=args.smiley_window,
            smiley_threshold=args.smiley_threshold,
            smiley_prediction_rate=args.smiley_rate,
            neutral_threshold=neutral_threshold,
            api_base_url=args.api_base_url,
            api_light_path=args.api_light_path,
            api_fan_path=args.api_fan_path,
            api_timeout=args.api_timeout,
            api_confidence_threshold=args.api_conf_threshold,
            api_neutral_hold=args.api_neutral_hold,
            api_cooldown=args.api_cooldown,
            api_rearm_timeout=args.api_rearm_timeout
        )
    elif args.direct:
        # Direct serial connection mode
        bci = NPGRealtimeBCI(
            model_path=str(model_path),
            confidence_threshold=args.confidence,
            smoothing_window=args.smoothing,
            window_overlap=args.overlap,
            simulate=False,
            use_accumulator=use_accumulator,
            accumulator_threshold=args.acc_threshold,
            accumulator_decay=args.acc_decay,
            neutral_zone=neutral_zone,
            use_smiley_feedback=args.smiley_feedback,
            smiley_window_duration=args.smiley_window,
            smiley_threshold=args.smiley_threshold,
            smiley_prediction_rate=args.smiley_rate,
            neutral_threshold=neutral_threshold,
            api_base_url=args.api_base_url,
            api_light_path=args.api_light_path,
            api_fan_path=args.api_fan_path,
            api_timeout=args.api_timeout,
            api_confidence_threshold=args.api_conf_threshold,
            api_neutral_hold=args.api_neutral_hold,
            api_cooldown=args.api_cooldown,
            api_rearm_timeout=args.api_rearm_timeout
        )
        # Replace adapter with direct serial adapter
        bci.adapter = NPGLiteDirectSerial(port=args.port, baudrate=args.baudrate)
    else:
        # LSL stream mode (requires Chords-Python)
        bci = NPGRealtimeBCI(
            model_path=str(model_path),
            confidence_threshold=args.confidence,
            smoothing_window=args.smoothing,
            window_overlap=args.overlap,
            simulate=False,
            use_accumulator=use_accumulator,
            accumulator_threshold=args.acc_threshold,
            accumulator_decay=args.acc_decay,
            neutral_zone=neutral_zone,
            use_smiley_feedback=args.smiley_feedback,
            smiley_window_duration=args.smiley_window,
            smiley_threshold=args.smiley_threshold,
            smiley_prediction_rate=args.smiley_rate,
            neutral_threshold=neutral_threshold,
            api_base_url=args.api_base_url,
            api_light_path=args.api_light_path,
            api_fan_path=args.api_fan_path,
            api_timeout=args.api_timeout,
            api_confidence_threshold=args.api_conf_threshold,
            api_neutral_hold=args.api_neutral_hold,
            api_cooldown=args.api_cooldown,
            api_rearm_timeout=args.api_rearm_timeout
        )
    
    # Connect
    if args.simulate:
        # Simulator auto-connects
        bci.adapter.connect()
    elif args.direct:
        # Direct serial connection
        if not bci.adapter.connect():
            print(f"❌ Failed to connect to {args.port}")
            print(f"   Try different port or baud rate:")
            print(f"   python npg_realtime_bci.py --direct --port COM3 --baudrate 115200")
            sys.exit(1)
    else:
        # LSL connection via Chords-Python
        if not bci.adapter.connect():
            print("❌ Failed to find LSL stream. Make sure Chords-Python is running:")
            print("   python -m chordspy.connection --protocol usb")
            sys.exit(1)
    
    # Load baseline bias if not disabled
    if not args.no_bias_correction:
        bci._load_baseline_bias()
    
    # Run calibration if requested
    if args.calibrate:
        success = bci.run_calibration(duration_seconds=args.calibrate_duration)
        if not success:
            print("\n❌ Calibration failed. Run normal BCI anyway? (y/n): ", end='')
            response = input().strip().lower()
            if response != 'y':
                sys.exit(1)
        else:
            print("\n✅ Calibration complete!")
            print("   Run again without --calibrate to use the BCI system")
            sys.exit(0)
    
    # Start processing
    print("\n🧠 Starting BCI system...")
    print("   Press Ctrl+C to stop\n")
    
    bci.start()


if __name__ == "__main__":
    main()
