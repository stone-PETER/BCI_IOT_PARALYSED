"""
NPG Lite Real-time BCI System
Complete integration: NPG Lite → Preprocessing → Inference → Commands

Real-time motor imagery classification for paralysis assistance
"""

import numpy as np
import logging
import time
import argparse
import sys
import threading
from typing import Optional
from pathlib import Path
from collections import deque

from npg_lite_adapter import NPGLiteAdapter, NPGLiteSimulator, NPGLiteDirectSerial
from npg_preprocessor import NPGPreprocessor, SlidingWindowBuffer
from npg_inference import NPGInferenceEngine


class NPGRealtimeBCI:
    """
    Complete real-time BCI system for NPG Lite.
    
    Pipeline:
    NPG Lite (256 Hz, 6 ch) → Resample (250 Hz) → Select channels (C3, Cz, C4) →
    Bandpass (8-30 Hz) → CAR → Z-score → Model → Classification → Command
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.7,
                 smoothing_window: int = 3,
                 window_overlap: float = 0.5,
                 simulate: bool = False):
        """
        Initialize real-time BCI system.
        
        Args:
            model_path: Path to trained model
            confidence_threshold: Minimum confidence for command execution
            smoothing_window: Number of predictions to smooth
            window_overlap: Overlap ratio for sliding windows (0-1)
            simulate: Use simulator instead of real device
        """
        self.simulate = simulate
        self.confidence_threshold = confidence_threshold
        self.is_running = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("="*70)
        self.logger.info("NPG Lite Real-time BCI System")
        self.logger.info("="*70)
        
        # Initialize components
        self.logger.info("\n1. Initializing components...")
        
        # NPG adapter
        if simulate:
            self.logger.info("   Using SIMULATOR mode")
            self.adapter = NPGLiteSimulator()
        else:
            self.logger.info("   Using HARDWARE mode")
            self.adapter = NPGLiteAdapter()
        
        # Preprocessor
        self.preprocessor = NPGPreprocessor()
        self.logger.info("   ✅ Preprocessor ready")
        
        # Inference engine
        self.inference = NPGInferenceEngine(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            smoothing_window=smoothing_window
        )
        self.logger.info("   ✅ Inference engine ready")
        
        # Sliding window buffer
        window_size = 1024  # 4 seconds @ 256 Hz
        stride = int(window_size * (1 - window_overlap))
        self.window_buffer = SlidingWindowBuffer(
            window_size=window_size,
            overlap=window_overlap
        )
        self.logger.info(f"   ✅ Window buffer ready (overlap={window_overlap:.0%})")
        
        # Command tracking
        self.last_command = None
        self.last_command_time = 0
        self.command_counts = {'LEFT_HAND': 0, 'RIGHT_HAND': 0, 'UNCERTAIN': 0}
        
        # Performance metrics
        self.processing_times = deque(maxlen=100)
        self.total_epochs_processed = 0
        self.start_time = None
        
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
        warmup_samples = 1024  # Need 4 seconds for first window
        chunk_size = 128  # Process in smaller chunks after warmup
        
        self.logger.info(f"Warming up... collecting {warmup_samples} samples")
        
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
            epoch_data: Raw data (1024, 6) @ 256 Hz
        """
        start_time = time.time()
        
        try:
            # Preprocess
            preprocessed = self.preprocessor.preprocess_for_model(epoch_data)
            
            # Inference
            class_idx, confidence, class_name = self.inference.predict_smoothed(preprocessed)
            
            # Determine command
            if confidence >= self.confidence_threshold:
                command = class_name
                self.command_counts[command] += 1
            else:
                command = "UNCERTAIN"
                self.command_counts["UNCERTAIN"] += 1
            
            # Check if command changed
            command_changed = (command != self.last_command and 
                             command != "UNCERTAIN")
            
            # Update tracking
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
            self.logger.error(f"Epoch processing error: {e}")
    
    def _log_command(self, command: str, confidence: float, proc_time: float):
        """Log a new command."""
        icon = "👈" if command == "LEFT_HAND" else "👉"
        
        self.logger.info(
            f"{icon} {command:12} | "
            f"Confidence: {confidence:5.1%} | "
            f"Processing: {proc_time*1000:5.1f}ms"
        )
    
    def _log_status(self):
        """Log periodic status update."""
        runtime = time.time() - self.start_time
        epochs_per_sec = self.total_epochs_processed / runtime
        avg_proc_time = np.mean(self.processing_times) if self.processing_times else 0
        
        self.logger.info(
            f"📊 Status | "
            f"Epochs: {self.total_epochs_processed:4} | "
            f"Rate: {epochs_per_sec:4.1f}/s | "
            f"Avg time: {avg_proc_time*1000:5.1f}ms"
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
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold for classification (0-1, default: 0.7)')
    parser.add_argument('--smoothing', type=int, default=3,
                       help='Smoothing window size for predictions (default: 3)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap ratio (0-1, default: 0.5)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Check model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = Path(__file__).parent / 'models' / 'best_eegnet_2class_bci2b.keras'
    
    if not model_path.exists():
        print(f"❌ Error: Model not found: {model_path}")
        print("   Train the model first with: python train_model_2b.py")
        sys.exit(1)
    
    # Create BCI system based on mode
    if args.simulate:
        # Simulator mode
        bci = NPGRealtimeBCI(
            model_path=str(model_path),
            confidence_threshold=args.confidence,
            smoothing_window=args.smoothing,
            window_overlap=args.overlap,
            simulate=True
        )
    elif args.direct:
        # Direct serial connection mode
        bci = NPGRealtimeBCI(
            model_path=str(model_path),
            confidence_threshold=args.confidence,
            smoothing_window=args.smoothing,
            window_overlap=args.overlap,
            simulate=False
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
            simulate=False
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
    
    # Start processing
    print("\n🧠 Starting BCI system...")
    print("   Press Ctrl+C to stop\n")
    
    bci.start()


if __name__ == "__main__":
    main()
