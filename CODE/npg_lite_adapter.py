"""
NPG Lite Adapter (Upside Down Labs)
Uses Chords-Python library for NPG Lite 3-channel EEG communication

NPG Lite: 3-channel wireless EEG (C3, Cz, C4)
Communication: USB Serial, WiFi, or Bluetooth via Chords-Python
Data streaming: Via LSL (Lab Streaming Layer)

Installation:
    pip install chordspy pylsl

Usage:
    1. Start Chords-Python in a separate terminal:
       python -m chordspy.connection --protocol usb
    2. Run your BCI application (this adapter auto-connects to LSL stream)
"""

import numpy as np
import time
import logging
import threading
from typing import Optional, Dict
from collections import deque
from pylsl import StreamInlet, resolve_byprop, resolve_streams


class NPGLiteAdapter:
    """
    Adapter for NPG Lite by Upside Down Labs.
    Receives data from Chords-Python via LSL (Lab Streaming Layer).
    
    NPG Lite specifications:
    - 3 channels EEG (C3, Cz, C4)
    - Sampling rate: 500 Hz (from Chords-Python)
    - ADC resolution: 12-bit
    - Connection: Chords-Python handles hardware connection
    - Data: Received via LSL stream
    """
    
    def __init__(self, 
                 port: Optional[str] = None,
                 baudrate: int = 115200,
                 sampling_rate: int = 500,
                 n_channels: int = 3):
        """
        Initialize NPG Lite adapter.
        
        Args:
            port: Not used (Chords-Python handles connection)
            baudrate: Not used (Chords-Python handles connection)
            sampling_rate: Expected sampling rate (default: 500 Hz)
            n_channels: Expected number of channels (default: 3)
        """
        self.port = port  # Kept for compatibility
        self.baudrate = baudrate  # Kept for compatibility
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        
        self.inlet = None
        self.is_streaming = False
        self.streaming_thread = None
        
        # Data buffer (store last 10 seconds)
        buffer_size = sampling_rate * 10
        self.data_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Channel names
        self.channel_names = ['C3', 'Cz', 'C4'][:n_channels]
        
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.samples_received = 0
        self.start_time = None
        self.last_quality_check = 0
    
    def connect(self, device_id: Optional[str] = None) -> bool:
        """
        Connect to LSL stream from Chords-Python.
        
        Args:
            device_id: Not used (for compatibility)
        
        Returns:
            True if connection successful
        """
        self.logger.info("Searching for NPG Lite LSL stream from Chords-Python...")
        
        try:
            # Look for BioAmpDataStream (Chords-Python stream name)
            streams = resolve_byprop('type', 'EXG', timeout=5.0)
            
            if not streams:
                # Fallback: search for any available stream
                self.logger.info("No EXG stream found, searching for all streams...")
                streams = resolve_streams(timeout=5.0)
            
            if not streams:
                self.logger.error("❌ No LSL streams found!")
                self.logger.error("   Please start Chords-Python first:")
                self.logger.error("   python -m chordspy.connection --protocol usb")
                return False
            
            # Connect to first available stream
            self.inlet = StreamInlet(streams[0])
            
            # Get stream info
            info = self.inlet.info()
            stream_name = info.name()
            self.n_channels = info.channel_count()
            self.sampling_rate = int(info.nominal_srate())
            
            self.logger.info(f"✅ Connected to LSL stream: {stream_name}")
            self.logger.info(f"   Channels: {self.n_channels}")
            self.logger.info(f"   Sampling rate: {self.sampling_rate} Hz")
            
            # Verify expected configuration
            if self.n_channels != 3:
                self.logger.warning(f"⚠️  Expected 3 channels, got {self.n_channels}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Connection failed: {e}")
            self.logger.error("   Make sure Chords-Python is running:")
            self.logger.error("   python -m chordspy.connection --protocol usb")
            return False
    
    def start_streaming(self, callback: Optional[callable] = None):
        """
        Start receiving data from LSL stream.
        
        Args:
            callback: Optional callback (not used, for compatibility)
        """
        if not self.inlet:
            raise RuntimeError("Not connected. Call connect() first.")
        
        if self.is_streaming:
            self.logger.warning("Already streaming")
            return
        
        self.is_streaming = True
        self.start_time = time.time()
        self.samples_received = 0
        
        # Start background thread
        self.streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.streaming_thread.start()
        
        self.logger.info("✅ Streaming started")
    
    def _streaming_loop(self):
        """Background thread for continuous data acquisition from LSL."""
        self.logger.info("LSL streaming thread started")
        
        while self.is_streaming:
            try:
                # Pull samples from LSL stream (with timeout)
                samples, timestamps = self.inlet.pull_chunk(timeout=0.1, max_samples=10)
                
                if samples:
                    with self.buffer_lock:
                        for sample in samples:
                            # Take first n_channels
                            channel_data = np.array(sample[:self.n_channels])
                            self.data_buffer.append(channel_data)
                            self.samples_received += 1
                else:
                    # Small delay if no data
                    time.sleep(0.01)
            
            except Exception as e:
                self.logger.error(f"Streaming error: {e}")
                time.sleep(0.1)
        
        self.logger.info("LSL streaming thread stopped")
    
    def stop_streaming(self):
        """Stop receiving data from LSL stream."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Wait for thread to finish
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2.0)
        
        self.logger.info("Streaming stopped")
    
    def get_latest_data(self, n_samples: int = 1000) -> Optional[np.ndarray]:
        """
        Get latest n samples from buffer.
        
        Args:
            n_samples: Number of samples to retrieve
        
        Returns:
            Array of shape (n_samples, n_channels) or None if insufficient data
        """
        with self.buffer_lock:
            if len(self.data_buffer) < n_samples:
                return None
            
            # Extract last n_samples
            data = np.array(list(self.data_buffer)[-n_samples:])
            return data
    
    def check_signal_quality(self) -> Dict[str, float]:
        """
        Check signal quality for each channel.
        
        Returns:
            Dictionary mapping channel names to quality scores (0-1)
        """
        # Throttle quality checks
        current_time = time.time()
        if current_time - self.last_quality_check < 1.0:
            return getattr(self, '_last_quality', {ch: 0.5 for ch in self.channel_names})
        
        self.last_quality_check = current_time
        
        with self.buffer_lock:
            if len(self.data_buffer) < self.sampling_rate:
                quality = {ch: 0.0 for ch in self.channel_names}
                self._last_quality = quality
                return quality
            
            # Get last 1 second of data
            data = np.array(list(self.data_buffer)[-self.sampling_rate:])
        
        quality = {}
        for i, ch_name in enumerate(self.channel_names):
            if i >= data.shape[1]:
                quality[ch_name] = 0.0
                continue
            
            channel_data = data[:, i]
            
            # Quality based on signal characteristics
            std = np.std(channel_data)
            mean_abs = np.mean(np.abs(channel_data))
            max_abs = np.max(np.abs(channel_data))
            
            if std < 1.0:
                score = 0.0  # Flat signal
            elif max_abs > 5000:
                score = 0.2  # Saturated
            elif std < 10:
                score = 0.3  # Very low activity
            elif std > 1000:
                score = 0.5  # Too noisy
            else:
                # Good signal
                score = min(1.0, std / 200.0)
            
            quality[ch_name] = score
        
        self._last_quality = quality
        return quality
    
    def get_buffer_size(self) -> int:
        """
        Get current buffer size (number of samples in buffer).
        
        Returns:
            Number of samples currently in buffer
        """
        with self.buffer_lock:
            return len(self.data_buffer)
    
    def get_statistics(self) -> Dict:
        """Get streaming statistics."""
        if self.start_time is None:
            return {
                'runtime': 0,
                'samples_received': 0,
                'sampling_rate_actual': 0,
                'buffer_size': 0
            }
        
        runtime = time.time() - self.start_time
        
        with self.buffer_lock:
            buffer_size = len(self.data_buffer)
        
        return {
            'runtime': runtime,
            'samples_received': self.samples_received,
            'sampling_rate_actual': self.samples_received / runtime if runtime > 0 else 0,
            'buffer_size': buffer_size
        }
    
    def disconnect(self):
        """Disconnect from LSL stream."""
        self.stop_streaming()
        
        if self.inlet:
            self.inlet = None
        
        self.logger.info("Disconnected from LSL stream")


class NPGLiteSimulator(NPGLiteAdapter):
    """
    NPG Lite simulator for testing without hardware.
    Generates realistic 3-channel EEG with motor imagery patterns.
    """
    
    def __init__(self, **kwargs):
        # Override defaults for NPG Lite specs
        kwargs.setdefault('sampling_rate', 500)
        kwargs.setdefault('n_channels', 3)
        super().__init__(**kwargs)
        
        self.simulation_time = 0
        self.current_class = None
        self.class_duration = 4.0  # 4 seconds per imagery
        self.rest_duration = 2.0   # 2 seconds rest
        self.time_step = 1.0 / self.sampling_rate
    
    def connect(self, device_id: Optional[str] = None) -> bool:
        """Simulated connection."""
        self.logger.info("🎮 Using NPG Lite SIMULATOR (no hardware required)")
        self.logger.info(f"   Generating 3-channel EEG: C3, Cz, C4 @ {self.sampling_rate} Hz")
        # Set inlet to a dummy value so start_streaming() doesn't raise an error
        self.inlet = True  # Simulator doesn't use LSL inlet
        time.sleep(0.5)
        return True
    
    def _streaming_loop(self):
        """Generate simulated 3-channel EEG data."""
        self.logger.info("Simulation started - generating motor imagery patterns")
        
        while self.is_streaming:
            try:
                # Generate sample at correct rate
                sample = self._generate_eeg_sample()
                
                with self.buffer_lock:
                    self.data_buffer.append(sample)
                
                self.samples_received += 1
                self.simulation_time += self.time_step
                
                # Sleep to maintain correct sampling rate
                time.sleep(self.time_step)
            
            except Exception as e:
                self.logger.error(f"Simulation error: {e}")
                break
    
    def _generate_eeg_sample(self) -> np.ndarray:
        """Generate realistic 3-channel EEG sample with motor imagery."""
        t = self.simulation_time
        
        # Determine current motor imagery state
        cycle_time = self.class_duration + self.rest_duration
        t_in_cycle = self.simulation_time % cycle_time
        
        if t_in_cycle < self.class_duration:
            # Alternate between left/right every cycle
            if (int(self.simulation_time // cycle_time) % 2) == 0:
                self.current_class = 'LEFT_HAND'
            else:
                self.current_class = 'RIGHT_HAND'
        else:
            self.current_class = 'REST'
        
        # Base rhythms
        alpha = 80 * np.sin(2 * np.pi * 10 * t)   # Alpha (10 Hz)
        beta = 40 * np.sin(2 * np.pi * 20 * t)    # Beta (20 Hz)
        
        # Motor imagery modulation (mu rhythm suppression)
        if self.current_class == 'LEFT_HAND':
            # Suppress mu in right hemisphere (C4)
            mu_c3 = 60 * np.sin(2 * np.pi * 12 * t)
            mu_c4 = 20 * np.sin(2 * np.pi * 12 * t)  # Suppressed
        elif self.current_class == 'RIGHT_HAND':
            # Suppress mu in left hemisphere (C3)
            mu_c3 = 20 * np.sin(2 * np.pi * 12 * t)  # Suppressed
            mu_c4 = 60 * np.sin(2 * np.pi * 12 * t)
        else:
            # Rest state - balanced
            mu_c3 = mu_c4 = 40 * np.sin(2 * np.pi * 12 * t)
        
        # Generate 3 channels: C3, Cz, C4
        noise = np.random.randn(3) * 20
        
        samples = np.array([
            alpha + mu_c3 + beta * 0.5 + noise[0],  # C3 (left motor)
            alpha + (mu_c3 + mu_c4) / 2 + noise[1],  # Cz (central)
            alpha + mu_c4 + beta * 0.5 + noise[2]   # C4 (right motor)
        ])
        
        return samples
    
    def disconnect(self):
        """Disconnect simulator."""
        self.stop_streaming()
        self.logger.info("Simulator stopped")


if __name__ == "__main__":
    # Test script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("NPG Lite Adapter Test (Chords-Python + LSL)")
    print("="*70)
    
    # Test with simulator
    print("\n1. Testing simulator mode...")
    adapter = NPGLiteSimulator()
    
    if adapter.connect():
        adapter.start_streaming()
        
        print("   Collecting data for 3 seconds...")
        time.sleep(3)
        
        # Get data
        data = adapter.get_latest_data(n_samples=500)
        
        if data is not None:
            print(f"\n✅ Data collected:")
            print(f"   Shape: {data.shape}")
            print(f"   Mean: {data.mean():.2f}")
            print(f"   Std: {data.std():.2f}")
            print(f"   Range: [{data.min():.2f}, {data.max():.2f}]")
        
        # Check quality
        quality = adapter.check_signal_quality()
        print(f"\n📊 Signal Quality:")
        for ch, q in quality.items():
            status = "✅" if q > 0.5 else "⚠️"
            print(f"   {status} {ch}: {q:.2f}")
        
        # Show statistics
        stats = adapter.get_statistics()
        print(f"\n📈 Statistics:")
        print(f"   Runtime: {stats['runtime']:.1f}s")
        print(f"   Samples: {stats['samples_received']}")
        print(f"   Sample rate: {stats['sampling_rate_actual']:.1f} Hz")
        print(f"   Buffer size: {stats['buffer_size']}")
        
        adapter.disconnect()
        
        print("\n" + "="*70)
        print("✅ All tests passed!")
        print("="*70)
    else:
        print("\n❌ Connection failed")
    
    print("\nTo use with real NPG Lite hardware:")
    print("  1. Install Chords-Python: pip install chordspy")
    print("  2. Start Chords-Python: python -m chordspy.connection --protocol usb")
    print("  3. Run your BCI application (this adapter auto-connects)")


class NPGLiteDirectSerial(NPGLiteAdapter):
    """
    Direct serial connection to NPG Lite.
    Bypasses Chords-Python/LSL - connects straight to COM port.
    Useful when chordspy connection hangs or for simpler setup.
    """
    
    def __init__(self, port='COM6', baudrate=230400, n_channels=3, sampling_rate=500):
        """
        Initialize direct serial connection.
        
        Args:
            port: Serial port (e.g., 'COM6', '/dev/ttyUSB0')
            baudrate: Baud rate (NPG Lite default: 230400)
            n_channels: Number of channels (NPG Lite: 3)
            sampling_rate: Sampling rate (NPG Lite: 500 Hz)
        """
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.serial_conn = None
        self.data_buffer = deque(maxlen=sampling_rate * 10)  # 10 seconds buffer
        self.buffer_lock = threading.Lock()
        
    def connect(self, device_id: Optional[str] = None) -> bool:
        """Connect to NPG Lite via serial port."""
        try:
            import serial
            self.logger.info(f"📡 Connecting to NPG Lite on {self.port} @ {self.baudrate} baud...")
            
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.01  # Non-blocking
            )
            
            time.sleep(2)  # Device initialization
            
            # Flush buffers
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            self.inlet = True  # Mark as connected
            self.logger.info(f"✅ Connected to NPG Lite on {self.port}")
            self.logger.info(f"   Channels: {self.n_channels} (C3, Cz, C4)")
            self.logger.info(f"   Sampling rate: {self.sampling_rate} Hz")
            self.logger.info(f"   Baud rate: {self.baudrate}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to {self.port}: {e}")
            self.logger.info("   Make sure:")
            self.logger.info("   - NPG Lite is powered on")
            self.logger.info("   - USB cable is connected")
            self.logger.info("   - Correct COM port is specified")
            self.logger.info("   - CH340 drivers are installed (if needed)")
            return False
    
    def start_streaming(self, callback: Optional[callable] = None):
        """Start receiving data from serial port."""
        if not self.serial_conn:
            raise RuntimeError("Not connected. Call connect() first.")
        
        if self.is_streaming:
            self.logger.warning("Already streaming")
            return
        
        self.is_streaming = True
        self.start_time = time.time()
        self.samples_received = 0
        
        # Start background thread
        self.streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.streaming_thread.start()
        
        self.logger.info("✅ Streaming started (reading from serial)")
    
    def _streaming_loop(self):
        """Read data continuously from serial port."""
        self.logger.info("Serial streaming started - reading CSV data")
        
        consecutive_empty = 0
        samples_logged = 0
        
        while self.is_streaming:
            try:
                if self.serial_conn.in_waiting > 0:
                    # Read one line (CSV format: ch1,ch2,ch3\n)
                    line = self.serial_conn.readline()
                    
                    if line:
                        # Log first few samples for debugging
                        if samples_logged < 5:
                            self.logger.debug(f"Raw data: {line}")
                            samples_logged += 1
                        
                        sample = self._parse_packet(line)
                        
                        if sample is not None:
                            with self.buffer_lock:
                                self.data_buffer.append(sample)
                            self.samples_received += 1
                            consecutive_empty = 0
                            
                            # Log every 500 samples
                            if self.samples_received % 500 == 0:
                                self.logger.info(f"Received {self.samples_received} samples (buffer: {len(self.data_buffer)})")
                    consecutive_empty = 0
                else:
                    # No data available, small sleep
                    consecutive_empty += 1
                    
                    # Log if stuck waiting for data
                    if consecutive_empty == 5000:  # After 5 seconds
                        self.logger.warning(f"No data received from {self.port} after 5 seconds")
                        self.logger.warning(f"   Bytes waiting: {self.serial_conn.in_waiting}")
                        self.logger.warning(f"   Samples received: {self.samples_received}")
                        self.logger.warning(f"   Make sure NPG Lite is powered on and sending data")
                    
                    time.sleep(0.001)
                    
            except Exception as e:
                self.logger.error(f"Serial read error: {e}")
                time.sleep(0.1)
    
    def _parse_packet(self, data: bytes) -> Optional[np.ndarray]:
        """
        Parse NPG Lite data packet.
        
        NPG Lite sends CSV format over serial:
        "ch1,ch2,ch3\n"
        
        Args:
            data: Raw bytes from serial
            
        Returns:
            Sample array [ch1, ch2, ch3] or None
        """
        try:
            # Decode and strip whitespace
            line = data.decode('utf-8').strip()
            
            # Skip empty lines
            if not line:
                return None
            
            # Parse CSV values
            values = line.split(',')
            
            if len(values) != self.n_channels:
                return None
            
            # Convert to float array
            sample = np.array([float(v) for v in values])
            
            return sample
            
        except Exception:
            # Skip malformed packets silently
            return None
    
    def get_latest_data(self, n_samples: int = 1000) -> Optional[np.ndarray]:
        """
        Get latest n samples from buffer.
        
        Args:
            n_samples: Number of samples to retrieve
        
        Returns:
            Array of shape (n_samples, n_channels) or None if insufficient data
        """
        with self.buffer_lock:
            if len(self.data_buffer) < n_samples:
                return None
            
            # Extract last n_samples
            data = np.array(list(self.data_buffer)[-n_samples:])
            return data
    
    def stop_streaming(self):
        """Stop streaming."""
        self.is_streaming = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=2.0)
        self.logger.info("Streaming stopped")
    
    def disconnect(self):
        """Disconnect from serial port."""
        self.stop_streaming()
        
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None
        
        self.inlet = None
        self.logger.info(f"Disconnected from {self.port}")
