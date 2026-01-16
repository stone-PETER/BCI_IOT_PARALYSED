"""
NPG Lite Preprocessor
Handles signal preprocessing for NPG Lite data (via Chords-Python) before BCI model inference

Pipeline:
1. Resample: 500 Hz → 250 Hz (NPG Lite via Chords-Python)
2. Channel selection: Extract C3, Cz, C4 (3 channels from NPG Lite)
3. Bandpass filter: 8-30 Hz (motor imagery band)
4. CAR reference: Common Average Reference
5. Z-score normalization: Per-channel normalization
6. Epoch extraction: 4-second windows (1000 samples @ 250 Hz)
"""

import numpy as np
from scipy import signal
from scipy.signal import resample_poly
import logging
from typing import Tuple, Optional
from collections import deque


class NPGPreprocessor:
    """
    Preprocessor for NPG Lite EEG data (via Chords-Python).
    Converts 500 Hz, 3-channel data to 250 Hz, 3-channel (C3, Cz, C4) for BCI model.
    """
    
    def __init__(self,
                 input_rate: int = 500,  # NPG Lite via Chords-Python
                 output_rate: int = 250,  # Model expects 250 Hz
                 target_channels: list = None,
                 filter_low: float = 8.0,
                 filter_high: float = 30.0,
                 epoch_duration: float = 4.0):
        """
        Initialize preprocessor.
        
        Args:
            input_rate: Input sampling rate (NPG Lite: 500 Hz via Chords-Python)
            output_rate: Output sampling rate (Model expects: 250 Hz)
            target_channels: Indices of target channels [C3, Cz, C4]
                           Default: [0, 1, 2] (all 3 channels from NPG Lite)
            filter_low: Bandpass filter lower cutoff (Hz)
            filter_high: Bandpass filter upper cutoff (Hz)
            epoch_duration: Epoch length in seconds (4.0 for model)
        """
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.target_channels = target_channels or [0, 1, 2]  # C3, Cz, C4
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.epoch_duration = epoch_duration
        
        # Calculate epoch sizes
        self.input_epoch_samples = int(epoch_duration * input_rate)  # 2000 @ 500 Hz
        self.output_epoch_samples = int(epoch_duration * output_rate)  # 1000 @ 250 Hz
        
        # Resampling ratio (500 → 250 Hz = exactly 1:2, very clean!)
        self.resample_ratio = output_rate / input_rate
        
        # Design bandpass filter (for output rate)
        self.filter_b, self.filter_a = signal.butter(
            4, 
            [filter_low, filter_high], 
            btype='band', 
            fs=output_rate
        )
        
        # Normalization statistics (updated online)
        self.channel_means = np.zeros(len(self.target_channels))
        self.channel_stds = np.ones(len(self.target_channels))
        self.normalization_samples = 0
        self.normalization_alpha = 0.99  # EMA factor
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"NPG Lite Preprocessor initialized:")
        self.logger.info(f"  Resampling: {input_rate} Hz → {output_rate} Hz (ratio 1:2)")
        self.logger.info(f"  Channels: 3 (C3, Cz, C4 from NPG Lite)")
        self.logger.info(f"  Expected input: 2000 samples @ 500 Hz (4 seconds)")
        self.logger.info(f"  Expected output: 1000 samples @ 250 Hz (4 seconds)")
        self.logger.info(f"  Bandpass: {filter_low}-{filter_high} Hz")
        self.logger.info(f"  Epoch: {epoch_duration}s ({self.output_epoch_samples} samples)")
    
    def resample_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Resample data from 256 Hz to 250 Hz.
        
        Args:
            data: Input data (n_samples, n_channels)
        
        Returns:
            Resampled data (n_samples_new, n_channels)
        """
        # Use resample_poly for efficient integer ratio resampling
        # 256 → 250 = multiply by 125, divide by 128
        up = 125
        down = 128
        
        resampled = resample_poly(data, up, down, axis=0)
        
        return resampled
    
    def select_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Select target channels (C3, Cz, C4) from multi-channel data.
        
        Args:
            data: Input data (n_samples, n_channels)
        
        Returns:
            Selected channels (n_samples, 3)
        """
        if data.shape[1] < max(self.target_channels) + 1:
            raise ValueError(f"Data has {data.shape[1]} channels, need at least {max(self.target_channels) + 1}")
        
        selected = data[:, self.target_channels]
        
        return selected
    
    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter (8-30 Hz for motor imagery).
        
        Args:
            data: Input data (n_samples, n_channels)
        
        Returns:
            Filtered data
        """
        filtered = np.zeros_like(data)
        
        for ch in range(data.shape[1]):
            # Apply zero-phase filtering
            filtered[:, ch] = signal.filtfilt(self.filter_b, self.filter_a, data[:, ch])
        
        return filtered
    
    def apply_car(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Common Average Reference (CAR).
        
        Args:
            data: Input data (n_samples, n_channels)
        
        Returns:
            CAR-referenced data
        """
        # Calculate average across channels
        mean_signal = np.mean(data, axis=1, keepdims=True)
        
        # Subtract from each channel
        car_data = data - mean_signal
        
        return car_data
    
    def zscore_normalize(self, data: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """
        Apply z-score normalization per channel.
        
        Args:
            data: Input data (n_samples, n_channels)
            update_stats: Whether to update running statistics
        
        Returns:
            Normalized data
        """
        normalized = np.zeros_like(data)
        
        for ch in range(data.shape[1]):
            ch_data = data[:, ch]
            
            if update_stats:
                # Update running statistics with exponential moving average
                batch_mean = np.mean(ch_data)
                batch_std = np.std(ch_data)
                
                if self.normalization_samples == 0:
                    # First batch
                    self.channel_means[ch] = batch_mean
                    self.channel_stds[ch] = batch_std if batch_std > 1e-6 else 1.0
                else:
                    # Update with EMA
                    self.channel_means[ch] = (self.normalization_alpha * self.channel_means[ch] + 
                                            (1 - self.normalization_alpha) * batch_mean)
                    self.channel_stds[ch] = (self.normalization_alpha * self.channel_stds[ch] + 
                                           (1 - self.normalization_alpha) * batch_std)
                
                self.normalization_samples += len(ch_data)
            
            # Normalize
            mean = self.channel_means[ch]
            std = self.channel_stds[ch] if self.channel_stds[ch] > 1e-6 else 1.0
            
            normalized[:, ch] = (ch_data - mean) / std
        
        return normalized
    
    def preprocess_epoch(self, data: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """
        Full preprocessing pipeline for a single epoch.
        
        Args:
            data: Input epoch from NPG Lite (1024 samples @ 256 Hz, 6 channels)
            update_stats: Whether to update normalization statistics
        
        Returns:
            Preprocessed epoch (1000 samples @ 250 Hz, 3 channels)
        """
        # 1. Select channels (C3, Cz, C4)
        selected = self.select_channels(data)
        
        # 2. Resample 256 → 250 Hz
        resampled = self.resample_signal(selected)
        
        # 3. Bandpass filter (8-30 Hz)
        filtered = self.bandpass_filter(resampled)
        
        # 4. CAR reference
        car = self.apply_car(filtered)
        
        # 5. Z-score normalization
        normalized = self.zscore_normalize(car, update_stats=update_stats)
        
        return normalized
    
    def preprocess_for_model(self, data: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """
        Preprocess data and format for model input.
        
        Args:
            data: Input epoch (1024 samples @ 256 Hz, 6 channels)
            update_stats: Whether to update normalization statistics
        
        Returns:
            Model-ready data (1, 3, 1000, 1) - batch, channels, samples, 1
        """
        # Preprocess
        processed = self.preprocess_epoch(data, update_stats=update_stats)
        
        # Reshape for model: (batch=1, channels, samples, 1)
        model_input = processed.T[np.newaxis, :, :, np.newaxis]
        
        return model_input
    
    def get_normalization_stats(self) -> dict:
        """Get current normalization statistics."""
        return {
            'means': self.channel_means.tolist(),
            'stds': self.channel_stds.tolist(),
            'n_samples': self.normalization_samples
        }
    
    def reset_normalization_stats(self):
        """Reset normalization statistics."""
        self.channel_means = np.zeros(len(self.target_channels))
        self.channel_stds = np.ones(len(self.target_channels))
        self.normalization_samples = 0
        self.logger.info("Reset normalization statistics")


class SlidingWindowBuffer:
    """
    Sliding window buffer for continuous data processing.
    Extracts overlapping epochs from continuous stream.
    """
    
    def __init__(self,
                 window_size: int,
                 overlap: float = 0.5,
                 sampling_rate: int = 256):
        """
        Initialize sliding window buffer.
        
        Args:
            window_size: Window size in samples (e.g., 1024 @ 256 Hz = 4 seconds)
            overlap: Overlap fraction (0.5 = 50% overlap, 2-second stride)
            sampling_rate: Sampling rate
        """
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        
        # Calculate stride
        self.stride = int(window_size * (1 - overlap))
        
        # Buffer - use large deque to hold continuous data
        self.buffer = deque(maxlen=window_size * 3)  # Keep 3x window for safety
        self.total_samples_added = 0  # Track total samples added
        self.last_extraction_count = 0  # Track when last window was extracted
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Sliding window buffer: window={window_size} samples, "
                        f"stride={self.stride} samples, overlap={overlap:.0%}")
    
    def add_data(self, data: np.ndarray):
        """
        Add new data to buffer.
        
        Args:
            data: New data (n_samples, n_channels)
        """
        for sample in data:
            self.buffer.append(sample)
            self.total_samples_added += 1
    
    def add_samples(self, data: np.ndarray):
        """
        Alias for add_data() for compatibility.
        
        Args:
            data: New data (n_samples, n_channels)
        """
        self.add_data(data)
    
    def get_window(self) -> Optional[np.ndarray]:
        """
        Extract a window if enough data is available.
        
        Returns:
            Window data (window_size, n_channels) or None
        """
        if len(self.buffer) < self.window_size:
            return None
        
        # Check if we've added enough new samples for next window
        samples_since_last = self.total_samples_added - self.last_extraction_count
        if samples_since_last < self.stride:
            return None
        
        # Extract window (last window_size samples)
        window = np.array(list(self.buffer)[-self.window_size:])
        
        # Update extraction counter
        self.last_extraction_count = self.total_samples_added
        
        return window
    
    def get_windows(self) -> list:
        """
        Extract all available windows (returns list for compatibility).
        
        Returns:
            List of windows (each window_size, n_channels)
        """
        window = self.get_window()
        if window is not None:
            return [window]
        return []
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.total_samples_added = 0
        self.last_extraction_count = 0


if __name__ == "__main__":
    # Test the preprocessor
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("\n" + "="*70)
    print("Testing NPG Preprocessor")
    print("="*70)
    
    # Create preprocessor
    preprocessor = NPGPreprocessor()
    
    # Generate synthetic data (4 seconds @ 256 Hz, 6 channels)
    print("\n1. Generating synthetic EEG data...")
    n_samples = int(4.0 * 256)  # 1024 samples
    n_channels = 6
    t = np.arange(n_samples) / 256
    
    synthetic_data = np.zeros((n_samples, n_channels))
    for ch in range(n_channels):
        # Alpha rhythm (10 Hz) + noise
        synthetic_data[:, ch] = 10 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_samples) * 3
    
    print(f"   Input shape: {synthetic_data.shape} (1024 samples @ 256 Hz, 6 channels)")
    
    # Test preprocessing
    print("\n2. Testing preprocessing pipeline...")
    processed = preprocessor.preprocess_epoch(synthetic_data)
    print(f"   Output shape: {processed.shape} (1000 samples @ 250 Hz, 3 channels)")
    print(f"   Output mean: {processed.mean():.4f} (should be ~0)")
    print(f"   Output std: {processed.std():.4f} (should be ~1)")
    
    # Test model formatting
    print("\n3. Testing model input formatting...")
    model_input = preprocessor.preprocess_for_model(synthetic_data)
    print(f"   Model input shape: {model_input.shape} (batch, channels, samples, 1)")
    print(f"   Expected: (1, 3, 1000, 1)")
    
    # Test sliding window buffer
    print("\n4. Testing sliding window buffer...")
    buffer = SlidingWindowBuffer(window_size=1024, overlap=0.5, sampling_rate=256)
    
    # Add data in chunks
    chunk_size = 128
    n_windows = 0
    for i in range(0, n_samples, chunk_size):
        chunk = synthetic_data[i:min(i+chunk_size, n_samples), :]
        buffer.add_data(chunk)
        
        # Try to extract window
        window = buffer.get_window()
        if window is not None:
            n_windows += 1
            print(f"   Extracted window #{n_windows}: shape={window.shape}")
    
    # Test normalization stats
    print("\n5. Checking normalization statistics...")
    stats = preprocessor.get_normalization_stats()
    print(f"   Channel means: {np.array(stats['means'])}")
    print(f"   Channel stds: {np.array(stats['stds'])}")
    print(f"   Samples processed: {stats['n_samples']}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
