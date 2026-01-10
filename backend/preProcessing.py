#!/usr/bin/env python3
"""
Real-time EEG Preprocessing Pipeline
Modular preprocessing components for production-ready BCI systems
"""

import numpy as np
import scipy.signal
from scipy.signal import butter, filtfilt, sosfiltfilt
from typing import Tuple, Optional, Union
import logging

class EEGPreprocessor:
    """
    Production-ready EEG preprocessing pipeline
    Supports both real-time streaming and batch processing
    """
    
    def __init__(self, 
                 sampling_rate: int = 250,
                 bandpass_low: float = 8.0,
                 bandpass_high: float = 30.0,
                 filter_order: int = 4,
                 channels: int = 22):
        """
        Initialize EEG preprocessor with configurable parameters
        
        Args:
            sampling_rate: EEG sampling frequency in Hz
            bandpass_low: Lower cutoff frequency for bandpass filter
            bandpass_high: Upper cutoff frequency for bandpass filter
            filter_order: Butterworth filter order
            channels: Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.filter_order = filter_order
        self.channels = channels
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize filter coefficients
        self._setup_filters()
        
        # Buffer for real-time filtering
        self.filter_buffer_size = int(0.5 * sampling_rate)  # 0.5 second buffer
        self.filter_buffer = np.zeros((channels, self.filter_buffer_size))
        self.buffer_index = 0
        
        self.logger.info(f"EEG Preprocessor initialized: {sampling_rate}Hz, "
                        f"Bandpass: {bandpass_low}-{bandpass_high}Hz, "
                        f"Channels: {channels}")
    
    def _setup_filters(self):
        """Setup Butterworth bandpass filter coefficients"""
        try:
            nyquist = self.sampling_rate / 2
            low_norm = self.bandpass_low / nyquist
            high_norm = self.bandpass_high / nyquist
            
            # Create second-order sections for stability
            self.sos = butter(self.filter_order, [low_norm, high_norm], 
                            btype='band', output='sos')
            
            self.logger.info("Bandpass filter initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Filter initialization failed: {e}")
            raise
    
    def bandpass_filter(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Apply bandpass filter to EEG data
        
        Args:
            data: EEG data array (channels x samples) or (samples x channels)
            axis: Axis along which to apply filter (default: last axis)
            
        Returns:
            Filtered EEG data with same shape as input
        """
        try:
            if data.size == 0:
                return data
                
            # Apply zero-phase filtering
            filtered_data = sosfiltfilt(self.sos, data, axis=axis)
            
            return filtered_data.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Bandpass filtering failed: {e}")
            return data
    
    def common_average_reference(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Common Average Reference (CAR) to EEG data
        
        Args:
            data: EEG data array (channels x samples)
            
        Returns:
            CAR-referenced EEG data
        """
        try:
            if data.ndim != 2:
                raise ValueError("Data must be 2D (channels x samples)")
            
            # Calculate common average across channels
            common_avg = np.mean(data, axis=0, keepdims=True)
            
            # Subtract common average from each channel
            car_data = data - common_avg
            
            return car_data.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"CAR processing failed: {e}")
            return data
    
    def z_score_normalize(self, data: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Apply z-score normalization to EEG data
        
        Args:
            data: EEG data array
            axis: Axis along which to normalize
            
        Returns:
            Z-score normalized data
        """
        try:
            # Calculate mean and standard deviation
            mean_val = np.mean(data, axis=axis, keepdims=True)
            std_val = np.std(data, axis=axis, keepdims=True)
            
            # Avoid division by zero
            std_val = np.where(std_val == 0, 1, std_val)
            
            # Apply z-score normalization
            normalized_data = (data - mean_val) / std_val
            
            return normalized_data.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Z-score normalization failed: {e}")
            return data
    
    def preprocess_epoch(self, epoch: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for a single epoch
        
        Args:
            epoch: EEG epoch data (channels x samples)
            
        Returns:
            Fully preprocessed epoch ready for classification
        """
        try:
            # Ensure correct data type
            epoch = epoch.astype(np.float32)
            
            # 1. Bandpass filter
            filtered = self.bandpass_filter(epoch, axis=1)
            
            # 2. Common Average Reference
            car_applied = self.common_average_reference(filtered)
            
            # 3. Z-score normalization
            normalized = self.z_score_normalize(car_applied, axis=1)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Epoch preprocessing failed: {e}")
            return epoch
    
    def preprocess_realtime_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Preprocess real-time data chunk with buffering
        
        Args:
            chunk: Small chunk of EEG data (channels x samples)
            
        Returns:
            Preprocessed chunk
        """
        try:
            # For real-time processing, we need to maintain filter state
            # This is a simplified version - full implementation would use
            # proper filter state management for continuous filtering
            
            return self.preprocess_epoch(chunk)
            
        except Exception as e:
            self.logger.error(f"Real-time chunk preprocessing failed: {e}")
            return chunk
    
    def format_for_model(self, epoch: np.ndarray) -> np.ndarray:
        """
        Format preprocessed epoch for EEGNet model input
        
        Args:
            epoch: Preprocessed epoch (channels x samples)
            
        Returns:
            Model-ready data (1, channels, samples, 1)
        """
        try:
            # EEGNet expects (batch, channels, samples, 1)
            if epoch.ndim == 2:  # (channels, samples)
                formatted = epoch[np.newaxis, :, :, np.newaxis]  # (1, channels, samples, 1)
            else:
                raise ValueError(f"Expected 2D epoch, got {epoch.ndim}D")
            
            return formatted.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Model formatting failed: {e}")
            # Return a safe default
            return np.zeros((1, self.channels, 1000, 1), dtype=np.float32)

class RealTimeBuffer:
    """
    Circular buffer for real-time EEG data streaming
    """
    
    def __init__(self, 
                 channels: int = 22,
                 buffer_size: int = 2500,  # 10 seconds at 250Hz
                 window_size: int = 1000):  # 4 seconds at 250Hz
        """
        Initialize real-time buffer
        
        Args:
            channels: Number of EEG channels
            buffer_size: Total buffer size in samples
            window_size: Classification window size in samples
        """
        self.channels = channels
        self.buffer_size = buffer_size
        self.window_size = window_size
        
        # Circular buffer
        self.buffer = np.zeros((channels, buffer_size), dtype=np.float32)
        self.write_index = 0
        self.samples_count = 0
        
        self.logger = logging.getLogger(__name__)
        
    def add_data(self, data: np.ndarray) -> None:
        """
        Add new data to the circular buffer
        
        Args:
            data: New EEG data (channels x samples)
        """
        try:
            if data.ndim != 2 or data.shape[0] != self.channels:
                raise ValueError(f"Expected data shape ({self.channels}, samples), "
                               f"got {data.shape}")
            
            samples = data.shape[1]
            
            # Handle buffer wrap-around
            if self.write_index + samples <= self.buffer_size:
                # No wrap-around needed
                self.buffer[:, self.write_index:self.write_index + samples] = data
            else:
                # Split across buffer boundary
                first_part = self.buffer_size - self.write_index
                self.buffer[:, self.write_index:] = data[:, :first_part]
                self.buffer[:, :samples - first_part] = data[:, first_part:]
            
            self.write_index = (self.write_index + samples) % self.buffer_size
            self.samples_count = min(self.samples_count + samples, self.buffer_size)
            
        except Exception as e:
            self.logger.error(f"Buffer add_data failed: {e}")
    
    def get_latest_window(self) -> Optional[np.ndarray]:
        """
        Get the latest window of data for classification
        
        Returns:
            Latest window data (channels x window_size) or None if insufficient data
        """
        try:
            if self.samples_count < self.window_size:
                return None
            
            # Calculate start index for the latest window
            start_idx = (self.write_index - self.window_size) % self.buffer_size
            
            if start_idx + self.window_size <= self.buffer_size:
                # No wrap-around
                window = self.buffer[:, start_idx:start_idx + self.window_size].copy()
            else:
                # Handle wrap-around
                first_part = self.buffer_size - start_idx
                window = np.zeros((self.channels, self.window_size), dtype=np.float32)
                window[:, :first_part] = self.buffer[:, start_idx:]
                window[:, first_part:] = self.buffer[:, :self.window_size - first_part]
            
            return window
            
        except Exception as e:
            self.logger.error(f"Buffer get_latest_window failed: {e}")
            return None
    
    def clear(self) -> None:
        """Clear the buffer"""
        self.buffer.fill(0)
        self.write_index = 0
        self.samples_count = 0

# Configuration constants
MOTOR_IMAGERY_CLASSES = {
    0: 'Left Hand',
    1: 'Right Hand', 
    2: 'Foot',
    3: 'Tongue'
}

MOTOR_IMAGERY_CHANNELS = list(range(22))  # Channels 0-21

# Default preprocessing parameters
DEFAULT_PREPROCESSING_CONFIG = {
    'sampling_rate': 250,
    'bandpass_low': 8.0,
    'bandpass_high': 30.0,
    'filter_order': 4,
    'channels': 22,
    'window_size_seconds': 2.0,
    'chunk_size_seconds': 0.128  # ~32 samples at 250Hz
}