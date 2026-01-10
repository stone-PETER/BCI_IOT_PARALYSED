#!/usr/bin/env python3
"""
Real-time EEG Data Simulator
Simulates real-time EEG streaming using recorded BCI datasets
"""

import numpy as np
import threading
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, List
import sys
import os

# Add CODE directory to path for imports
code_dir = Path(__file__).parent.parent / 'CODE'
sys.path.append(str(code_dir))

from gdf_data_loader import GDFDataLoader
from bci4_2a_loader import BCI4_2A_Loader
from preProcessing import MOTOR_IMAGERY_CLASSES

class EEGDataSimulator:
    """
    Real-time EEG data simulator using recorded BCI datasets
    """
    
    def __init__(self, 
                 sampling_rate: int = 250,
                 chunk_size: int = 32):
        """
        Initialize EEG data simulator
        
        Args:
            sampling_rate: Target sampling rate in Hz
            chunk_size: Number of samples per streaming chunk
        """
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self.chunk_interval = chunk_size / sampling_rate  # Time between chunks
        
        # Data storage
        self.datasets = {}
        self.current_class = 0  # Default to Left Hand
        self.current_epoch_index = 0
        self.current_sample_index = 0
        
        # Streaming control
        self.is_streaming = False
        self.stream_thread = None
        self.data_callback = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load BCI datasets
        self._load_datasets()
        
        self.logger.info(f"EEG Simulator initialized: {sampling_rate}Hz, "
                        f"Chunk size: {chunk_size} samples")
    
    def _load_datasets(self):
        """Load BCI competition datasets for simulation"""
        try:
            self.logger.info("Loading BCI datasets for simulation...")
            
            # Load BCI Competition III Dataset 3a
            try:
                gdf_loader = GDFDataLoader()
                bci3_data, bci3_labels = gdf_loader.load_data()
                
                # Organize by class
                for class_idx in range(4):
                    class_mask = bci3_labels == class_idx
                    if np.any(class_mask):
                        if class_idx not in self.datasets:
                            self.datasets[class_idx] = []
                        
                        class_epochs = bci3_data[class_mask]
                        for epoch in class_epochs:
                            # Transpose to (channels, samples) format
                            if epoch.shape[0] == 1000:  # (samples, channels)
                                epoch = epoch.T  # (channels, samples)
                            self.datasets[class_idx].append(epoch)
                
                self.logger.info(f"BCI III 3a loaded: {len(bci3_data)} epochs")
                
            except Exception as e:
                self.logger.warning(f"Could not load BCI III 3a: {e}")
            
            # Load BCI Competition IV Dataset 2a
            try:
                bci4_loader = BCI4_2A_Loader()
                
                # Load a few subjects for variety
                for subject in ['A01', 'A02', 'A03']:
                    try:
                        subject_data, subject_labels = bci4_loader.load_subject_data(subject, 'T')
                        
                        # Organize by class
                        for class_idx in range(4):
                            class_mask = subject_labels == class_idx
                            if np.any(class_mask):
                                if class_idx not in self.datasets:
                                    self.datasets[class_idx] = []
                                
                                class_epochs = subject_data[class_mask]
                                for epoch in class_epochs:
                                    # Transpose to (channels, samples) format
                                    if epoch.shape[0] == 1000:  # (samples, channels)
                                        epoch = epoch.T  # (channels, samples)
                                    self.datasets[class_idx].append(epoch)
                        
                        self.logger.info(f"BCI IV 2a {subject} loaded: {len(subject_data)} epochs")
                        
                    except Exception as e:
                        self.logger.warning(f"Could not load BCI IV 2a {subject}: {e}")
                
            except Exception as e:
                self.logger.warning(f"Could not load BCI IV 2a: {e}")
            
            # Create synthetic data if no real data available
            if not self.datasets:
                self.logger.warning("No real data loaded, creating synthetic data")
                self._create_synthetic_data()
            
            # Report loaded data
            for class_idx, epochs in self.datasets.items():
                class_name = MOTOR_IMAGERY_CLASSES.get(class_idx, f"Class {class_idx}")
                self.logger.info(f"{class_name}: {len(epochs)} epochs available")
                
        except Exception as e:
            self.logger.error(f"Dataset loading failed: {e}")
            self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic EEG data for testing"""
        self.logger.info("Creating synthetic EEG data...")
        
        np.random.seed(42)  # For reproducible synthetic data
        
        for class_idx in range(4):
            self.datasets[class_idx] = []
            
            # Create 10 synthetic epochs per class
            for _ in range(10):
                # Generate synthetic EEG-like signals
                epoch = self._generate_synthetic_eeg_epoch(class_idx)
                self.datasets[class_idx].append(epoch)
        
        self.logger.info("Synthetic EEG data created")
    
    def _generate_synthetic_eeg_epoch(self, class_idx: int) -> np.ndarray:
        """
        Generate a synthetic EEG epoch with class-specific patterns
        
        Args:
            class_idx: Motor imagery class index (0-3)
            
        Returns:
            Synthetic EEG epoch (22 channels x 1000 samples)
        """
        # Base parameters
        channels = 22
        samples = 1000
        time_vec = np.linspace(0, 4, samples)  # 4 seconds
        
        # Initialize with baseline noise
        epoch = np.random.normal(0, 5, (channels, samples)).astype(np.float32)
        
        # Add class-specific patterns
        if class_idx == 0:  # Left Hand
            # Add mu rhythm suppression in C3 area (channels 9-11)
            for ch in [9, 10, 11]:
                mu_suppression = -10 * np.sin(2 * np.pi * 10 * time_vec) * np.exp(-0.5 * time_vec)
                epoch[ch] += mu_suppression
                
        elif class_idx == 1:  # Right Hand
            # Add mu rhythm suppression in C4 area (channels 12-14)
            for ch in [12, 13, 14]:
                mu_suppression = -10 * np.sin(2 * np.pi * 10 * time_vec) * np.exp(-0.5 * time_vec)
                epoch[ch] += mu_suppression
                
        elif class_idx == 2:  # Foot
            # Add patterns in Cz area (channels 10-12)
            for ch in [10, 11, 12]:
                foot_pattern = 8 * np.sin(2 * np.pi * 12 * time_vec) * (1 + 0.5 * np.sin(2 * np.pi * 2 * time_vec))
                epoch[ch] += foot_pattern
                
        elif class_idx == 3:  # Tongue
            # Add patterns in central areas (channels 8-15)
            for ch in range(8, 16):
                tongue_pattern = 6 * np.sin(2 * np.pi * 15 * time_vec + np.pi * ch / 8)
                epoch[ch] += tongue_pattern
        
        # Add common alpha rhythm (8-13 Hz) in occipital channels
        for ch in [18, 19, 20, 21]:
            alpha_freq = 8 + 5 * np.random.random()  # 8-13 Hz
            alpha_amplitude = 15 + 10 * np.random.random()
            alpha_wave = alpha_amplitude * np.sin(2 * np.pi * alpha_freq * time_vec)
            epoch[ch] += alpha_wave
        
        return epoch
    
    def set_motor_imagery_class(self, class_idx: int):
        """
        Set the current motor imagery class for simulation
        
        Args:
            class_idx: Class index (0=Left Hand, 1=Right Hand, 2=Foot, 3=Tongue)
        """
        if class_idx in self.datasets:
            self.current_class = class_idx
            self.current_epoch_index = 0
            self.current_sample_index = 0
            
            class_name = MOTOR_IMAGERY_CLASSES.get(class_idx, f"Class {class_idx}")
            self.logger.info(f"Switched to motor imagery class: {class_name}")
        else:
            self.logger.error(f"Class {class_idx} not available in datasets")
    
    def start_streaming(self, data_callback: Callable[[np.ndarray, int], None]):
        """
        Start real-time EEG streaming
        
        Args:
            data_callback: Callback function that receives (data_chunk, true_class)
        """
        if self.is_streaming:
            self.logger.warning("Streaming already active")
            return
        
        self.data_callback = data_callback
        self.is_streaming = True
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        
        self.logger.info("EEG streaming started")
    
    def stop_streaming(self):
        """Stop real-time EEG streaming"""
        self.is_streaming = False
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.0)
        
        self.logger.info("EEG streaming stopped")
    
    def _stream_worker(self):
        """Worker thread for streaming EEG data"""
        try:
            while self.is_streaming:
                # Get current epoch data
                if self.current_class not in self.datasets:
                    time.sleep(self.chunk_interval)
                    continue
                
                epochs = self.datasets[self.current_class]
                if not epochs:
                    time.sleep(self.chunk_interval)
                    continue
                
                # Get current epoch
                epoch = epochs[self.current_epoch_index]
                
                # Extract chunk
                start_idx = self.current_sample_index
                end_idx = min(start_idx + self.chunk_size, epoch.shape[1])
                
                if start_idx >= epoch.shape[1]:
                    # Move to next epoch
                    self.current_epoch_index = (self.current_epoch_index + 1) % len(epochs)
                    self.current_sample_index = 0
                    continue
                
                # Get data chunk
                chunk = epoch[:, start_idx:end_idx].copy()
                
                # Handle end of epoch
                if end_idx == epoch.shape[1]:
                    self.current_epoch_index = (self.current_epoch_index + 1) % len(epochs)
                    self.current_sample_index = 0
                else:
                    self.current_sample_index = end_idx
                
                # Send data through callback
                if self.data_callback:
                    self.data_callback(chunk, self.current_class)
                
                # Wait for next chunk
                time.sleep(self.chunk_interval)
                
        except Exception as e:
            self.logger.error(f"Streaming worker error: {e}")
            self.is_streaming = False
    
    def get_available_classes(self) -> Dict[int, str]:
        """
        Get available motor imagery classes
        
        Returns:
            Dictionary mapping class indices to class names
        """
        available = {}
        for class_idx in self.datasets:
            if self.datasets[class_idx]:  # Has data
                available[class_idx] = MOTOR_IMAGERY_CLASSES.get(class_idx, f"Class {class_idx}")
        
        return available
    
    def get_class_epoch_count(self, class_idx: int) -> int:
        """
        Get number of available epochs for a class
        
        Args:
            class_idx: Class index
            
        Returns:
            Number of epochs available
        """
        if class_idx in self.datasets:
            return len(self.datasets[class_idx])
        return 0

# Convenience function for quick testing
def test_simulator():
    """Test the EEG simulator"""
    logging.basicConfig(level=logging.INFO)
    
    def data_handler(chunk, true_class):
        print(f"Received chunk: {chunk.shape}, True class: {MOTOR_IMAGERY_CLASSES[true_class]}")
    
    simulator = EEGDataSimulator()
    
    print("Available classes:", simulator.get_available_classes())
    
    # Test streaming
    simulator.set_motor_imagery_class(0)  # Left Hand
    simulator.start_streaming(data_handler)
    
    time.sleep(5)  # Stream for 5 seconds
    
    simulator.stop_streaming()
    print("Test completed")

if __name__ == "__main__":
    test_simulator()