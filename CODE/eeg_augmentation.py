"""
Data Augmentation for EEG Motor Imagery
Implements various augmentation techniques to improve model generalization
"""

import numpy as np
import random
from scipy import signal
from scipy.interpolate import interp1d
import logging


class EEGAugmentationPipeline:
    """Comprehensive EEG data augmentation pipeline."""
    
    def __init__(self, 
                 time_shift_range=0.05,      # ±50ms time shifts
                 noise_level=0.1,            # 10% noise
                 amplitude_range=(0.8, 1.2), # 80%-120% amplitude
                 time_warp_range=(0.95, 1.05), # 95%-105% time stretching
                 frequency_shift_range=2.0,   # ±2Hz frequency shifting
                 apply_probability=0.5):      # 50% chance to apply each augmentation
        
        self.time_shift_range = time_shift_range
        self.noise_level = noise_level
        self.amplitude_range = amplitude_range
        self.time_warp_range = time_warp_range
        self.frequency_shift_range = frequency_shift_range
        self.apply_probability = apply_probability
        
        self.logger = logging.getLogger(__name__)
    
    def time_shift(self, data, sampling_rate=250):
        """
        Apply random time shifts to EEG data.
        
        Args:
            data: EEG data (epochs, timepoints, channels)
            sampling_rate: Sampling rate in Hz
        
        Returns:
            Augmented data
        """
        if random.random() > self.apply_probability:
            return data
            
        epochs, timepoints, channels = data.shape
        max_shift_samples = int(self.time_shift_range * sampling_rate)
        
        augmented_data = np.zeros_like(data)
        
        for epoch in range(epochs):
            # Random shift for this epoch
            shift_samples = random.randint(-max_shift_samples, max_shift_samples)
            
            for channel in range(channels):
                if shift_samples > 0:
                    # Shift right (delay)
                    augmented_data[epoch, shift_samples:, channel] = data[epoch, :-shift_samples, channel]
                    # Pad beginning with first value
                    augmented_data[epoch, :shift_samples, channel] = data[epoch, 0, channel]
                elif shift_samples < 0:
                    # Shift left (advance)
                    augmented_data[epoch, :shift_samples, channel] = data[epoch, -shift_samples:, channel]
                    # Pad end with last value
                    augmented_data[epoch, shift_samples:, channel] = data[epoch, -1, channel]
                else:
                    # No shift
                    augmented_data[epoch, :, channel] = data[epoch, :, channel]
        
        return augmented_data
    
    def add_noise(self, data):
        """
        Add Gaussian noise to EEG data.
        
        Args:
            data: EEG data (epochs, timepoints, channels)
            
        Returns:
            Noisy data
        """
        if random.random() > self.apply_probability:
            return data
            
        # Calculate noise level based on signal standard deviation
        signal_std = np.std(data, axis=1, keepdims=True)  # Per epoch and channel
        noise = np.random.normal(0, signal_std * self.noise_level, data.shape)
        
        return data + noise
    
    def amplitude_scaling(self, data):
        """
        Apply random amplitude scaling to EEG data.
        
        Args:
            data: EEG data (epochs, timepoints, channels)
            
        Returns:
            Scaled data
        """
        if random.random() > self.apply_probability:
            return data
            
        epochs = data.shape[0]
        scales = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], (epochs, 1, 1))
        
        return data * scales
    
    def time_warping(self, data):
        """
        Apply time warping (stretching/compression) to EEG data.
        
        Args:
            data: EEG data (epochs, timepoints, channels)
            
        Returns:
            Time-warped data
        """
        if random.random() > self.apply_probability:
            return data
            
        epochs, timepoints, channels = data.shape
        augmented_data = np.zeros_like(data)
        
        for epoch in range(epochs):
            # Random warping factor for this epoch
            warp_factor = random.uniform(self.time_warp_range[0], self.time_warp_range[1])
            
            # Create time indices for warping
            original_indices = np.arange(timepoints)
            warped_length = int(timepoints * warp_factor)
            
            if warped_length <= 1:
                # If too compressed, just copy original
                augmented_data[epoch] = data[epoch]
                continue
            
            warped_indices = np.linspace(0, timepoints - 1, warped_length)
            
            for channel in range(channels):
                # Interpolate to create warped signal
                interp_func = interp1d(original_indices, data[epoch, :, channel], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                warped_signal = interp_func(warped_indices)
                
                # Resample back to original length
                if warped_length < timepoints:
                    # Pad if compressed
                    pad_length = timepoints - warped_length
                    warped_signal = np.pad(warped_signal, (0, pad_length), mode='edge')
                elif warped_length > timepoints:
                    # Truncate if stretched
                    warped_signal = warped_signal[:timepoints]
                
                augmented_data[epoch, :, channel] = warped_signal
        
        return augmented_data
    
    def frequency_shift(self, data, sampling_rate=250):
        """
        Apply random frequency shifting using modulation.
        
        Args:
            data: EEG data (epochs, timepoints, channels)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Frequency-shifted data
        """
        if random.random() > self.apply_probability:
            return data
            
        epochs, timepoints, channels = data.shape
        
        # Create time vector
        t = np.arange(timepoints) / sampling_rate
        
        augmented_data = np.zeros_like(data)
        
        for epoch in range(epochs):
            # Random frequency shift for this epoch
            freq_shift = random.uniform(-self.frequency_shift_range, self.frequency_shift_range)
            
            # Create modulation signal
            modulation = np.exp(1j * 2 * np.pi * freq_shift * t)
            
            for channel in range(channels):
                # Apply frequency shift via modulation in frequency domain
                signal_fft = np.fft.fft(data[epoch, :, channel])
                
                # Shift in frequency domain (circular shift)
                shift_bins = int(freq_shift * timepoints / sampling_rate)
                signal_fft_shifted = np.roll(signal_fft, shift_bins)
                
                # Convert back to time domain
                shifted_signal = np.fft.ifft(signal_fft_shifted).real
                augmented_data[epoch, :, channel] = shifted_signal
        
        return augmented_data
    
    def random_channel_dropout(self, data, dropout_probability=0.1):
        """
        Randomly set some channels to zero (channel dropout).
        
        Args:
            data: EEG data (epochs, timepoints, channels)
            dropout_probability: Probability of dropping each channel
            
        Returns:
            Data with some channels dropped
        """
        if random.random() > self.apply_probability:
            return data
            
        epochs, timepoints, channels = data.shape
        augmented_data = data.copy()
        
        for epoch in range(epochs):
            for channel in range(channels):
                if random.random() < dropout_probability:
                    augmented_data[epoch, :, channel] = 0
        
        return augmented_data
    
    def apply_augmentation(self, data, sampling_rate=250, augmentation_factor=2):
        """
        Apply full augmentation pipeline to create multiple versions of data.
        
        Args:
            data: Original EEG data (epochs, timepoints, channels)
            sampling_rate: Sampling rate in Hz
            augmentation_factor: How many augmented versions to create per original epoch
            
        Returns:
            Augmented data with shape (epochs * (1 + augmentation_factor), timepoints, channels)
        """
        original_epochs, timepoints, channels = data.shape
        
        # Start with original data
        all_data = [data]
        
        self.logger.info(f"Applying data augmentation with factor {augmentation_factor}")
        
        for aug_round in range(augmentation_factor):
            augmented = data.copy()
            
            # Apply augmentations in sequence
            augmented = self.time_shift(augmented, sampling_rate)
            augmented = self.add_noise(augmented)
            augmented = self.amplitude_scaling(augmented)
            augmented = self.time_warping(augmented)
            augmented = self.frequency_shift(augmented, sampling_rate)
            augmented = self.random_channel_dropout(augmented)
            
            all_data.append(augmented)
        
        # Concatenate all versions
        final_data = np.concatenate(all_data, axis=0)
        
        self.logger.info(f"Augmentation complete: {original_epochs} → {final_data.shape[0]} epochs")
        
        return final_data
    
    def augment_with_labels(self, data, labels, sampling_rate=250, augmentation_factor=2):
        """
        Apply augmentation while preserving labels.
        
        Args:
            data: EEG data (epochs, timepoints, channels)
            labels: Corresponding labels
            sampling_rate: Sampling rate in Hz
            augmentation_factor: Augmentation factor
            
        Returns:
            Tuple of (augmented_data, augmented_labels)
        """
        augmented_data = self.apply_augmentation(data, sampling_rate, augmentation_factor)
        
        # Replicate labels for augmented data
        original_epochs = data.shape[0]
        augmented_labels = np.tile(labels, augmentation_factor + 1)
        
        return augmented_data, augmented_labels


def demo_augmentation():
    """Demonstrate augmentation effects on sample data."""
    import matplotlib.pyplot as plt
    
    # Generate sample EEG-like data
    epochs, timepoints, channels = 5, 1000, 3
    t = np.linspace(0, 4, timepoints)  # 4 seconds
    
    sample_data = np.zeros((epochs, timepoints, channels))
    
    # Create synthetic EEG with different frequency components per channel
    for epoch in range(epochs):
        for channel in range(channels):
            # Mix of alpha (10Hz), beta (20Hz), and noise
            signal = (np.sin(2 * np.pi * 10 * t) * 0.5 +
                     np.sin(2 * np.pi * 20 * t) * 0.3 +
                     np.random.normal(0, 0.1, timepoints))
            sample_data[epoch, :, channel] = signal
    
    # Apply augmentation
    augmenter = EEGAugmentationPipeline(apply_probability=1.0)  # Always apply for demo
    augmented_data = augmenter.apply_augmentation(sample_data, augmentation_factor=1)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for channel in range(channels):
        # Original data
        axes[0, channel].plot(t, sample_data[0, :, channel], 'b-', label='Original')
        axes[0, channel].set_title(f'Original - Channel {channel+1}')
        axes[0, channel].set_xlabel('Time (s)')
        axes[0, channel].set_ylabel('Amplitude')
        
        # Augmented data
        axes[1, channel].plot(t, augmented_data[epochs, :, channel], 'r-', label='Augmented')
        axes[1, channel].set_title(f'Augmented - Channel {channel+1}')
        axes[1, channel].set_xlabel('Time (s)')
        axes[1, channel].set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig('augmentation_demo.png', dpi=300)
    plt.show()
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Augmented data shape: {augmented_data.shape}")


if __name__ == "__main__":
    demo_augmentation()