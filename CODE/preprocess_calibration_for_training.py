#!/usr/bin/env python3
"""
Preprocess personal calibration data to match BCI4_2B training distribution.

This script applies the same preprocessing as BCI4_2B to make calibration data
similar to what the model was trained on, improving transfer learning performance.

Key transformations:
- Notch filter (50 Hz powerline)
- Bandpass filter (4-40 Hz motor imagery band)
- Common Average Reference (CAR) spatial filtering
- Artifact removal (high amplitude detection)
- Z-score normalization per trial

Usage:
    python preprocess_calibration_for_training.py --calibration-file=calibration_data/alan_*.npz
"""

import numpy as np
import argparse
import logging
from pathlib import Path
from scipy import signal
import json
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CalibrationPreprocessor:
    """Preprocess calibration data to match BCI4_2B training conditions."""
    
    def __init__(self, fs: float = 250.0):
        """
        Initialize preprocessor.
        
        Args:
            fs: Sampling rate (Hz)
        """
        self.fs = fs
        self.nyquist = fs / 2
    
    def notch_filter(self, X: np.ndarray, notch_freq: float = 50.0, 
                     quality: float = 30.0) -> np.ndarray:
        """
        Apply notch filter to remove powerline noise.
        
        Args:
            X: Input signal (n_trials, n_channels, n_samples)
            notch_freq: Frequency to notch (Hz, default: 50 Hz)
            quality: Quality factor (higher = narrower)
        
        Returns:
            X_filtered: Notch-filtered signal
        """
        logger.info(f"Applying notch filter ({notch_freq} Hz)...")
        
        # Design notch filter
        w0 = notch_freq / self.nyquist
        b, a = signal.iirnotch(w0, quality)
        
        X_filtered = np.zeros_like(X)
        for trial_idx in range(X.shape[0]):
            for ch_idx in range(X.shape[1]):
                X_filtered[trial_idx, ch_idx, :] = signal.filtfilt(b, a, X[trial_idx, ch_idx, :])
        
        return X_filtered
    
    def bandpass_filter(self, X: np.ndarray, 
                       low_freq: float = 4.0, 
                       high_freq: float = 40.0,
                       order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to motor imagery frequency band.
        
        Standard motor imagery bands:
        - Mu (8-12 Hz): resting motor rhythm
        - Beta (18-30 Hz): active motor imagery
        - Combined: 4-40 Hz for safety
        
        Args:
            X: Input signal (n_trials, n_channels, n_samples)
            low_freq: Lower cutoff (Hz)
            high_freq: Upper cutoff (Hz)
            order: Filter order
        
        Returns:
            X_filtered: Bandpass-filtered signal
        """
        logger.info(f"Applying bandpass filter ({low_freq}-{high_freq} Hz, order={order})...")
        
        # Design bandpass filter
        sos = signal.butter(order, [low_freq, high_freq], btype='band', 
                           fs=self.fs, output='sos')
        
        X_filtered = np.zeros_like(X)
        for trial_idx in range(X.shape[0]):
            for ch_idx in range(X.shape[1]):
                X_filtered[trial_idx, ch_idx, :] = signal.sosfiltfilt(sos, X[trial_idx, ch_idx, :])
        
        return X_filtered
    
    def common_average_reference(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Common Average Reference (CAR) spatial filtering.
        
        CAR removes common noise across channels by subtracting 
        the average signal across all channels from each channel.
        
        This is standard preprocessing in BCI systems.
        
        Args:
            X: Input signal (n_trials, n_channels, n_samples)
        
        Returns:
            X_car: CAR-filtered signal
        """
        logger.info("Applying Common Average Reference (CAR)...")
        
        # Compute average across channels for each trial
        X_car = X.copy()
        for trial_idx in range(X.shape[0]):
            channel_mean = np.mean(X[trial_idx, :, :], axis=0, keepdims=True)
            X_car[trial_idx, :, :] = X[trial_idx, :, :] - channel_mean
        
        return X_car
    
    def remove_artifacts(self, X: np.ndarray, y: np.ndarray,
                        threshold_std: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove trials with high-amplitude artifacts.
        
        Criteria:
        - Any channel exceeding ±Nσ from baseline (default 5σ to be permissive)
        - Flat signal (no variance)
        - Extreme outliers
        
        Note: Threshold is set high (5σ) to avoid removing valid motor imagery data.
        Motor imagery can have high amplitude spikes that are normal.
        
        Args:
            X: Input signal (n_trials, n_channels, n_samples)
            y: Labels (n_trials,)
            threshold_std: Threshold in standard deviations (default: 5.0σ)
        
        Returns:
            X_clean, y_clean: Data with artifact trials removed
        """
        logger.info(f"Removing artifact trials (threshold={threshold_std}σ)...")
        
        n_trials_orig = X.shape[0]
        valid_trials = []
        
        for trial_idx in range(X.shape[0]):
            trial = X[trial_idx]
            
            # Check for flat signal (no variance) - strict criterion
            if np.any(np.std(trial, axis=1) < 1e-6):
                continue
            
            # Check for very high amplitudes - relaxed criterion with high threshold
            trial_mean = np.mean(trial)
            trial_std = np.std(trial)
            
            # Use threshold_std to determine if value is extreme
            if trial_std > 1e-6:
                if np.any(np.abs(trial - trial_mean) > threshold_std * trial_std):
                    continue
            
            # Check for NaN/Inf - strict criterion
            if np.any(~np.isfinite(trial)):
                continue
            
            valid_trials.append(trial_idx)
        
        X_clean = X[valid_trials]
        y_clean = y[valid_trials]
        
        n_removed = n_trials_orig - len(valid_trials)
        logger.info(f"  Removed {n_removed} artifact trials ({n_removed/n_trials_orig*100:.1f}%)")
        logger.info(f"  Remaining: {len(valid_trials)} trials")
        
        return X_clean, y_clean
    
    def zscore_normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization per trial, per channel.
        
        This normalizes amplitude differences between subjects and sessions.
        
        Args:
            X: Input signal (n_trials, n_channels, n_samples)
        
        Returns:
            X_normalized: Z-score normalized signal
        """
        logger.info("Applying z-score normalization...")
        
        X_normalized = np.zeros_like(X)
        for trial_idx in range(X.shape[0]):
            for ch_idx in range(X.shape[1]):
                channel_data = X[trial_idx, ch_idx, :]
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                X_normalized[trial_idx, ch_idx, :] = (channel_data - mean) / (std + 1e-8)
        
        return X_normalized
    
    def preprocess(self, X: np.ndarray, y: np.ndarray, 
                   apply_notch: bool = True,
                   apply_bandpass: bool = True,
                   apply_car: bool = True,
                   apply_artifact_removal: bool = True,
                   apply_zscore: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            X: Raw calibration data (n_trials, n_channels, n_samples)
            y: Labels (n_trials,)
            apply_notch: Apply 50 Hz notch filter
            apply_bandpass: Apply motor imaging bandpass filter
            apply_car: Apply Common Average Reference
            apply_artifact_removal: Remove artifact trials
            apply_zscore: Apply z-score normalization
        
        Returns:
            X_processed, y_processed: Preprocessed data
            stats: Preprocessing statistics
        """
        print("\n" + "="*70)
        print("CALIBRATION DATA PREPROCESSING (BCI4_2B-MATCHED)")
        print("="*70 + "\n")
        
        logger.info(f"Starting preprocessing pipeline...")
        logger.info(f"  Input shape: {X.shape}")
        logger.info(f"  Sampling rate: {self.fs} Hz")
        logger.info(f"  Classes: LEFT={np.sum(y==0)}, RIGHT={np.sum(y==1)}")
        
        stats = {
            'original_shape': list(X.shape),
            'original_n_trials': X.shape[0],
            'original_class_distribution': {
                'LEFT': int(np.sum(y == 0)),
                'RIGHT': int(np.sum(y == 1))
            },
            'steps_applied': []
        }
        
        X_processed = X.copy().astype(np.float64)
        y_processed = y.copy()
        
        # Step 1: Notch filter
        if apply_notch:
            logger.info("\n[Step 1/5] 50 Hz Notch Filter")
            X_processed = self.notch_filter(X_processed)
            stats['steps_applied'].append('notch_50hz')
        
        # Step 2: Bandpass filter
        if apply_bandpass:
            logger.info("\n[Step 2/5] 4-40 Hz Bandpass Filter")
            X_processed = self.bandpass_filter(X_processed, low_freq=4.0, high_freq=40.0)
            stats['steps_applied'].append('bandpass_4_40hz')
        
        # Step 3: CAR spatial filtering
        if apply_car:
            logger.info("\n[Step 3/5] Common Average Reference (CAR)")
            X_processed = self.common_average_reference(X_processed)
            stats['steps_applied'].append('car')
        
        # Step 4: Artifact removal
        if apply_artifact_removal:
            logger.info("\n[Step 4/5] Artifact Removal")
            X_processed, y_processed = self.remove_artifacts(X_processed, y_processed, threshold_std=5.0)
            stats['steps_applied'].append('artifact_removal')
            stats['artifacts_removed'] = X.shape[0] - X_processed.shape[0]
        
        # Step 5: Z-score normalization
        if apply_zscore:
            logger.info("\n[Step 5/5] Z-Score Normalization")
            X_processed = self.zscore_normalize(X_processed)
            stats['steps_applied'].append('zscore_normalization')
        
        logger.info(f"\n" + "="*70)
        logger.info(f"✅ PREPROCESSING COMPLETE")
        logger.info(f"="*70)
        logger.info(f"  Original shape: {stats['original_shape']}")
        logger.info(f"  Final shape: {list(X_processed.shape)}")
        logger.info(f"  Trials retained: {len(y_processed)} ({len(y_processed)/stats['original_n_trials']*100:.1f}%)")
        logger.info(f"  Final class distribution:")
        logger.info(f"    LEFT:  {np.sum(y_processed == 0)}")
        logger.info(f"    RIGHT: {np.sum(y_processed == 1)}")
        logger.info(f"  Preprocessing steps: {len(stats['steps_applied'])}")
        for i, step in enumerate(stats['steps_applied'], 1):
            logger.info(f"    {i}. {step}")
        
        stats['final_shape'] = list(X_processed.shape)
        stats['final_n_trials'] = X_processed.shape[0]
        stats['final_class_distribution'] = {
            'LEFT': int(np.sum(y_processed == 0)),
            'RIGHT': int(np.sum(y_processed == 1))
        }
        stats['retention_rate'] = float(len(y_processed) / stats['original_n_trials'])
        
        print("="*70 + "\n")
        
        return X_processed, y_processed, stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Preprocess calibration data to match BCI4_2B training conditions'
    )
    parser.add_argument(
        '--calibration-file',
        type=str,
        required=True,
        help='Path to calibration NPZ file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='calibration_data_preprocessed',
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--no-notch',
        action='store_true',
        help='Skip 50 Hz notch filter'
    )
    parser.add_argument(
        '--no-bandpass',
        action='store_true',
        help='Skip bandpass filter'
    )
    parser.add_argument(
        '--no-car',
        action='store_true',
        help='Skip Common Average Reference'
    )
    parser.add_argument(
        '--artifact-removal',
        action='store_true',
        help='Enable artifact removal (disabled by default for motor imagery data)'
    )
    parser.add_argument(
        '--no-zscore',
        action='store_true',
        help='Skip z-score normalization'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    calibration_path = Path(args.calibration_file)
    if not calibration_path.exists():
        print(f"❌ Calibration file not found: {args.calibration_file}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load calibration data
    logger.info(f"Loading calibration data: {calibration_path.name}")
    try:
        data = np.load(calibration_path)
        
        # Handle both formats:
        # Format 1: X and y arrays
        if 'X' in data and 'y' in data:
            X = data['X']  # (n_trials, n_channels, n_samples)
            y = data['y']  # (n_trials,)
            rest_epochs = None
        
        # Format 2: epochs_left, epochs_right, epochs_rest (from NPG recording)
        elif 'epochs_left' in data and 'epochs_right' in data:
            epochs_left = data['epochs_left']   # (n_left, samples, channels)
            epochs_right = data['epochs_right'] # (n_right, samples, channels)
            rest_epochs = data.get('epochs_rest', None)
            
            # Convert to (n_trials, channels, samples) format
            X_left = np.transpose(epochs_left, (0, 2, 1))  # (n_left, channels, samples)
            X_right = np.transpose(epochs_right, (0, 2, 1))  # (n_right, channels, samples)
            
            # Combine with labels
            X = np.vstack([X_left, X_right])
            y = np.hstack([np.zeros(len(X_left), dtype=int), np.ones(len(X_right), dtype=int)])
            
            # Convert rest epochs if present
            if rest_epochs is not None:
                rest_epochs = np.transpose(rest_epochs, (0, 2, 1))
        else:
            raise ValueError(f"Unknown NPZ format. Keys: {list(data.keys())}")
    
    except Exception as e:
        logger.error(f"Failed to load calibration file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info(f"  Loaded shape: {X.shape}")
    logger.info(f"  Class distribution: LEFT={np.sum(y==0)}, RIGHT={np.sum(y==1)}")
    if rest_epochs is not None:
        logger.info(f"  REST epochs: {len(rest_epochs)}")
    
    # Preprocess
    preprocessor = CalibrationPreprocessor(fs=250.0)
    try:
        X_proc, y_proc, stats = preprocessor.preprocess(
            X, y,
            apply_notch=not args.no_notch,
            apply_bandpass=not args.no_bandpass,
            apply_car=not args.no_car,
            apply_artifact_removal=args.artifact_removal,  # Disabled by default
            apply_zscore=not args.no_zscore
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save preprocessed data
    output_file = output_dir / calibration_path.name.replace('.npz', '_preprocessed.npz')
    try:
        np.savez_compressed(str(output_file), X=X_proc, y=y_proc)
        logger.info(f"💾 Saved preprocessed data: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save preprocessed data: {e}")
        return 1
    
    # Save statistics
    stats_file = output_dir / calibration_path.name.replace('.npz', '_preprocess_stats.json')
    try:
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"💾 Saved statistics: {stats_file}")
    except Exception as e:
        logger.error(f"Failed to save statistics: {e}")
        return 1
    
    # Print next steps
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"\n1. Fine-tune with preprocessed data:")
    print(f"   python fine_tune_personal_model.py \\")
    print(f"     --calibration-file={output_file} \\")
    print(f"     --strategy=full")
    print(f"\n2. Compare results:")
    print(f"   Original: {args.calibration_file}")
    print(f"   Preprocessed: {output_file}")
    print(f"   (Check if class separability improved)")
    print("\n3. Re-record calibration with ACTUAL HAND MOVEMENT:")
    print(f"   - Move/squeeze your hand during LEFT trials")
    print(f"   - Move/squeeze your hand during RIGHT trials")
    print(f"   - Collect 150-200 trials per class")
    print(f"   - Then preprocess and fine-tune again")
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
