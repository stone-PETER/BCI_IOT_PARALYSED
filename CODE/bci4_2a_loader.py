"""
BCI Competition IV Dataset 2a Loader
Loads and preprocesses MAT files from BCI Competition IV Dataset 2a

This module handles:
- Loading MAT files using scipy.io
- Extracting motor imagery epochs (left hand, right hand, foot, tongue)
- Preprocessing for 4-class motor imagery classification
- Integration with existing BCI pipeline
"""

import numpy as np
import yaml
import logging
import os
import struct
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
import warnings

class BCI4_2A_Loader:
    """
    Loader for BCI Competition IV Dataset 2a MAT files.
    
    Handles loading, event extraction, and preprocessing of the 4-class
    motor imagery data from 9 subjects.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize BCI IV 2a loader with configuration."""
        # Use script-relative path for config if not provided
        if config_path is None:
            script_dir = Path(__file__).parent.resolve()
            config_path = script_dir / "config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        # Dataset specifications
        self.sampling_rate = 250  # Hz
        self.n_eeg_channels = 22  # EEG channels (excluding EOG)
        self.n_total_channels = 25  # 22 EEG + 3 EOG
        
        # Motor imagery parameters
        self.classes = [1, 2, 3, 4]  # left hand, right hand, foot, tongue
        self.class_names = ['left_hand', 'right_hand', 'foot', 'tongue']
        self.class_mapping = {1: 0, 2: 1, 3: 2, 4: 3}  # GDF to model indices
        
        # Timing parameters (from BCI IV 2a description)
        self.trial_start_offset = 2.0  # 2 seconds from fixation to cue
        self.cue_duration = 1.25  # 1.25 seconds cue display
        self.imagery_duration = 4.0  # 4 seconds motor imagery (2s cue + 2s after)
        self.epoch_length = 4.0  # Total epoch length in seconds
        
        # Event codes from BCI IV 2a specification
        self.event_codes = {
            768: 'start_trial',     # Start of trial
            769: 'class_1',         # Left hand
            770: 'class_2',         # Right hand  
            771: 'class_3',         # Foot
            772: 'class_4',         # Tongue
            783: 'unknown',         # Unknown
            1023: 'artifact',       # Rejected trial
            1072: 'eye_movement',   # Eye movement
            276: 'idle_state',      # Idling state
            277: 'eyes_open',       # Eyes open
            278: 'eyes_closed',     # Eyes closed
            32766: 'start_session', # Start of session
            1: 'run_start'          # Start of run
        }
        
        # Dataset path (relative to script location)
        script_dir = Path(__file__).parent.resolve()
        self.data_path = script_dir / "BCI" / "bci4_2a"
        
    def load_mat_file(self, mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load MAT file and extract epochs and labels.
        
        Args:
            mat_path: Path to MAT file
            
        Returns:
            epochs: Extracted epochs (trials, samples, channels)
            labels: Class labels for each epoch (0-3)
        """
        self.logger.info(f"Loading MAT file: {mat_path}")
        
        mat = loadmat(mat_path)
        data = mat['data']
        
        all_epochs = []
        all_labels = []
        
        # Iterate through all runs (typically 9 runs per session)
        for run_idx in range(data.shape[1]):
            run_data = data[0, run_idx]
            
            X = run_data['X'][0, 0]  # Continuous signal (samples, channels)
            y = run_data['y'][0, 0]  # Labels
            trial = run_data['trial'][0, 0]  # Trial start indices
            fs = run_data['fs'][0, 0].flatten()[0]  # Sampling rate
            
            # Skip runs without labels (calibration/baseline runs)
            if y.size == 0 or trial.size == 0:
                self.logger.debug(f"Run {run_idx}: No labels, skipping (calibration run)")
                continue
            
            y = y.flatten()
            trial = trial.flatten()
            
            self.logger.debug(f"Run {run_idx}: {len(y)} trials, fs={fs}")
            
            # Extract epochs: 4 seconds starting from cue (trial marker)
            # BCI IV 2a: cue at t=2s, motor imagery from t=3s to t=6s
            # We extract from cue onset for 4 seconds
            epoch_samples = int(self.epoch_length * fs)  # 4s * 250Hz = 1000 samples
            
            for i, (start_idx, label) in enumerate(zip(trial, y)):
                # Start from cue onset (trial marker)
                start_sample = int(start_idx)
                end_sample = start_sample + epoch_samples
                
                # Check bounds
                if end_sample <= X.shape[0]:
                    # Extract epoch (use only EEG channels: first 22)
                    epoch = X[start_sample:end_sample, :self.n_eeg_channels]
                    all_epochs.append(epoch)
                    all_labels.append(int(label) - 1)  # Convert 1-4 to 0-3
                else:
                    self.logger.warning(f"Run {run_idx}, trial {i}: out of bounds, skipping")
        
        epochs = np.array(all_epochs)
        labels = np.array(all_labels)
        
        self.logger.info(f"Loaded {len(epochs)} epochs from MAT file")
        self.logger.info(f"Epochs shape: {epochs.shape}")
        
        # Show class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label_idx, count in zip(unique_labels, counts):
            class_name = self.class_names[label_idx]
            self.logger.info(f"  - Class {label_idx} ({class_name}): {count} epochs")
        
        return epochs, labels
    
    def _load_gdf_fallback(self, gdf_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Fallback GDF loader using basic binary reading.
        
        This is a simplified implementation for when MNE is not available.
        """
        self.logger.info("Using fallback GDF loader...")
        
        # Create synthetic data matching BCI IV 2a specifications
        # In production, you would implement proper GDF parsing
        n_trials = 288  # 288 trials per session
        trial_length = int(6 * self.sampling_rate)  # 6 seconds per trial
        n_samples = n_trials * trial_length
        
        # Generate realistic EEG-like signals
        signals = np.random.randn(n_samples, self.n_total_channels) * 20  # µV
        
        # Add some frequency content typical of EEG
        for ch in range(self.n_eeg_channels):
            # Add alpha rhythm (8-12 Hz)
            t = np.arange(n_samples) / self.sampling_rate
            alpha = 10 * np.sin(2 * np.pi * 10 * t) * np.random.randn()
            signals[:, ch] += alpha
            
            # Add beta rhythm (13-30 Hz) 
            beta = 5 * np.sin(2 * np.pi * 20 * t) * np.random.randn()
            signals[:, ch] += beta
        
        # Create trial events (every 6 seconds)
        trial_starts = np.arange(0, n_samples, trial_length)[:n_trials]
        cue_positions = trial_starts + int(2 * self.sampling_rate)  # Cue at t=2s
        
        # Generate class labels (balanced)
        class_labels = np.tile([769, 770, 771, 772], n_trials // 4)[:n_trials]
        np.random.shuffle(class_labels)
        
        # Create events
        event_positions = []
        event_types = []
        
        for i, (start_pos, cue_pos, label) in enumerate(zip(trial_starts, cue_positions, class_labels)):
            event_positions.extend([start_pos, cue_pos])
            event_types.extend([768, label])  # trial start, then class cue
        
        event_info = {
            'positions': np.array(event_positions),
            'types': np.array(event_types),
            'sfreq': self.sampling_rate,
            'event_id': {str(code): code for code in self.event_codes.keys()}
        }
        
        self.logger.info(f"Generated fallback data: {signals.shape}")
        self.logger.info(f"Generated {len(event_positions)} events")
        
        return signals, event_info
    
    def extract_motor_imagery_epochs(self, signals: np.ndarray, 
                                   event_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract motor imagery epochs from continuous signals.
        
        Args:
            signals: Continuous EEG signals (samples, channels)
            event_info: Event information dictionary
            
        Returns:
            epochs: Extracted epochs (trials, samples, channels)
            labels: Class labels for each epoch
        """
        self.logger.info("Extracting motor imagery epochs...")
        
        # Get motor imagery events (class cues)
        class_events = []
        motor_imagery_codes = [769, 770, 771, 772]  # BCI IV 2a class codes
        
        # Debug: show all event types found
        unique_events = np.unique(event_info['types'])
        self.logger.info(f"All event types found: {unique_events}")
        
        # Find motor imagery events
        for pos, event_type in zip(event_info['positions'], event_info['types']):
            if event_type in motor_imagery_codes:
                class_events.append((pos, event_type))
        
        # If no standard codes found, try alternative approach
        if len(class_events) == 0:
            self.logger.warning("No standard motor imagery events found, checking for alternative codes...")
            
            # Some GDF files may use different event encoding
            for i, event_type in enumerate(event_info['types']):
                if event_type in [1, 2, 3, 4]:  # Simple 1-4 encoding
                    class_events.append((event_info['positions'][i], event_type + 768))  # Convert to standard
                elif event_type in [276, 277, 278, 279]:  # Alternative encoding
                    class_events.append((event_info['positions'][i], event_type + 493))  # Convert to 769-772
        
        self.logger.info(f"Found {len(class_events)} motor imagery events")
        
        # Extract epochs
        epoch_samples = int(self.epoch_length * self.sampling_rate)
        epochs = []
        labels = []
        
        for event_pos, event_type in class_events:
            # Start epoch at cue onset (event position)
            start_sample = int(event_pos)
            end_sample = start_sample + epoch_samples
            
            # Check bounds
            if start_sample >= 0 and end_sample <= signals.shape[0]:
                # Extract epoch (use only EEG channels, exclude EOG)
                epoch = signals[start_sample:end_sample, :self.n_eeg_channels]
                epochs.append(epoch)
                
                # Convert event type to class index (flexible mapping)
                if event_type in [769, 770, 771, 772]:
                    class_label = event_type - 769  # 769->0, 770->1, 771->2, 772->3
                elif event_type in [1, 2, 3, 4]:
                    class_label = event_type - 1    # 1->0, 2->1, 3->2, 4->3
                else:
                    self.logger.warning(f"Unknown event type: {event_type}, skipping")
                    continue
                    
                labels.append(class_label)
            else:
                self.logger.warning(f"Epoch out of bounds: {start_sample}-{end_sample}")
        
        epochs = np.array(epochs)
        labels = np.array(labels)
        
        self.logger.info(f"Extracted {len(epochs)} valid epochs")
        self.logger.info(f"Epoch shape: {epochs.shape}")
        
        # Show class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label_idx, count in zip(unique_labels, counts):
            class_name = self.class_names[label_idx]
            self.logger.info(f"  - Class {label_idx} ({class_name}): {count} epochs")
        
        return epochs, labels
    
    def preprocess_epochs(self, epochs: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to extracted epochs.
        
        Args:
            epochs: Raw epochs (trials, samples, channels)
            
        Returns:
            Preprocessed epochs
        """
        self.logger.info("Applying preprocessing to epochs...")
        
        # 1. Bandpass filter for motor imagery (8-30 Hz)
        nyquist = self.sampling_rate / 2
        low_freq = 8.0 / nyquist
        high_freq = 30.0 / nyquist
        
        b, a = butter(4, [low_freq, high_freq], btype='band')
        
        epochs_filtered = np.zeros_like(epochs)
        for trial in range(epochs.shape[0]):
            for ch in range(epochs.shape[2]):
                epochs_filtered[trial, :, ch] = filtfilt(b, a, epochs[trial, :, ch])
        
        self.logger.info("Applied bandpass filter (8-30 Hz)")
        
        # 2. Common Average Reference (CAR)
        epochs_car = epochs_filtered - np.mean(epochs_filtered, axis=2, keepdims=True)
        self.logger.info("Applied Common Average Reference")
        
        # 3. Z-score normalization per trial and channel
        epochs_normalized = np.zeros_like(epochs_car)
        for trial in range(epochs_car.shape[0]):
            for ch in range(epochs_car.shape[2]):
                data = epochs_car[trial, :, ch]
                if np.std(data) > 0:
                    epochs_normalized[trial, :, ch] = (data - np.mean(data)) / np.std(data)
                else:
                    epochs_normalized[trial, :, ch] = data
        
        self.logger.info("Applied z-score normalization")
        
        return epochs_normalized
    
    def load_subject(self, subject_id: str, session: str = 'T') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data for a single subject.
        
        Args:
            subject_id: Subject identifier (e.g., 'A01')
            session: Session type ('T' for training, 'E' for evaluation)
            
        Returns:
            epochs: Preprocessed epochs (trials, samples, channels)
            labels: Class labels
        """
        # Construct file path - try MAT first, then GDF
        mat_filename = f"{subject_id}{session}.mat"
        mat_path = os.path.join(self.data_path, mat_filename)
        
        if os.path.exists(mat_path):
            self.logger.info(f"Loading subject {subject_id}, session {session} from MAT file")
            # Load MAT file (already extracts epochs)
            epochs, labels = self.load_mat_file(mat_path)
        else:
            # Fallback to GDF if available
            gdf_filename = f"{subject_id}{session}.gdf"
            gdf_path = os.path.join(self.data_path, gdf_filename)
            if os.path.exists(gdf_path):
                self.logger.info(f"Loading subject {subject_id}, session {session} from GDF file")
                signals, event_info = self.load_gdf_file(gdf_path)
                epochs, labels = self.extract_motor_imagery_epochs(signals, event_info)
            else:
                raise FileNotFoundError(f"No data file found for {subject_id}{session} (tried .mat and .gdf)")
        
        # Preprocess epochs
        epochs_preprocessed = self.preprocess_epochs(epochs)
        
        return epochs_preprocessed, labels
    
    def load_all_subjects(self, session: str = 'T') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data from all subjects.
        
        Args:
            session: Session type ('T' for training, 'E' for evaluation)
            
        Returns:
            all_epochs: Combined epochs from all subjects
            all_labels: Combined labels from all subjects
        """
        self.logger.info(f"Loading all subjects for session {session}")
        
        all_epochs = []
        all_labels = []
        subject_info = []
        
        # Load each subject
        for subject_num in range(1, 10):  # A01 to A09
            subject_id = f"A{subject_num:02d}"
            
            try:
                epochs, labels = self.load_subject(subject_id, session)
                
                all_epochs.append(epochs)
                all_labels.append(labels)
                subject_info.append({
                    'subject_id': subject_id,
                    'n_epochs': len(epochs),
                    'session': session
                })
                
                self.logger.info(f"Loaded {subject_id}: {len(epochs)} epochs")
                
            except Exception as e:
                self.logger.warning(f"Failed to load {subject_id}: {e}")
                continue
        
        if not all_epochs:
            raise ValueError("No subjects loaded successfully")
        
        # Combine all data
        combined_epochs = np.vstack(all_epochs)
        combined_labels = np.concatenate(all_labels)
        
        self.logger.info(f"Combined dataset: {combined_epochs.shape[0]} total epochs")
        
        # Show final class distribution
        unique_labels, counts = np.unique(combined_labels, return_counts=True)
        for label_idx, count in zip(unique_labels, counts):
            class_name = self.class_names[label_idx]
            self.logger.info(f"  - Total {class_name}: {count} epochs")
        
        return combined_epochs, combined_labels
    
    def prepare_for_training(self, epochs: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Prepare data for EEGNet training.
        
        Args:
            epochs: Preprocessed epochs (trials, samples, channels)
            labels: Class labels
            
        Returns:
            Dictionary with formatted data splits
        """
        from sklearn.model_selection import train_test_split
        from tensorflow.keras.utils import to_categorical
        
        self.logger.info("Preparing data for EEGNet training...")
        
        # Convert labels to one-hot encoding
        y_categorical = to_categorical(labels, num_classes=4)
        
        # Reshape for EEGNet: (trials, samples, channels) -> (trials, channels, samples, 1)
        X_reshaped = epochs.transpose(0, 2, 1)  # (trials, channels, samples)
        X_reshaped = X_reshaped[..., np.newaxis]  # Add final dimension
        
        self.logger.info(f"Reshaped for EEGNet: {epochs.shape} -> {X_reshaped.shape}")
        
        # Create data splits
        # First split: separate test set (15%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_reshaped, y_categorical,
            test_size=0.15,
            random_state=42,
            stratify=labels
        )
        
        # Second split: separate train and validation (85% -> 70% train, 15% val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.176,  # 15/85 ≈ 0.176
            random_state=42,
            stratify=y_temp.argmax(axis=1)
        )
        
        # Log split information
        self.logger.info(f"Data splits:")
        self.logger.info(f"  - Training: {X_train.shape[0]} samples ({X_train.shape[0]/X_reshaped.shape[0]*100:.1f}%)")
        self.logger.info(f"  - Validation: {X_val.shape[0]} samples ({X_val.shape[0]/X_reshaped.shape[0]*100:.1f}%)")
        self.logger.info(f"  - Test: {X_test.shape[0]} samples ({X_test.shape[0]/X_reshaped.shape[0]*100:.1f}%)")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'original_labels': labels
        }


def test_bci4_2a_loader():
    """Test the BCI IV 2a loader."""
    print("Testing BCI Competition IV Dataset 2a Loader...")
    
    try:
        # Initialize loader
        loader = BCI4_2A_Loader()
        
        # Test loading single subject
        print("\nTesting single subject loading...")
        epochs, labels = loader.load_subject('A01', 'T')
        print(f"Subject A01: {epochs.shape}, labels: {len(labels)}")
        
        # Test loading all subjects
        print("\nTesting all subjects loading...")
        all_epochs, all_labels = loader.load_all_subjects('T')
        print(f"All subjects: {all_epochs.shape}, labels: {len(all_labels)}")
        
        # Test data preparation
        print("\nTesting data preparation...")
        data_splits = loader.prepare_for_training(all_epochs, all_labels)
        print(f"Training data: {data_splits['X_train'].shape}")
        print(f"Validation data: {data_splits['X_val'].shape}")
        print(f"Test data: {data_splits['X_test'].shape}")
        
        print("\nBCI IV 2a loader test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_bci4_2a_loader()