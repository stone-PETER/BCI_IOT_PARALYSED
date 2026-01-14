"""
BCI Competition IV Dataset 2b Loader
Binary motor imagery classification (Left hand vs Right hand)
3 EEG channels: C3, Cz, C4 at 250 Hz
"""

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from pathlib import Path
import logging
import yaml


class BCI4_2B_Loader:
    """
    Load and preprocess BCI Competition IV Dataset 2b.
    
    Dataset characteristics:
    - 9 subjects (B01-B09)
    - 2 classes: Left hand (769), Right hand (770)
    - 3 EEG channels: C3, Cz, C4
    - 3 EOG channels (not used for classification)
    - 250 Hz sampling rate
    - 5 sessions per subject: 01T, 02T, 03T (training), 04E, 05E (evaluation)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize loader with configuration."""
        # Get script directory for relative path resolution
        script_dir = Path(__file__).parent.resolve()
        
        # Load configuration
        if isinstance(config_path, str):
            config_path = script_dir / config_path
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Dataset paths
        self.data_path = script_dir / "BCI" / "bci4_2b"
        
        # Dataset parameters
        self.n_channels = 3  # Only EEG channels (C3, Cz, C4)
        self.n_classes = 2   # Left hand, Right hand
        self.sample_rate = 250  # Hz
        self.channel_names = ['C3', 'Cz', 'C4']
        
        # Event codes from BCI IV 2b
        self.event_codes = {
            'trial_start': 768,    # 0x0300
            'left_hand': 769,      # 0x0301 - Class 1
            'right_hand': 770,     # 0x0302 - Class 2
            'feedback': 781,       # 0x030D
            'rejected': 1023,      # 0x03FF - Artifact
            'new_run': 32766       # 0x7FFE
        }
        
        # Class mapping
        self.class_names = {
            769: 'left_hand',    # Class 1
            770: 'right_hand'    # Class 2
        }
        
        # Preprocessing parameters
        self.filter_low = self.config['preprocessing'].get('filter_low', 8)
        self.filter_high = self.config['preprocessing'].get('filter_high', 30)
        self.apply_car = self.config['preprocessing'].get('apply_car', True)
        self.normalize = self.config['preprocessing'].get('normalize', 'zscore')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def load_mat_file(self, mat_path: Path):
        """
        Load BCI IV 2b MAT file and extract epochs.
        
        Args:
            mat_path: Path to MAT file
            
        Returns:
            epochs: Array of shape (n_trials, n_samples, n_channels)
            labels: Array of shape (n_trials,) with values 0 (left) or 1 (right)
        """
        self.logger.info(f"Loading MAT file: {mat_path}")
        
        # Load MAT file
        mat_data = loadmat(str(mat_path))
        
        # Extract signal data and header
        # BCI IV 2b structure: mat_data contains 's' (signal) and 'h' (header)
        if 's' in mat_data:
            signal = mat_data['s']  # Shape: (n_samples, 6) - 3 EEG + 3 EOG
        else:
            raise ValueError(f"No signal data found in {mat_path}")
        
        # Extract only EEG channels (first 3 columns)
        eeg_signal = signal[:, :self.n_channels]  # Shape: (n_samples, 3)
        
        # Extract events from header
        if 'h' in mat_data:
            header = mat_data['h']
            events = self._extract_events(header)
        else:
            raise ValueError(f"No header found in {mat_path}")
        
        # Extract epochs based on events
        epochs, labels = self._extract_epochs_from_events(eeg_signal, events)
        
        self.logger.info(f"Loaded {len(epochs)} epochs from MAT file")
        self.logger.info(f"Epochs shape: {epochs.shape}")
        
        # Log class distribution
        unique, counts = np.unique(labels, return_counts=True)
        for cls, count in zip(unique, counts):
            cls_name = 'left_hand' if cls == 0 else 'right_hand'
            self.logger.info(f"  - Class {cls} ({cls_name}): {count} epochs")
        
        return epochs, labels
    
    def _extract_events(self, header):
        """Extract event information from MAT file header."""
        events = []
        
        # Navigate the nested MATLAB structure
        # header is typically a structured array
        if hasattr(header, 'dtype') and header.dtype.names:
            if 'EVENT' in header.dtype.names:
                event_data = header['EVENT'][0, 0]
                
                # Extract event types, positions, and durations
                if hasattr(event_data, 'dtype') and event_data.dtype.names:
                    event_types = event_data['TYP'][0, 0].flatten() if 'TYP' in event_data.dtype.names else []
                    event_pos = event_data['POS'][0, 0].flatten() if 'POS' in event_data.dtype.names else []
                    
                    for i in range(len(event_types)):
                        events.append({
                            'type': int(event_types[i]),
                            'position': int(event_pos[i])
                        })
        
        return events
    
    def _extract_epochs_from_events(self, signal, events):
        """
        Extract epochs based on cue events.
        
        Timing for BCI IV 2b:
        - Cue onset at t=3s
        - Motor imagery from t=3s to t=7s (4 seconds)
        - We extract 4 seconds (1000 samples at 250 Hz)
        """
        epochs_list = []
        labels_list = []
        
        epoch_length = 1000  # 4 seconds at 250 Hz
        
        for event in events:
            event_type = event['type']
            event_pos = event['position']
            
            # Check if this is a cue event (left or right)
            if event_type in [self.event_codes['left_hand'], self.event_codes['right_hand']]:
                # Extract epoch starting from cue onset
                start_sample = event_pos
                end_sample = start_sample + epoch_length
                
                # Check if we have enough samples
                if end_sample <= len(signal):
                    epoch = signal[start_sample:end_sample, :]  # Shape: (1000, 3)
                    
                    # Convert label to 0/1
                    label = 0 if event_type == self.event_codes['left_hand'] else 1
                    
                    epochs_list.append(epoch)
                    labels_list.append(label)
        
        # Convert to numpy arrays
        epochs = np.array(epochs_list)  # Shape: (n_trials, 1000, 3)
        labels = np.array(labels_list)  # Shape: (n_trials,)
        
        return epochs, labels
    
    def apply_bandpass_filter(self, data):
        """Apply bandpass filter to data."""
        nyquist = self.sample_rate / 2
        low = self.filter_low / nyquist
        high = self.filter_high / nyquist
        
        b, a = butter(5, [low, high], btype='band')
        
        # Filter each channel
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[2]):
            filtered_data[:, :, ch] = filtfilt(b, a, data[:, :, ch], axis=1)
        
        return filtered_data
    
    def apply_common_average_reference(self, data):
        """Apply Common Average Reference (CAR) to data."""
        # Compute average across channels
        avg_signal = np.mean(data, axis=2, keepdims=True)
        # Subtract average from each channel
        car_data = data - avg_signal
        return car_data
    
    def normalize_data(self, data):
        """Normalize data using z-score normalization."""
        if self.normalize == 'zscore':
            # Z-score normalization per channel
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            normalized_data = (data - mean) / (std + 1e-8)
        else:
            normalized_data = data
        
        return normalized_data
    
    def preprocess_epochs(self, epochs):
        """Apply preprocessing pipeline to epochs."""
        self.logger.info("Applying preprocessing to epochs...")
        
        # 1. Bandpass filter
        epochs = self.apply_bandpass_filter(epochs)
        self.logger.info(f"Applied bandpass filter ({self.filter_low}-{self.filter_high} Hz)")
        
        # 2. Common Average Reference
        if self.apply_car:
            epochs = self.apply_common_average_reference(epochs)
            self.logger.info("Applied Common Average Reference")
        
        # 3. Normalization
        epochs = self.normalize_data(epochs)
        self.logger.info(f"Applied {self.normalize} normalization")
        
        return epochs
    
    def load_subject(self, subject_id: str, sessions: list = None):
        """
        Load data for a specific subject.
        
        Args:
            subject_id: Subject ID (e.g., 'B01')
            sessions: List of session IDs (e.g., ['T', 'E'])
                     T = Training, E = Evaluation
                     Default: ['T'] (training session only)
        
        Returns:
            epochs: Array of shape (n_trials, n_samples, n_channels)
            labels: Array of shape (n_trials,)
        """
        if sessions is None:
            # Use training session by default
            sessions = ['T']
        
        all_epochs = []
        all_labels = []
        
        for session in sessions:
            # Construct filename: B01T.mat or B01E.mat
            filename = f"{subject_id}{session}.mat"
            mat_path = self.data_path / filename
            
            if not mat_path.exists():
                self.logger.warning(f"File not found: {mat_path}")
                continue
            
            self.logger.info(f"Loading subject {subject_id}, session {session} from MAT file")
            
            # Load data
            epochs, labels = self.load_mat_file(mat_path)
            
            # Preprocess
            epochs = self.preprocess_epochs(epochs)
            
            all_epochs.append(epochs)
            all_labels.append(labels)
            
            self.logger.info(f"Loaded {subject_id}{session}: {len(epochs)} epochs")
        
        # Concatenate all sessions
        if all_epochs:
            combined_epochs = np.concatenate(all_epochs, axis=0)
            combined_labels = np.concatenate(all_labels, axis=0)
            
            self.logger.info(f"Combined {subject_id}: {len(combined_epochs)} total epochs")
            
            return combined_epochs, combined_labels
        else:
            raise ValueError(f"No data loaded for subject {subject_id}")
    
    def load_all_subjects(self, sessions: list = None):
        """
        Load data for all subjects.
        
        Args:
            sessions: List of session IDs to load
        
        Returns:
            all_epochs: Combined epochs from all subjects
            all_labels: Combined labels from all subjects
        """
        subjects = [f'B{i:02d}' for i in range(1, 10)]  # B01 to B09
        
        self.logger.info(f"Loading all subjects for sessions {sessions}")
        
        all_epochs_list = []
        all_labels_list = []
        
        for subject in subjects:
            try:
                epochs, labels = self.load_subject(subject, sessions)
                all_epochs_list.append(epochs)
                all_labels_list.append(labels)
            except Exception as e:
                self.logger.error(f"Error loading {subject}: {e}")
                continue
        
        # Combine all subjects
        all_epochs = np.concatenate(all_epochs_list, axis=0)
        all_labels = np.concatenate(all_labels_list, axis=0)
        
        self.logger.info(f"Combined dataset: {len(all_epochs)} total epochs")
        
        # Log class distribution
        unique, counts = np.unique(all_labels, return_counts=True)
        for cls, count in zip(unique, counts):
            cls_name = 'left_hand' if cls == 0 else 'right_hand'
            self.logger.info(f"  - Total {cls_name}: {count} epochs")
        
        return all_epochs, all_labels


if __name__ == "__main__":
    # Test the loader
    import logging
    logging.basicConfig(level=logging.INFO)
    
    loader = BCI4_2B_Loader()
    
    # Test loading one subject
    print("\n" + "="*70)
    print("Testing BCI IV 2b Loader")
    print("="*70)
    
    epochs, labels = loader.load_subject('B01', sessions=['01T'])
    print(f"\nLoaded B01 session 01T:")
    print(f"  Epochs shape: {epochs.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {np.unique(labels)}")
    
    # Test loading all subjects
    all_epochs, all_labels = loader.load_all_subjects(sessions=['01T', '02T', '03T'])
    print(f"\nLoaded all subjects (3 training sessions):")
    print(f"  Total epochs: {len(all_epochs)}")
    print(f"  Epochs shape: {all_epochs.shape}")
    print(f"  Class distribution: {np.bincount(all_labels)}")
