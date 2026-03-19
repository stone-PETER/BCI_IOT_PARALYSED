"""
BCI Competition IV Dataset 2b Loader
Binary motor imagery classification: Left hand vs Right hand
3 EEG channels: C3, Cz, C4
"""

import numpy as np
import yaml
import logging
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

# Import augmentation pipeline
try:
    from eeg_augmentation import EEGAugmentationPipeline
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    logging.warning("EEG augmentation not available - install dependencies or check eeg_augmentation.py")


class BCI4_2B_Loader:
    """Data loader for BCI Competition IV Dataset 2b."""
    
    def __init__(self, config_path: str = "config_2b.yaml"):
        """Initialize the loader with configuration."""
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data_path = Path(self.config['paths']['data_path'])
        self.n_channels = self.config['model']['chans']
        self.n_classes = self.config['model']['nb_classes']
        self.n_samples = self.config['model']['samples']
        self.sampling_rate = 250  # Hz
        
        # Preprocessing config
        self.preprocess_config = self.config.get('preprocessing', {})
        
        # Augmentation config
        self.augmentation_config = self.config.get('augmentation', {})
        if self.augmentation_config.get('enabled', False) and AUGMENTATION_AVAILABLE:
            self.augmenter = EEGAugmentationPipeline(
                time_shift_range=self.augmentation_config.get('time_shift_range', 0.05),
                noise_level=self.augmentation_config.get('noise_level', 0.08),
                amplitude_range=tuple(self.augmentation_config.get('amplitude_range', [0.85, 1.15])),
                time_warp_range=tuple(self.augmentation_config.get('time_warp_range', [0.96, 1.04])),
                frequency_shift_range=self.augmentation_config.get('frequency_shift_range', 1.5),
                apply_probability=self.augmentation_config.get('apply_probability', 0.7)
            )
        else:
            self.augmenter = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_mat_file(self, mat_path: Path):
        """
        Load BCI IV 2b data from MAT file.
        
        BCI IV 2b format:
        - mat['data'] is (1, 3) array with 3 sessions/runs
        - Each session has:
          * X: Continuous EEG signal (samples, 6 channels)
          * trial: Trial start indices in X
          * y: Trial labels (1=left hand, 2=right hand)
          * fs: Sampling frequency (250 Hz)
        
        Returns:
            epochs: (n_trials, n_samples, 3)
            labels: (n_trials,) - 0 or 1
        """
        self.logger.info(f"Loading MAT file: {mat_path}")
        
        mat_data = loadmat(mat_path)
        
        if 'data' not in mat_data:
            raise ValueError(f"No 'data' field in {mat_path}")
        
        data = mat_data['data']  # Shape: (1, n_sessions)
        n_sessions = data.shape[1]
        
        self.logger.info(f"  Found {n_sessions} sessions")
        
        all_epochs = []
        all_labels = []
        
        # Process each session
        for session_idx in range(n_sessions):
            session_data = data[0, session_idx]
            
            # Extract data
            X_continuous = session_data['X'][0, 0]  # (n_samples, 6 channels)
            trial_indices = session_data['trial'][0, 0].flatten()  # Trial starts
            labels = session_data['y'][0, 0].flatten()  # 1 or 2
            
            # Extract only first 3 channels (EEG: C3, Cz, C4)
            X_eeg = X_continuous[:, :3]
            
            # Extract epochs
            for start_sample in trial_indices:
                start = int(start_sample)
                end = start + self.n_samples
                
                if end <= X_eeg.shape[0]:
                    epoch = X_eeg[start:end, :]
                    all_epochs.append(epoch)
            
            # Convert labels: 1 → 0 (left), 2 → 1 (right)
            labels_binary = labels - 1
            all_labels.append(labels_binary)
        
        # Concatenate
        all_epochs = np.array(all_epochs)
        all_labels = np.concatenate(all_labels, axis=0)
        
        self.logger.info(f"  Loaded {len(all_epochs)} total epochs")
        
        return all_epochs, all_labels
    
    def preprocess_epochs(self, epochs):
        """
        Apply preprocessing to epochs.
        
        Args:
            epochs: (n_trials, n_samples, n_channels)
        
        Returns:
            preprocessed epochs
        """
        # Check master preprocessing switch
        if not self.preprocess_config.get('enabled', True):
            self.logger.debug("Preprocessing DISABLED (enabled=false)")
            return epochs
        
        # Bandpass filter - support both flat and nested config structures
        # Flat: filter_low, filter_high | Nested: filter.enabled, filter.low, filter.high
        filter_low = self.preprocess_config.get('filter_low')
        filter_high = self.preprocess_config.get('filter_high')
        filter_nested = self.preprocess_config.get('filter', {})
        
        if filter_low is not None and filter_high is not None:
            # Flat config structure (config_2b.yaml)
            self.logger.debug(f"Applying bandpass filter: {filter_low}-{filter_high} Hz")
            epochs = self._bandpass_filter(epochs, filter_low, filter_high)
        elif filter_nested.get('enabled', False):
            # Nested config structure (legacy)
            low = filter_nested.get('low', 8)
            high = filter_nested.get('high', 30)
            self.logger.debug(f"Applying bandpass filter: {low}-{high} Hz")
            epochs = self._bandpass_filter(epochs, low, high)
        
        # Spatial filtering: Small Laplacian (preferred) or CAR
        # Small Laplacian must match npg_preprocessor.apply_small_laplacian()
        apply_laplacian = self.preprocess_config.get('apply_laplacian', False)
        apply_car = self.preprocess_config.get('apply_car', False)
        car_nested = self.preprocess_config.get('car', {})

        if apply_laplacian:
            self.logger.debug("Applying Small Laplacian spatial filter")
            epochs = self._apply_small_laplacian(epochs)
        elif apply_car or car_nested.get('enabled', False):
            self.logger.debug("Applying Common Average Reference")
            epochs = self._apply_car(epochs)

        # Z-score normalization - support both flat and nested config
        # Flat: normalize | Nested: zscore.enabled
        normalize_method = self.preprocess_config.get('normalize')
        zscore_nested = self.preprocess_config.get('zscore', {})
        
        if normalize_method == 'zscore' or zscore_nested.get('enabled', False):
            self.logger.debug("Applying z-score normalization")
            epochs = self._zscore_normalize(epochs)
        
        return epochs
    
    def _bandpass_filter(self, epochs, low, high):
        """Apply bandpass filter."""
        nyq = self.sampling_rate / 2
        b, a = butter(4, [low / nyq, high / nyq], btype='band')
        
        filtered = np.zeros_like(epochs)
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[2]):
                filtered[i, :, j] = filtfilt(b, a, epochs[i, :, j])
        
        return filtered
    
    def _apply_small_laplacian(self, epochs):
        """
        Apply Small Laplacian spatial filter for 3-channel (C3, Cz, C4) setup.

        Identical math to npg_preprocessor.apply_small_laplacian() so that
        training and real-time inference see the same spatially filtered data.

        C3_new = C3 - 0.5 * Cz
        Cz_new = Cz - 0.25 * (C3 + C4)
        C4_new = C4 - 0.5 * Cz

        Args:
            epochs: (n_trials, n_samples, 3) - channel order must be C3, Cz, C4
        Returns:
            Laplacian-filtered epochs with same shape
        """
        if epochs.shape[2] != 3:
            self.logger.warning(f"Small Laplacian expects 3 channels, got {epochs.shape[2]}. Skipping.")
            return epochs

        c3 = epochs[:, :, 0]  # (n_trials, n_samples)
        cz = epochs[:, :, 1]
        c4 = epochs[:, :, 2]

        laplacian = np.zeros_like(epochs)
        laplacian[:, :, 0] = c3 - 0.5 * cz          # C3 referenced to Cz
        laplacian[:, :, 1] = cz - 0.25 * (c3 + c4)  # Cz referenced to motor average
        laplacian[:, :, 2] = c4 - 0.5 * cz          # C4 referenced to Cz
        return laplacian

    def _apply_car(self, epochs):
        """Apply Common Average Reference."""
        mean_signal = epochs.mean(axis=2, keepdims=True)
        return epochs - mean_signal
    
    def _zscore_normalize(self, epochs):
        """
        Apply z-score normalization PER EPOCH PER CHANNEL.

        Normalises each trial independently so that the model sees the same
        kind of data at training time as it does during real-time inference
        (where each 4-second window is normalised on its own statistics).

        Global normalisation (across all trials) was previously used but it
        destroys the within-trial variance the model needs for classification.
        """
        normalized = np.zeros_like(epochs)
        for i in range(epochs.shape[0]):          # per trial
            for j in range(epochs.shape[2]):      # per channel
                mean = epochs[i, :, j].mean()
                std  = epochs[i, :, j].std()
                normalized[i, :, j] = (epochs[i, :, j] - mean) / (std + 1e-8)
        return normalized
    
    def apply_augmentation(self, epochs, labels, sampling_rate=250):
        """
        Apply data augmentation to epochs and labels.
        
        Args:
            epochs: EEG epochs (n_epochs, n_samples, n_channels)
            labels: Corresponding labels
            sampling_rate: Sampling rate in Hz
            
        Returns:
            augmented_epochs, augmented_labels
        """
        if not self.augmentation_config.get('enabled', False):
            self.logger.debug("Data augmentation DISABLED")
            return epochs, labels
            
        if self.augmenter is None:
            self.logger.warning("Augmentation enabled but no augmenter available")
            return epochs, labels
            
        augmentation_factor = self.augmentation_config.get('augmentation_factor', 1)
        
        if augmentation_factor <= 0:
            return epochs, labels
            
        self.logger.info(f"Applying data augmentation (factor={augmentation_factor})")
        
        augmented_epochs, augmented_labels = self.augmenter.augment_with_labels(
            epochs, labels, sampling_rate, augmentation_factor
        )
        
        return augmented_epochs, augmented_labels

    def load_subject(self, subject_id: str, sessions: list = None):
        """
        Load data for a subject.
        
        Args:
            subject_id: 'B01' to 'B09'
            sessions: ['T'] for training, ['E'] for evaluation
                     Default: ['T']
        
        Returns:
            epochs, labels
        """
        if sessions is None:
            sessions = ['T']
        
        all_epochs = []
        all_labels = []
        
        for session in sessions:
            filename = f"{subject_id}{session}.mat"
            mat_path = self.data_path / filename
            
            if not mat_path.exists():
                self.logger.warning(f"File not found: {mat_path}")
                continue
            
            epochs, labels = self.load_mat_file(mat_path)
            epochs = self.preprocess_epochs(epochs)
            
            all_epochs.append(epochs)
            all_labels.append(labels)
        
        if all_epochs:
            combined_epochs = np.concatenate(all_epochs, axis=0)
            combined_labels = np.concatenate(all_labels, axis=0)
            return combined_epochs, combined_labels
        else:
            raise ValueError(f"No data loaded for {subject_id}")
    
    def load_personal_calibration(self, npz_path: str):
        """
        Load personal calibration data from NPZ file.
        
        Supports two formats:
        1. Original NPG format: epochs_left, epochs_right, epochs_rest (from record_personal_calibration.py)
        2. Preprocessed format: X, y (from preprocess_calibration_for_training.py)
        
        Args:
            npz_path: Path to calibration NPZ file
        
        Returns:
            epochs: (n_trials, n_channels, n_samples) - combined LEFT + RIGHT trials
            labels: (n_trials,) - 0 for LEFT, 1 for RIGHT
            rest_epochs: (n_rest, n_channels, n_samples) - REST trials (for threshold calibration)
        """
        self.logger.info(f"Loading personal calibration data: {npz_path}")
        
        # Load NPZ file
        data = np.load(npz_path)
        keys = list(data.keys())
        
        # Detect format
        if 'X' in keys and 'y' in keys:
            # Preprocessed format (from preprocess_calibration_for_training.py)
            self.logger.info("  Format: Preprocessed (X, y)")
            epochs = data['X']
            labels = data['y']
            rest_epochs = np.array([])  # No REST trials in preprocessed format
            
            self.logger.info(f"  Total training: {len(epochs)} epochs")
            self.logger.info(f"  LEFT trials (0):  {np.sum(labels == 0)}")
            self.logger.info(f"  RIGHT trials (1): {np.sum(labels == 1)}")
            self.logger.info(f"  REST trials: {len(rest_epochs)} (not available in preprocessed data)")
        
        elif 'epochs_left' in keys and 'epochs_right' in keys:
            # Original NPG format (from record_personal_calibration.py)
            self.logger.info("  Format: Original NPG (epochs_left, epochs_right, epochs_rest)")
            
            epochs_left = data['epochs_left']
            epochs_right = data['epochs_right']
            epochs_rest = data['epochs_rest']
            
            # Create labels: 0 for LEFT, 1 for RIGHT
            labels_left = np.zeros(len(epochs_left), dtype=np.int32)
            labels_right = np.ones(len(epochs_right), dtype=np.int32)
            
            # Combine LEFT and RIGHT for training
            epochs = np.concatenate([epochs_left, epochs_right], axis=0)
            labels = np.concatenate([labels_left, labels_right], axis=0)
            rest_epochs = epochs_rest
            
            self.logger.info(f"  LEFT trials:  {len(epochs_left)}")
            self.logger.info(f"  RIGHT trials: {len(epochs_right)}")
            self.logger.info(f"  REST trials:  {len(epochs_rest)}")
            self.logger.info(f"  Total training: {len(epochs)} epochs")
            
            if 'sampling_rate' in data and 'channel_names' in data:
                self.logger.info(f"  Sampling rate: {data['sampling_rate']} Hz")
                self.logger.info(f"  Channels: {data['channel_names']}")
        
        else:
            raise ValueError(
                f"Unknown NPZ format. Expected either:\n"
                f"  - Preprocessed: X, y (from preprocess_calibration_for_training.py)\n"
                f"  - Original: epochs_left, epochs_right, epochs_rest (from record_personal_calibration.py)\n"
                f"Found keys: {keys}"
            )
        
        return epochs, labels, rest_epochs
    
    def load_mixed_data(self, 
                       personal_npz: str,
                       personal_weight: float = 0.7,
                       benchmark_sessions: list = ['T']):
        """
        Load and mix personal calibration data with benchmark datasets.
        
        Useful for transfer learning with limited personal data.
        
        Args:
            personal_npz: Path to personal calibration NPZ
            personal_weight: Weight for personal data (0-1)
            benchmark_sessions: Which benchmark sessions to include
        
        Returns:
            epochs: Mixed epochs from personal + benchmark
            labels: Corresponding labels
        """
        self.logger.info("Loading mixed personal + benchmark data")
        
        # Load personal data
        personal_epochs, personal_labels, _ = self.load_personal_calibration(personal_npz)
        
        # Load benchmark data
        benchmark_epochs, benchmark_labels = self.load_all_subjects(sessions=benchmark_sessions)
        
        # Calculate sampling ratios
        n_personal = len(personal_epochs)
        n_benchmark_target = int(n_personal * (1 - personal_weight) / personal_weight)
        
        # Sample from benchmark data
        if n_benchmark_target < len(benchmark_epochs):
            indices = np.random.choice(
                len(benchmark_epochs),
                size=n_benchmark_target,
                replace=False
            )
            benchmark_epochs = benchmark_epochs[indices]
            benchmark_labels = benchmark_labels[indices]
        
        # Combine
        epochs = np.concatenate([personal_epochs, benchmark_epochs], axis=0)
        labels = np.concatenate([personal_labels, benchmark_labels], axis=0)
        
        self.logger.info(f"  Personal data: {n_personal} epochs ({personal_weight*100:.0f}%)")
        self.logger.info(f"  Benchmark data: {len(benchmark_epochs)} epochs ({(1-personal_weight)*100:.0f}%)")
        self.logger.info(f"  Total: {len(epochs)} epochs")
        
        return epochs, labels
    
    def load_all_subjects(self, sessions: list = None):
        """
        Load all subjects.
        
        Args:
            sessions: ['T'] or ['E'] or ['T', 'E']
        
        Returns:
            all_epochs, all_labels
        """
        subjects = [f'B{i:02d}' for i in range(1, 10)]
        
        self.logger.info(f"Loading all subjects for sessions {sessions}")
        
        all_epochs_list = []
        all_labels_list = []
        
        for subject in subjects:
            try:
                epochs, labels = self.load_subject(subject, sessions)
                all_epochs_list.append(epochs)
                all_labels_list.append(labels)
                self.logger.info(f"  {subject}: {len(epochs)} epochs")
            except Exception as e:
                self.logger.error(f"Error loading {subject}: {e}")
                continue
        
        all_epochs = np.concatenate(all_epochs_list, axis=0)
        all_labels = np.concatenate(all_labels_list, axis=0)

        self.logger.info(f"Total: {len(all_epochs)} epochs from {len(all_epochs_list)} subjects")
        # NOTE: Augmentation intentionally NOT applied here.
        # It must be applied ONLY to the training split after train/test separation
        # to prevent augmented copies from leaking into the validation/test sets.
        return all_epochs, all_labels


if __name__ == "__main__":
    # Test the loader
    loader = BCI4_2B_Loader()
    
    # Test single subject
    print("\nTesting single subject load:")
    epochs, labels = loader.load_subject('B01', sessions=['T'])
    print(f"Loaded B01: epochs shape={epochs.shape}, labels shape={labels.shape}")
    print(f"Unique labels: {np.unique(labels, return_counts=True)}")
    
    # Test all subjects
    print("\nTesting all subjects load:")
    all_epochs, all_labels = loader.load_all_subjects(sessions=['T'])
    print(f"Total epochs: {all_epochs.shape}")
    print(f"Class distribution: {np.unique(all_labels, return_counts=True)}")
