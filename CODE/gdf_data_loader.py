"""
GDF Data Loader for BCI Competition III Dataset 3a
Handles loading and preprocessing of motor imagery EEG data in GDF format

This module specifically handles the BCI Competition III Dataset 3a format:
- k3b_s.txt: Raw EEG data (60 channels, 250 Hz)
- k3b_HDR_TRIG.txt: Event triggers (motor imagery cue onsets)
- k3b_HDR_Classlabel.txt: Class labels (1=left hand, 2=right hand, 3=foot, 4=tongue)
- k3b_HDR_ArtifactSelection.txt: Artifact markers (0=clean, 1=artifact)
"""

import numpy as np
import yaml
import logging
import os
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Import from local preprocessing module in CODE directory
import sys
from pathlib import Path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
from preprocessing import PreProcessor


class GDFDataLoader:
    """
    Data loader for BCI Competition III Dataset 3a (GDF format).
    
    Handles loading raw EEG data, event triggers, class labels, and artifact markers
    from the competition format files.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize GDF data loader with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize preprocessor
        self.preprocessor = PreProcessor(config_path)
        
        # BCI Competition III specific parameters
        self.sampling_rate = self.config['eeg']['sampling_rate']  # 250 Hz
        self.channels = self.config['eeg']['channels']  # 60 channels
        self.motor_imagery_channels = self.config['eeg']['motor_imagery_channels']
        self.epoch_length = self.config['eeg']['epoch_length']  # 4.0 seconds
        self.baseline_length = self.config['eeg']['baseline_length']  # 0.5 seconds
        
        # Motor imagery parameters
        self.mi_config = self.config['eeg']['motor_imagery']
        self.class_mapping = self.mi_config['class_mapping']  # {1:0, 2:1, 3:2, 4:3}
        
        # File paths
        self.data_path = self.config['paths']['bci_competition_path']
        self.eeg_file = self.config['paths']['eeg_data_file']
        self.triggers_file = self.config['paths']['triggers_file']
        self.labels_file = self.config['paths']['labels_file']
        self.artifacts_file = self.config['paths']['artifacts_file']
        
        # Data storage
        self.raw_eeg_data = None
        self.triggers = None
        self.labels = None
        self.artifacts = None
        self.processed_epochs = None
        self.processed_labels = None
    
    def load_raw_eeg_data(self) -> np.ndarray:
        """
        Load raw EEG data from k3b_s.txt file.
        
        Returns:
            Raw EEG data of shape (samples, channels)
        """
        eeg_path = os.path.join(self.data_path, self.eeg_file)
        
        if not os.path.exists(eeg_path):
            raise FileNotFoundError(f"EEG data file not found: {eeg_path}")
        
        self.logger.info(f"Loading raw EEG data from: {eeg_path}")
        self.logger.info("This may take a moment due to large file size...")
        
        # Load EEG data (space-separated values, each row is a sample)
        try:
            raw_data = np.loadtxt(eeg_path)
            self.logger.info(f"Loaded EEG data shape: {raw_data.shape}")
            
            # Validate expected shape
            if raw_data.shape[1] != self.channels:
                self.logger.warning(f"Expected {self.channels} channels, got {raw_data.shape[1]}")
            
            self.raw_eeg_data = raw_data
            return raw_data
            
        except Exception as e:
            self.logger.error(f"Error loading EEG data: {e}")
            raise
    
    def load_triggers(self) -> np.ndarray:
        """
        Load event triggers from k3b_HDR_TRIG.txt file.
        
        Returns:
            Trigger positions (sample indices) as 1D array
        """
        triggers_path = os.path.join(self.data_path, self.triggers_file)
        
        if not os.path.exists(triggers_path):
            raise FileNotFoundError(f"Triggers file not found: {triggers_path}")
        
        self.logger.info(f"Loading triggers from: {triggers_path}")
        
        # Load trigger positions
        triggers = np.loadtxt(triggers_path, dtype=int).flatten()
        self.logger.info(f"Loaded {len(triggers)} triggers")
        
        self.triggers = triggers
        return triggers
    
    def load_labels(self) -> np.ndarray:
        """
        Load class labels from k3b_HDR_Classlabel.txt file.
        
        Returns:
            Class labels (1=left hand, 2=right hand, 3=foot, 4=tongue, NaN=test)
        """
        labels_path = os.path.join(self.data_path, self.labels_file)
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        self.logger.info(f"Loading labels from: {labels_path}")
        
        # Load labels (may contain NaN for test trials)
        labels_raw = []
        with open(labels_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line != 'NaN':
                    try:
                        label = int(float(line))
                        labels_raw.append(label)
                    except ValueError:
                        labels_raw.append(np.nan)
                else:
                    labels_raw.append(np.nan)
        
        labels = np.array(labels_raw)
        
        # Count valid vs test trials
        valid_labels = labels[~np.isnan(labels)]
        test_trials = np.sum(np.isnan(labels))
        
        self.logger.info(f"Loaded {len(labels)} total trials:")
        self.logger.info(f"  - Training trials: {len(valid_labels)}")
        self.logger.info(f"  - Test trials (NaN): {test_trials}")
        
        # Show class distribution for training trials
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_name = self.mi_config['class_names'][int(label-1)]
            self.logger.info(f"  - Class {int(label)} ({class_name}): {count} trials")
        
        self.labels = labels
        return labels
    
    def load_artifacts(self) -> np.ndarray:
        """
        Load artifact markers from k3b_HDR_ArtifactSelection.txt file.
        
        Returns:
            Artifact markers (0=clean trial, 1=artifact present)
        """
        artifacts_path = os.path.join(self.data_path, self.artifacts_file)
        
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")
        
        self.logger.info(f"Loading artifact markers from: {artifacts_path}")
        
        # Load artifact markers
        artifacts = np.loadtxt(artifacts_path, dtype=int).flatten()
        
        clean_trials = np.sum(artifacts == 0)
        artifact_trials = np.sum(artifacts == 1)
        
        self.logger.info(f"Loaded {len(artifacts)} artifact markers:")
        self.logger.info(f"  - Clean trials: {clean_trials}")
        self.logger.info(f"  - Artifact trials: {artifact_trials}")
        
        self.artifacts = artifacts
        return artifacts
    
    def load_all_metadata(self) -> Dict:
        """
        Load all metadata files (triggers, labels, artifacts).
        
        Returns:
            Dictionary containing all loaded metadata
        """
        self.logger.info("Loading all BCI Competition III Dataset 3a metadata...")
        
        triggers = self.load_triggers()
        labels = self.load_labels()
        artifacts = self.load_artifacts()
        
        # Validate that all arrays have the same length
        if not (len(triggers) == len(labels) == len(artifacts)):
            raise ValueError(
                f"Metadata length mismatch: "
                f"triggers={len(triggers)}, labels={len(labels)}, artifacts={len(artifacts)}"
            )
        
        return {
            'triggers': triggers,
            'labels': labels,
            'artifacts': artifacts
        }
    
    def extract_motor_imagery_epochs(
        self, 
        eeg_data: np.ndarray, 
        triggers: np.ndarray, 
        labels: np.ndarray,
        artifacts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract motor imagery epochs from continuous EEG data.
        
        Args:
            eeg_data: Raw EEG data of shape (samples, channels)
            triggers: Trigger positions (sample indices)
            labels: Class labels for each trigger
            artifacts: Artifact markers for each trigger
            
        Returns:
            epochs: Extracted epochs of shape (trials, samples, channels)
            epoch_labels: Corresponding class labels for valid epochs
        """
        self.logger.info("Extracting motor imagery epochs...")
        
        # Calculate epoch parameters
        epoch_samples = int(self.epoch_length * self.sampling_rate)  # 4s * 250Hz = 1000 samples
        baseline_samples = int(self.baseline_length * self.sampling_rate)  # 0.5s * 250Hz = 125 samples
        
        epochs = []
        epoch_labels = []
        
        # Process each trigger
        for i, (trigger_pos, label, artifact) in enumerate(zip(triggers, labels, artifacts)):
            
            # Skip test trials (NaN labels) and artifact trials
            if np.isnan(label) or artifact == 1:
                continue
            
            # Skip if label not in valid motor imagery classes (1,2,3,4)
            if label not in [1, 2, 3, 4]:
                continue
            
            # Calculate epoch boundaries
            start_sample = int(trigger_pos) - baseline_samples
            end_sample = start_sample + epoch_samples
            
            # Check if epoch is within data bounds
            if start_sample >= 0 and end_sample <= eeg_data.shape[0]:
                
                # Extract epoch
                epoch = eeg_data[start_sample:end_sample, :]
                epochs.append(epoch)
                
                # Convert label to model index (1,2,3,4 -> 0,1,2,3)
                model_label = self.class_mapping[int(label)]
                epoch_labels.append(model_label)
                
            else:
                self.logger.warning(f"Trial {i}: Epoch out of bounds (trigger at {trigger_pos})")
        
        epochs = np.array(epochs)
        epoch_labels = np.array(epoch_labels)
        
        self.logger.info(f"Extracted {len(epochs)} valid motor imagery epochs")
        self.logger.info(f"Epoch shape: {epochs.shape}")
        
        # Show final class distribution
        unique_labels, counts = np.unique(epoch_labels, return_counts=True)
        for model_idx, count in zip(unique_labels, counts):
            class_name = self.mi_config['class_names'][model_idx]
            self.logger.info(f"  - Class {model_idx} ({class_name}): {count} epochs")
        
        return epochs, epoch_labels
    
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline to load and preprocess BCI Competition III Dataset 3a.
        
        Returns:
            X: Preprocessed epochs of shape (trials, samples, channels)
            y: Corresponding class labels (0,1,2,3)
        """
        self.logger.info("Starting complete BCI Competition III data loading and preprocessing...")
        
        # Load raw EEG data
        eeg_data = self.load_raw_eeg_data()
        
        # Load metadata
        metadata = self.load_all_metadata()
        
        # Extract motor imagery epochs
        epochs, labels = self.extract_motor_imagery_epochs(
            eeg_data,
            metadata['triggers'],
            metadata['labels'], 
            metadata['artifacts']
        )
        
        if len(epochs) == 0:
            raise ValueError("No valid motor imagery epochs extracted")
        
        # Apply preprocessing pipeline
        self.logger.info("Applying preprocessing pipeline...")
        
        # Select motor imagery channels
        if self.motor_imagery_channels:
            epochs_selected = epochs[:, :, self.motor_imagery_channels]
            self.logger.info(f"Selected {len(self.motor_imagery_channels)} motor imagery channels")
        else:
            epochs_selected = epochs
            self.logger.info("Using all channels")
        
        # Apply bandpass filter
        epochs_filtered = self.preprocessor.apply_bandpass_filter(
            epochs_selected,
            self.preprocessor.preprocessing_config['bandpass']['low_freq'],
            self.preprocessor.preprocessing_config['bandpass']['high_freq'],
            self.sampling_rate,
            self.preprocessor.preprocessing_config['bandpass']['filter_order']
        )
        self.logger.info("Applied bandpass filter")
        
        # Apply CAR reference
        if self.preprocessor.preprocessing_config['reference'] == "CAR":
            epochs_referenced = self.preprocessor.apply_car_reference(epochs_filtered)
            self.logger.info("Applied CAR reference")
        else:
            epochs_referenced = epochs_filtered
        
        # Normalize epochs
        epochs_normalized = self.preprocessor.normalize_epochs(
            epochs_referenced,
            self.preprocessor.preprocessing_config['normalization']
        )
        self.logger.info("Applied normalization")
        
        # Store processed data
        self.processed_epochs = epochs_normalized
        self.processed_labels = labels
        
        self.logger.info("BCI Competition III data loading and preprocessing completed successfully!")
        self.logger.info(f"Final data shape: {epochs_normalized.shape}")
        self.logger.info(f"Final labels shape: {labels.shape}")
        
        return epochs_normalized, labels
    
    def prepare_for_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for training with proper splits for 4-class motor imagery.
        
        Args:
            X: Input epochs of shape (trials, samples, channels)
            y: Labels (0,1,2,3 for left hand, right hand, foot, tongue)
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train/validation/test splits
        """
        self.logger.info("Preparing data for 4-class motor imagery training...")
        
        # Convert labels to categorical (one-hot encoding)
        y_categorical = to_categorical(y, num_classes=4)
        self.logger.info(f"Converted to 4-class categorical: {y_categorical.shape}")
        
        # Reshape for EEGNet: (trials, samples, channels) -> (trials, channels, samples, 1)
        X_transposed = X.transpose(0, 2, 1)  # (trials, samples, channels) -> (trials, channels, samples)
        X_reshaped = X_transposed[..., np.newaxis]  # Add final dimension for Conv2D
        
        self.logger.info(f"Reshaped for EEGNet: {X.shape} -> {X_reshaped.shape}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_reshaped, y_categorical,
            test_size=test_split,
            random_state=random_state,
            stratify=y  # Use original labels for stratification
        )
        
        # Second split: separate train and validation
        validation_size = validation_split / (1 - test_split)  # Adjust for already removed test set
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_temp.argmax(axis=1)  # Use categorical labels for stratification
        )
        
        # Log split information
        self.logger.info(f"Data splits for 4-class motor imagery:")
        self.logger.info(f"  - Training: {X_train.shape[0]} samples ({X_train.shape[0]/X_reshaped.shape[0]*100:.1f}%)")
        self.logger.info(f"  - Validation: {X_val.shape[0]} samples ({X_val.shape[0]/X_reshaped.shape[0]*100:.1f}%)")
        self.logger.info(f"  - Test: {X_test.shape[0]} samples ({X_test.shape[0]/X_reshaped.shape[0]*100:.1f}%)")
        
        # Show class distribution in each split
        for split_name, y_split in [("Training", y_train), ("Validation", y_val), ("Test", y_test)]:
            labels_split = y_split.argmax(axis=1)
            unique_labels, counts = np.unique(labels_split, return_counts=True)
            distribution = {self.mi_config['class_names'][i]: 0 for i in range(4)}
            for label, count in zip(unique_labels, counts):
                distribution[self.mi_config['class_names'][label]] = count
            self.logger.info(f"  - {split_name} distribution: {distribution}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'original_labels': y
        }
    
    def get_dataset_info(self) -> Dict:
        """
        Get comprehensive information about the loaded BCI Competition III dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        info = {
            "dataset": "BCI Competition III Dataset 3a",
            "task": "4-class motor imagery",
            "classes": self.mi_config['class_names'],
            "sampling_rate": self.sampling_rate,
            "total_channels": self.channels,
            "motor_imagery_channels": len(self.motor_imagery_channels),
            "epoch_length": self.epoch_length,
            "baseline_length": self.baseline_length
        }
        
        if self.processed_epochs is not None and self.processed_labels is not None:
            unique_labels, counts = np.unique(self.processed_labels, return_counts=True)
            class_distribution = {}
            for label, count in zip(unique_labels, counts):
                class_name = self.mi_config['class_names'][label]
                class_distribution[class_name] = int(count)
            
            info.update({
                "total_epochs": len(self.processed_epochs),
                "epochs_shape": self.processed_epochs.shape,
                "class_distribution": class_distribution,
                "data_loaded": True
            })
        else:
            info["data_loaded"] = False
        
        return info


def test_gdf_data_loader():
    """Test function for GDF data loader."""
    print("Testing GDF BCI Competition III data loader...")
    
    try:
        # Initialize data loader
        data_loader = GDFDataLoader()
        
        # Load and preprocess data
        X, y = data_loader.load_and_preprocess_data()
        
        print(f"Loaded data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Unique labels: {np.unique(y)}")
        
        # Prepare for training
        data_splits = data_loader.prepare_for_training(X, y)
        
        print(f"Training data shape: {data_splits['X_train'].shape}")
        print(f"Validation data shape: {data_splits['X_val'].shape}")
        print(f"Test data shape: {data_splits['X_test'].shape}")
        
        # Get dataset info
        info = data_loader.get_dataset_info()
        print(f"Dataset info: {info}")
        
        print("GDF data loader test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure the BCI Competition III Dataset 3a files exist in the correct location")


if __name__ == "__main__":
    test_gdf_data_loader()