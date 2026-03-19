#!/usr/bin/env python3
"""
Personal Model Fine-Tuning Script

Fine-tunes pretrained EEGNet model on personal calibration data using transfer learning.
Implements three strategies with layer freezing:
- head-only: Freeze all except final Dense layer (30+ trials needed)
- last-block: Freeze Block 1, train Block 2 + Dense head (100+ trials, recommended)
- full: Train entire model (500+ trials needed)

Also calibrates neutral threshold using REST trials for 3-state classification.

PERSONALIZATION MODE (--personalize):
  Aggressive personalization optimized for 85%+ accuracy:
  - Minimal validation split: 5% (vs 20% default)
  - Extended training: 500 epochs (vs 100 default)
  - Patient early stopping: 100 patience (vs 25 default)
  - Forces model to learn your unique signal patterns

Usage:
    # Standard fine-tuning (balanced approach)
    python fine_tune_personal_model.py --calibration-file=calibration_data/alan_*.npz --strategy=full
    
    # AGGRESSIVE PERSONALIZATION (85%+ target)
    python fine_tune_personal_model.py --calibration-file=calibration_data/alan_*.npz --strategy=full --personalize
    
    # Head-only fine-tuning (fastest, minimal data)
    python fine_tune_personal_model.py --calibration-file=calibration_data/alan_*.npz --strategy=head-only
    
    # Last-block fine-tuning (recommended, balanced)
    python fine_tune_personal_model.py --calibration-file=calibration_data/alan_*.npz --strategy=last-block
"""

import numpy as np
import argparse
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.ndimage import gaussian_filter1d, laplace

# Add CODE directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bci4_2b_loader_v2 import BCI4_2B_Loader
from eegnet_model import EEGNet


# Fine-tuning parameters (optimized for domain shift)
LEARNING_RATE = 5e-4  # Higher learning rate for faster adaptation (original: 1e-3 is too aggressive)
MAX_EPOCHS = 500      # Maximum epochs for aggressive personalization
EARLY_STOPPING_PATIENCE = 100  # Very patient - allow extensive training for personalization
VAL_SPLIT = 0.05      # Use only 5% for validation, rest for training (was 0.2)


class PersonalModelFineTuner:
    """Fine-tunes pretrained EEGNet on personal calibration data."""
    
    def __init__(self,
                 calibration_file: str,
                 base_model_path: str = "models/best_eegnet_2class_bci2b.keras",
                 strategy: str = "last-block",
                 mix_benchmark: bool = False,
                 personal_weight: float = 0.7,
                 from_scratch: bool = False,
                 apply_filter: bool = False,
                 personalize: bool = False,
                 no_validation: bool = False):
        """
        Initialize fine-tuner.
        
        Args:
            calibration_file: Path to personal calibration NPZ file
            base_model_path: Path to pretrained model
            strategy: Freezing strategy ('head-only', 'last-block', 'full')
            mix_benchmark: Whether to mix personal data with benchmark data
            personal_weight: Weight for personal data when mixing (0-1)
            from_scratch: Build and train a fresh EEGNet model (no pretrained weights)
            personalize: Aggressive personalization mode (minimal val split, max epochs, patient early stopping)
            no_validation: Train on 100% data without validation split (maximum personalization overfitting)
        """
        self.calibration_file = Path(calibration_file)
        self.base_model_path = Path(__file__).parent / base_model_path
        self.strategy = strategy
        self.mix_benchmark = mix_benchmark
        self.personal_weight = personal_weight
        self.from_scratch = from_scratch
        self.apply_filter = apply_filter
        self.personalize = personalize
        self.no_validation = no_validation

        if self.from_scratch and self.strategy != "full":
            self.strategy = "full"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Extract user_id from filename
        self.user_id = self.calibration_file.stem.split('_calibration_')[0]
        
        # Paths
        self.save_dir = Path(__file__).parent / "models" / "personalized"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Personal Model Fine-Tuner initialized")
        self.logger.info(f"  User ID: {self.user_id}")
        self.logger.info(f"  Strategy: {self.strategy}")
        if self.from_scratch and strategy != "full":
            self.logger.info("  Note: strategy overridden to 'full' for from-scratch training")
        if self.from_scratch:
            self.logger.info("  Model mode: from scratch (no pretrained weights)")
        else:
            self.logger.info(f"  Base model: {self.base_model_path}")
        self.logger.info(f"  Laplacian filter: {'ENABLED' if self.apply_filter else 'DISABLED'}")
    
    def load_base_model(self) -> keras.Model:
        """Load pretrained base model."""
        if self.from_scratch:
            self.logger.info("Building fresh EEGNet model from config_2b.yaml...")
            eegnet = EEGNet(config_path="config_2b.yaml")
            model = eegnet.build_model()
            self.logger.info(f"✅ Built fresh model with {model.count_params():,} parameters")
            self.logger.info(f"   Layers: {len(model.layers)}")
            self.logger.info("   Layer structure:")
            for idx, layer in enumerate(model.layers):
                self.logger.info(f"     [{idx:2d}] {layer.name:30s} {layer.__class__.__name__}")
            return model

        self.logger.info("Loading pretrained base model...")
        
        if not self.base_model_path.exists():
            raise FileNotFoundError(
                f"Base model not found: {self.base_model_path}\n"
                "Please train base model first using train_model_2b.py"
            )
        
        model = keras.models.load_model(self.base_model_path)
        
        self.logger.info(f"✅ Loaded model with {model.count_params():,} parameters")
        self.logger.info(f"   Layers: {len(model.layers)}")
        
        # Show layer structure
        self.logger.info("   Layer structure:")
        for idx, layer in enumerate(model.layers):
            self.logger.info(f"     [{idx:2d}] {layer.name:30s} {layer.__class__.__name__}")
        
        return model
    
    def freeze_layers(self, model: keras.Model) -> Tuple[keras.Model, int]:
        """
        Freeze layers according to strategy.
        
        EEGNet structure:
        - Block 1 (layers 0-6): Temporal conv + Depthwise conv + pooling + dropout
        - Block 2 (layers 7-12): Separable conv + pooling + dropout
        - Head (layers 13-15): Flatten + Dense + Softmax
        
        Args:
            model: Model to freeze
        
        Returns:
            model: Model with frozen layers
            n_trainable: Number of trainable parameters
        """
        self.logger.info(f"Applying '{self.strategy}' freezing strategy...")
        
        if self.strategy == "head-only":
            # Freeze everything except final Dense layer
            # Keep layers[0:-2] frozen (all except Dense + Softmax)
            for layer in model.layers[:-2]:
                layer.trainable = False
            self.logger.info("  ❄️  Frozen: Block 1 + Block 2 (all conv layers)")
            self.logger.info("  🔥 Training: Dense classification head only")
        
        elif self.strategy == "last-block":
            # Freeze Block 1, train Block 2 + head
            # Freeze layers[0:7] (Block 1)
            for layer in model.layers[:7]:
                layer.trainable = False
            self.logger.info("  ❄️  Frozen: Block 1 (temporal + spatial filters)")
            self.logger.info("  🔥 Training: Block 2 (separable conv) + Dense head")
        
        elif self.strategy == "full":
            # Train entire model
            for layer in model.layers:
                layer.trainable = True
            self.logger.info("  🔥 Training: Entire model (all layers)")
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Count trainable parameters
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        total_params = model.count_params()
        frozen_params = total_params - trainable_params
        
        self.logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        self.logger.info(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        
        return model, trainable_params
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load calibration data.
        
        Returns:
            X: Training epochs (n_trials, 3, 1000)
            y: Labels (n_trials,)
            rest_epochs: REST epochs for threshold calibration
        """
        self.logger.info("Loading calibration data...")
        
        loader = BCI4_2B_Loader("config_2b.yaml")
        
        if self.mix_benchmark:
            self.logger.info(f"  Mixing personal + benchmark (personal weight: {self.personal_weight})")
            X, y = loader.load_mixed_data(
                personal_npz=str(self.calibration_file),
                personal_weight=self.personal_weight,
                benchmark_sessions=['T']
            )
            # Load REST separately
            _, _, rest_epochs = loader.load_personal_calibration(str(self.calibration_file))
        else:
            X, y, rest_epochs = loader.load_personal_calibration(str(self.calibration_file))
        
        self.logger.info(f"✅ Loaded {len(X)} training epochs")
        self.logger.info(f"   LEFT (0):  {np.sum(y == 0)} epochs")
        self.logger.info(f"   RIGHT (1): {np.sum(y == 1)} epochs")
        self.logger.info(f"   REST: {len(rest_epochs)} epochs")
        
        return X, y, rest_epochs

    def _to_model_input(self, X: np.ndarray) -> np.ndarray:
        """
        Convert epochs to EEGNet input layout: (trials, channels, samples, 1).

        Supports both common epoch layouts:
        - (trials, samples, channels)
        - (trials, channels, samples)
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D epochs array, got shape={X.shape}")

        if X.shape[1] == 3:
            # (trials, channels, samples) -> already channel-first
            X_model = X
        elif X.shape[2] == 3:
            # (trials, samples, channels) -> transpose to channel-first
            X_model = X.transpose(0, 2, 1)
        else:
            raise ValueError(
                f"Cannot infer channel axis from shape={X.shape}. "
                "Expected either (n, 3, samples) or (n, samples, 3)."
            )

        return X_model[..., np.newaxis]
    
    def apply_laplacian_filter(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Laplacian filter to each channel of EEG data.
        
        Laplacian filter emphasizes high-frequency components and 
        edge artifacts commonly associated with motor imagery.
        
        Args:
            X: Input array (n_trials, n_channels, n_samples) or (n_trials, n_channels, n_samples, 1)
        
        Returns:
            X_filtered: Filtered data (same shape as input)
        """
        # Remove last dimension if present
        if X.ndim == 4:
            X_work = X[..., 0]  # Convert to (n_trials, n_channels, n_samples)
            add_dim = True
        else:
            X_work = X.copy()
            add_dim = False
        
        X_filtered = np.zeros_like(X_work)
        
        # Apply 1D Laplacian along time axis for each trial and channel
        for trial_idx in range(X_work.shape[0]):
            for ch_idx in range(X_work.shape[1]):
                # Simple 1D Laplacian: f''(x) ≈ f(x-1) - 2*f(x) + f(x+1)
                X_filtered[trial_idx, ch_idx, :] = laplace(X_work[trial_idx, ch_idx, :])
        
        # Restore dimension if needed
        if add_dim:
            X_filtered = X_filtered[..., np.newaxis]
        
        return X_filtered
    
    def augment_data(self, X: np.ndarray, y: np.ndarray, 
                     factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data with small perturbations.
        
        Useful for small personal datasets (< 100 trials).
        Applies:
        - Small Gaussian noise
        - Temporal shifts (jitter)
        - Amplitude scaling
        
        Args:
            X: Training data (n_trials, n_channels, n_samples, 1)
            y: Labels (n_trials, n_classes) one-hot encoded
            factor: Augmentation factor (e.g., 2 = double dataset)
        
        Returns:
            X_aug, y_aug: Augmented data
        """
        n_trials = X.shape[0]
        
        if factor <= 1:
            return X, y
        
        self.logger.info(f"Augmenting data with factor={factor}...")
        
        X_aug_list = [X]
        y_aug_list = [y]
        
        np.random.seed(42)
        
        for _ in range(factor - 1):
            # Random selection of augmentation for each trial
            X_new = X.copy()
            
            for i in range(n_trials):
                augmentation_type = np.random.choice(['noise', 'shift', 'scale', 'mixed'])
                
                if augmentation_type == 'noise':
                    # Add small Gaussian noise
                    noise = np.random.normal(0, X[i].std() * 0.05, X[i].shape)
                    X_new[i] = X[i] + noise
                
                elif augmentation_type == 'shift':
                    # Temporal shift (circular)
                    shift = np.random.randint(-10, 11)
                    X_new[i] = np.roll(X[i], shift, axis=-2)
                
                elif augmentation_type == 'scale':
                    # Amplitude scaling
                    scale = np.random.uniform(0.9, 1.1)
                    X_new[i] = X[i] * scale
                
                else:  # mixed
                    # Combine 2 augmentations
                    noise = np.random.normal(0, X[i].std() * 0.03, X[i].shape)
                    scale = np.random.uniform(0.95, 1.05)
                    X_new[i] = (X[i] + noise) * scale
            
            X_aug_list.append(X_new)
            y_aug_list.append(y)
        
        X_aug = np.vstack(X_aug_list)
        y_aug = np.vstack(y_aug_list)
        
        self.logger.info(f"  Augmented data: {X.shape} -> {X_aug.shape}")
        
        return X_aug, y_aug
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights to handle imbalanced data.
        
        Useful when one class (e.g., RIGHT hand) has fewer trials.
        
        Args:
            y: One-hot encoded labels (n_trials, n_classes)
        
        Returns:
            class_weights: Dict mapping class index to weight
        """
        # Get class counts from one-hot encoding
        class_counts = np.sum(y, axis=0)
        total = np.sum(class_counts)
        
        # Inverse frequency weighting
        class_weights = {}
        for i, count in enumerate(class_counts):
            if count > 0:
                # Weight inversely proportional to frequency
                class_weights[i] = total / (len(class_counts) * count)
            else:
                class_weights[i] = 1.0
        
        # Normalize so mean weight = 1
        mean_weight = np.mean(list(class_weights.values()))
        class_weights = {k: v / mean_weight for k, v in class_weights.items()}
        
        self.logger.info(f"Class weights computed:")
        for class_idx, weight in sorted(class_weights.items()):
            self.logger.info(f"  Class {class_idx}: {weight:.4f}")
        
        return class_weights
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                     apply_filters: bool = True,
                     augment: bool = False) -> Tuple:
        """
        Prepare data for training with preprocessing.
        
        Args:
            X: Raw training epochs (n_trials, 3, 1000)
            y: Labels (n_trials,)
            apply_filters: Whether to apply Laplacian filter
            augment: Whether to augment data (for small datasets < 100 trials)
        
        Returns:
            X_train, X_val, y_train, y_val: Preprocessed and split data
        """
        self.logger.info(f"Preparing data for training...")
        
        # Step 1: Apply z-score normalization per trial
        self.logger.info(f"  Step 1: Z-score normalization...")
        X_normalized = np.array([
            (trial - np.mean(trial, axis=1, keepdims=True)) / 
            (np.std(trial, axis=1, keepdims=True) + 1e-8)
            for trial in X
        ])
        self.logger.info(f"    ✓ Normalized shape: {X_normalized.shape}")
        
        # Step 2: Apply Laplacian filter if requested
        if apply_filters:
            self.logger.info(f"  Step 2: Applying Laplacian filter...")
            X_normalized = self.apply_laplacian_filter(X_normalized)
            self.logger.info(f"    ✓ Filtered shape: {X_normalized.shape}")
        
        # Step 3: Reshape for model
        self.logger.info(f"  Step 3: Reshaping for model...")
        X_model = self._to_model_input(X_normalized)
        self.logger.info(f"    ✓ Model input shape: {X_model.shape}")
        
        # Step 4: One-hot encode labels
        y_onehot = tf.keras.utils.to_categorical(y, num_classes=2)
        
        # Step 5: Data augmentation if small dataset
        n_trials = len(X)
        aug_factor = 1
        
        if augment and n_trials < 100:
            aug_factor = max(2, 100 // n_trials)
            self.logger.info(f"  Step 4: Data augmentation (factor={aug_factor})...")
            X_model, y_onehot = self.augment_data(X_model, y_onehot, factor=aug_factor)
        
        # Step 6: Train/val split (stratified to maintain class balance)
        self.logger.info(f"  Step 5: Train/val split...")
        
        # Get original labels for stratification
        y_for_split = np.argmax(y_onehot, axis=1)
        
        # Determine validation split
        if self.no_validation:
            # Train on 100% of data, use dummy validation (won't be used)
            val_split = 0.0
            self.logger.info(f"   🔥 NO VALIDATION MODE: Training on 100% of data")
            self.logger.info(f"   ⚠️ WARNING: Full overfitting mode - model will memorize your data!")
            X_train, X_val, y_train, y_val = X_model, X_model, y_onehot, y_onehot
        else:
            # Use minimal validation split in personalization mode
            val_split = 0.05 if self.personalize else VAL_SPLIT
            X_train, X_val, y_train, y_val = train_test_split(
                X_model, y_onehot,
                test_size=val_split,
                stratify=y_for_split,
                random_state=42
            )
        
        self.logger.info(f"✅ Data preparation complete!")
        if self.no_validation:
            self.logger.info(f"   🔥 NO VALIDATION: Using 100% for training")
        elif self.personalize:
            self.logger.info(f"   🔥 PERSONALIZATION MODE: Minimal validation split (5%)")
        self.logger.info(f"   Final training set: {X_train.shape}")
        self.logger.info(f"   Final validation set: {X_val.shape}")
        self.logger.info(f"   Class distribution (training):")
        for cls in range(2):
            count = np.sum(np.argmax(y_train, axis=1) == cls)
            pct = count / len(y_train) * 100
            self.logger.info(f"     Class {cls}: {count} epochs ({pct:.1f}%)")
        
        return X_train, X_val, y_train, y_val
    
    def fine_tune(self, model: keras.Model, X_train, X_val, y_train, y_val) -> Dict:
        """
        Fine-tune model on personal data.
        
        Args:
            model: Model with frozen layers
            X_train, X_val, y_train, y_val: Training data
        
        Returns:
            history: Training history dict
        """
        self.logger.info("Fine-tuning model...")
        
        # Compile with low learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Compute class weights for imbalanced data
        class_weights = self.compute_class_weights(y_train)
        
        # Determine training parameters based on personalization mode
        if self.no_validation:
            # Full overfitting - use aggressive settings
            max_epochs = 1000
            early_stop_patience = 500  # Nearly never stop early
            self.logger.info(f"  🔥 FULL MEMORIZATION SETTINGS: 1000 epochs, 500 patience")
        else:
            max_epochs = 500 if self.personalize else MAX_EPOCHS
            early_stop_patience = 100 if self.personalize else EARLY_STOPPING_PATIENCE
        
        # Callbacks
        checkpoint_path = self.save_dir / f"{self.user_id}_finetuned_best.keras"
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=early_stop_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train with class weights
        self.logger.info(f"Training configuration:")
        if self.no_validation:
            self.logger.info(f"  🔥🔥🔥 FULL MEMORIZATION MODE - OVERFITTING ON YOUR DATA 🔥🔥🔥")
        elif self.personalize:
            self.logger.info(f"  🔥 PERSONALIZATION MODE ENABLED")
        self.logger.info(f"  Learning rate: {LEARNING_RATE}")
        self.logger.info(f"  Max epochs: {max_epochs}")
        self.logger.info(f"  Early stopping patience: {early_stop_patience}")
        self.logger.info(f"  Batch size: 32")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=2
        )
        
        # Get best results
        best_epoch = np.argmax(history.history['val_accuracy']) + 1
        best_val_acc = np.max(history.history['val_accuracy'])
        best_train_acc = history.history['accuracy'][best_epoch - 1]
        
        # Compute per-class metrics on validation set
        val_predictions = model.predict(X_val, verbose=0)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        val_true_classes = np.argmax(y_val, axis=1)
        
        # Confusion matrix
        cm = confusion_matrix(val_true_classes, val_pred_classes)
        
        # Per-class accuracy
        per_class_acc = []
        for i in range(2):
            class_mask = val_true_classes == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(val_true_classes[class_mask], val_pred_classes[class_mask])
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0.0)
        
        self.logger.info(f"\n✅ Fine-tuning complete!")
        self.logger.info(f"   Best epoch: {best_epoch}/{len(history.history['val_accuracy'])}")
        self.logger.info(f"   Training accuracy: {best_train_acc:.4f}")
        self.logger.info(f"   Validation accuracy: {best_val_acc:.4f}")
        self.logger.info(f"\n✅ Per-class validation accuracy:")
        self.logger.info(f"   Class 0 (LEFT): {per_class_acc[0]:.4f}")
        self.logger.info(f"   Class 1 (RIGHT): {per_class_acc[1]:.4f}")
        self.logger.info(f"\n✅ Confusion matrix (rows=true, cols=predicted):")
        self.logger.info(f"   {cm}")
        self.logger.info(f"\n✅ Classification report:")
        self.logger.info(f"{classification_report(val_true_classes, val_pred_classes, target_names=['LEFT', 'RIGHT'])}")
        
        # Add metrics to history
        history.history['per_class_accuracy'] = per_class_acc
        history.history['confusion_matrix'] = cm.tolist()
        
        return history.history
    
    def calibrate_neutral_threshold(self, 
                                    model: keras.Model,
                                    rest_epochs: np.ndarray) -> Dict:
        """
        Calibrate neutral threshold using REST trials.
        
        Computes 75th percentile of max confidence during rest state.
        Falls back to default threshold if REST trials not available.
        
        Args:
            model: Fine-tuned model
            rest_epochs: REST epochs (n_rest, 3, 1000) or empty array
        
        Returns:
            threshold_info: Dict with threshold and statistics
        """
        self.logger.info("Calibrating neutral threshold...")
        
        # Check if REST epochs are available
        if len(rest_epochs) == 0:
            # No REST trials available (e.g., from preprocessed data)
            # Use default threshold based on model expectations
            self.logger.warning("⚠️  No REST trials available (preprocessed data)")
            self.logger.info("   Using default threshold: 0.70")
            
            threshold_info = {
                'threshold': 0.70,  # Default conservative threshold
                'mean': None,
                'median': None,
                'std': None,
                'p75': 0.70,
                'p90': None,
                'recommended_min': 0.65,
                'recommended_max': 0.75,
                'n_rest_trials': 0,
                'source': 'default'
            }
            self.logger.info(f"✅ Default threshold set:")
            self.logger.info(f"   Threshold: {threshold_info['threshold']:.4f}")
            self.logger.info(f"   Recommended range: {threshold_info['recommended_min']:.4f} - {threshold_info['recommended_max']:.4f}")
            self.logger.info(f"\n   To improve: Record REST trials with NPG device and re-calibrate")
            return threshold_info
        
        # Prepare REST data for model
        X_rest = self._to_model_input(rest_epochs)
        
        # Get predictions
        predictions = model.predict(X_rest, verbose=0)
        
        # Compute max confidence for each REST epoch
        max_confidences = np.max(predictions, axis=1)
        
        # Compute 75th percentile
        threshold = np.percentile(max_confidences, 75)
        
        # Statistics
        mean_conf = np.mean(max_confidences)
        std_conf = np.std(max_confidences)
        median_conf = np.median(max_confidences)
        p90_conf = np.percentile(max_confidences, 90)
        
        self.logger.info(f"✅ Neutral threshold calibrated from {len(rest_epochs)} REST trials:")
        self.logger.info(f"   Threshold (75th percentile): {threshold:.4f}")
        self.logger.info(f"   Mean REST confidence: {mean_conf:.4f}")
        self.logger.info(f"   Median REST confidence: {median_conf:.4f}")
        self.logger.info(f"   Std deviation: {std_conf:.4f}")
        self.logger.info(f"   90th percentile: {p90_conf:.4f}")
        self.logger.info(f"\n   Recommended range: {threshold-0.05:.4f} - {threshold+0.05:.4f}")
        
        threshold_info = {
            'threshold': float(threshold),
            'mean': float(mean_conf),
            'median': float(median_conf),
            'std': float(std_conf),
            'p75': float(threshold),
            'p90': float(p90_conf),
            'recommended_min': float(threshold - 0.05),
            'recommended_max': float(threshold + 0.05),
            'n_rest_trials': len(rest_epochs),
            'source': 'REST_trials'
        }
        
        return threshold_info
    
    def save_model_and_metadata(self,
                                model: keras.Model,
                                threshold_info: Dict,
                                history: Dict,
                                n_trainable_params: int):
        """
        Save fine-tuned model, threshold, and metadata.
        Also registers model in model_registry.json.
        
        Args:
            model: Fine-tuned model
            threshold_info: Neutral threshold information
            history: Training history
            n_trainable_params: Number of trainable parameters
        """
        # Save model
        model_path = self.save_dir / f"{self.user_id}_finetuned.keras"
        model.save(model_path)
        self.logger.info(f"💾 Saved model: {model_path}")
        
        # Save threshold
        threshold_path = self.save_dir / f"{self.user_id}_neutral_threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump(threshold_info, f, indent=2)
        self.logger.info(f"💾 Saved threshold: {threshold_path}")
        
        # Save metadata
        metadata = {
            'user_id': self.user_id,
            'timestamp': datetime.now().isoformat(),
            'calibration_file': str(self.calibration_file),
            'base_model': str(self.base_model_path),
            'fine_tuning_strategy': self.strategy,
            'mixed_with_benchmark': self.mix_benchmark,
            'personal_weight': self.personal_weight if self.mix_benchmark else 1.0,
            'learning_rate': LEARNING_RATE,
            'max_epochs': MAX_EPOCHS,
            'best_epoch': int(np.argmax(history['val_accuracy']) + 1),
            'best_val_accuracy': float(np.max(history['val_accuracy'])),
            'best_train_accuracy': float(history['accuracy'][np.argmax(history['val_accuracy'])]),
            'trainable_parameters': int(n_trainable_params),
            'per_class_accuracy': {
                'class_0_left': float(history['per_class_accuracy'][0]),
                'class_1_right': float(history['per_class_accuracy'][1])
            },
            'confusion_matrix': {
                'values': history['confusion_matrix'],
                'labels': ['LEFT (predicted)', 'RIGHT (predicted)'],
                'note': 'rows=true class, cols=predicted class'
            },
            'preprocessing': {
                'z_score_normalization': True,
                'laplacian_filter_applied': self.apply_filter,
                'data_augmentation_applied': 'auto (if < 100 trials)'
            },
            'neutral_threshold': threshold_info
        }
        
        metadata_path = self.save_dir / f"{self.user_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"💾 Saved metadata: {metadata_path}")
        
        # Register in model registry (best effort)
        registry_metadata = {
            'timestamp': metadata['timestamp'],
            'val_accuracy': metadata['best_val_accuracy'],
            'fine_tuning_strategy': self.strategy,
            'calibration_file': str(self.calibration_file.name)
        }

        try:
            from model_factory import ModelFactory
            if hasattr(ModelFactory, 'register_personalized_model'):
                ModelFactory.register_personalized_model(
                    user_id=self.user_id,
                    model_path=str(model_path),
                    threshold=threshold_info['threshold'],
                    metadata=registry_metadata
                )
                self.logger.info(f"📝 Registered in model registry")
            else:
                registry_path = Path(__file__).parent / "model_registry.json"
                try:
                    with open(registry_path, 'r') as f:
                        registry = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    registry = {}

                if not isinstance(registry, dict):
                    registry = {}

                personalized = registry.get('personalized_models', {})
                personalized[self.user_id] = {
                    'model_path': str(model_path),
                    'threshold': threshold_info['threshold'],
                    'metadata': registry_metadata
                }
                registry['personalized_models'] = personalized

                with open(registry_path, 'w') as f:
                    json.dump(registry, f, indent=2)
                self.logger.info(f"📝 Registered in fallback registry: {registry_path}")
        except Exception as registry_error:
            self.logger.warning(f"⚠️ Registry update skipped: {registry_error}")
        
        return model_path, threshold_path, metadata_path
    
    def run(self) -> Dict:
        """
        Run complete fine-tuning pipeline.
        
        Returns:
            results: Dict with paths and statistics
        """
        print("\n" + "="*70)
        print("PERSONAL MODEL FINE-TUNING")
        print("="*70 + "\n")
        
        # 1. Load base model
        model = self.load_base_model()
        
        # 2. Freeze layers
        model, n_trainable = self.freeze_layers(model)
        
        # 3. Load data
        X, y, rest_epochs = self.load_data()
        
        # 4. Prepare data with preprocessing
        # Auto-augment if small dataset
        should_augment = len(X) < 100
        if should_augment:
            self.logger.info(f"Small dataset detected ({len(X)} trials), enabling augmentation...")
        
        X_train, X_val, y_train, y_val = self.prepare_data(
            X, y,
            apply_filters=self.apply_filter,
            augment=should_augment
        )
        
        # 5. Fine-tune
        history = self.fine_tune(model, X_train, X_val, y_train, y_val)
        
        # 6. Calibrate neutral threshold
        threshold_info = self.calibrate_neutral_threshold(model, rest_epochs)
        
        # 7. Save everything
        model_path, threshold_path, metadata_path = self.save_model_and_metadata(
            model, threshold_info, history, n_trainable
        )
        
        print("\n" + "="*70)
        print("FINE-TUNING COMPLETE!")
        print("="*70)
        print(f"\n✅ Personalized model ready for: {self.user_id}")
        print(f"   Model: {model_path}")
        print(f"   Threshold: {threshold_path}")
        print(f"   Metadata: {metadata_path}")
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print(f"\n1. Validate 3-state system:")
        print(f"   python validate_3state_system.py --user-id={self.user_id}")
        print(f"\n2. Use in real-time BCI:")
        print(f"   python npg_realtime_bci.py --user-id={self.user_id}")
        print("="*70 + "\n")
        
        return {
            'model_path': str(model_path),
            'threshold_path': str(threshold_path),
            'metadata_path': str(metadata_path),
            'val_accuracy': float(np.max(history['val_accuracy'])),
            'neutral_threshold': threshold_info['threshold']
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fine-tune EEGNet on personal calibration data'
    )
    parser.add_argument(
        '--calibration-file',
        type=str,
        required=True,
        help='Path to calibration NPZ file'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='models/best_eegnet_2class_bci2b.keras',
        help='Path to pretrained base model'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['head-only', 'last-block', 'full'],
        default='last-block',
        help='Fine-tuning strategy (default: last-block)'
    )
    parser.add_argument(
        '--mix-benchmark',
        action='store_true',
        help='Mix personal data with benchmark data (for small datasets)'
    )
    parser.add_argument(
        '--personal-weight',
        type=float,
        default=0.7,
        help='Weight for personal data when mixing (0-1, default: 0.7)'
    )
    parser.add_argument(
        '--from-scratch',
        action='store_true',
        help='Train a fresh EEGNet model from scratch (no pretrained weights)'
    )
    parser.add_argument(
        '--filter',
        action='store_true',
        help='Enable Laplacian filter (default: disabled for better performance)'
    )
    parser.add_argument(
        '--personalize',
        action='store_true',
        help='AGGRESSIVE PERSONALIZATION MODE: Minimal validation (5%), max epochs (500), patient early stopping (100 patience). Targets 85%+ accuracy by learning your unique patterns.'
    )
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='FULL MEMORIZATION MODE: Train on 100%% of data without validation split. Maximum overfitting - model will memorize your exact patterns. Use for maximum personalization.'
    )
    
    args = parser.parse_args()
    
    # Validate
    if not Path(args.calibration_file).exists():
        print(f"❌ Calibration file not found: {args.calibration_file}")
        return 1
    
    # Create fine-tuner
    tuner = PersonalModelFineTuner(
        calibration_file=args.calibration_file,
        base_model_path=args.base_model,
        strategy=args.strategy,
        mix_benchmark=args.mix_benchmark,
        personal_weight=args.personal_weight,
        from_scratch=args.from_scratch,
        apply_filter=args.filter,
        personalize=args.personalize,
        no_validation=args.no_validation
    )
    
    # Run fine-tuning
    try:
        results = tuner.run()
        return 0
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
