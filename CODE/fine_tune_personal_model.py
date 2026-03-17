#!/usr/bin/env python3
"""
Personal Model Fine-Tuning Script

Fine-tunes pretrained EEGNet model on personal calibration data using transfer learning.
Implements three strategies with layer freezing:
- head-only: Freeze all except final Dense layer (30+ trials needed)
- last-block: Freeze Block 1, train Block 2 + Dense head (100+ trials, recommended)
- full: Train entire model (500+ trials needed)

Also calibrates neutral threshold using REST trials for 3-state classification.

Usage:
    # Head-only fine-tuning (fastest, minimal data)
    python fine_tune_personal_model.py --calibration-file=calibration_data/john_doe_*.npz --strategy=head-only
    
    # Last-block fine-tuning (recommended, balanced)
    python fine_tune_personal_model.py --calibration-file=calibration_data/john_doe_*.npz --strategy=last-block
    
    # Full fine-tuning (requires lots of data)
    python fine_tune_personal_model.py --calibration-file=calibration_data/john_doe_*.npz --strategy=full
    
    # Mix with benchmark data (for small personal datasets)
    python fine_tune_personal_model.py --calibration-file=calibration_data/john_doe_*.npz --mix-benchmark --personal-weight=0.7
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Add CODE directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bci4_2b_loader_v2 import BCI4_2B_Loader


# Fine-tuning parameters
LEARNING_RATE = 1e-4  # 10x lower than base training (1e-3)
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
VAL_SPLIT = 0.2


class PersonalModelFineTuner:
    """Fine-tunes pretrained EEGNet on personal calibration data."""
    
    def __init__(self,
                 calibration_file: str,
                 base_model_path: str = "models/best/eegnet_2class_bci2b.keras",
                 strategy: str = "last-block",
                 mix_benchmark: bool = False,
                 personal_weight: float = 0.7):
        """
        Initialize fine-tuner.
        
        Args:
            calibration_file: Path to personal calibration NPZ file
            base_model_path: Path to pretrained model
            strategy: Freezing strategy ('head-only', 'last-block', 'full')
            mix_benchmark: Whether to mix personal data with benchmark data
            personal_weight: Weight for personal data when mixing (0-1)
        """
        self.calibration_file = Path(calibration_file)
        self.base_model_path = Path(__file__).parent / base_model_path
        self.strategy = strategy
        self.mix_benchmark = mix_benchmark
        self.personal_weight = personal_weight
        
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
        self.logger.info(f"  Strategy: {strategy}")
        self.logger.info(f"  Base model: {self.base_model_path}")
    
    def load_base_model(self) -> keras.Model:
        """Load pretrained base model."""
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
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            X: Epochs (n_trials, 3, 1000)
            y: Labels (n_trials,)
        
        Returns:
            X_train, X_val, y_train, y_val
        """
        # Reshape for model: (trials, channels, samples, 1)
        X = X[..., np.newaxis]
        
        # One-hot encode labels
        y_onehot = tf.keras.utils.to_categorical(y, num_classes=2)
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_onehot,
            test_size=VAL_SPLIT,
            stratify=y,
            random_state=42
        )
        
        self.logger.info(f"Data split:")
        self.logger.info(f"  Training:   {len(X_train)} epochs")
        self.logger.info(f"  Validation: {len(X_val)} epochs")
        
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
        
        # Callbacks
        checkpoint_path = self.save_dir / f"{self.user_id}_finetuned_best.keras"
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        self.logger.info(f"  Learning rate: {LEARNING_RATE}")
        self.logger.info(f"  Max epochs: {MAX_EPOCHS}")
        self.logger.info(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=32,
            callbacks=callbacks,
            verbose=2
        )
        
        # Get best results
        best_epoch = np.argmax(history.history['val_accuracy']) + 1
        best_val_acc = np.max(history.history['val_accuracy'])
        best_train_acc = history.history['accuracy'][best_epoch - 1]
        
        self.logger.info(f"\n✅ Fine-tuning complete!")
        self.logger.info(f"   Best epoch: {best_epoch}/{len(history.history['val_accuracy'])}")
        self.logger.info(f"   Training accuracy: {best_train_acc:.4f}")
        self.logger.info(f"   Validation accuracy: {best_val_acc:.4f}")
        
        return history.history
    
    def calibrate_neutral_threshold(self, 
                                    model: keras.Model,
                                    rest_epochs: np.ndarray) -> Dict:
        """
        Calibrate neutral threshold using REST trials.
        
        Computes 75th percentile of max confidence during rest state.
        
        Args:
            model: Fine-tuned model
            rest_epochs: REST epochs (n_rest, 3, 1000)
        
        Returns:
            threshold_info: Dict with threshold and statistics
        """
        self.logger.info("Calibrating neutral threshold from REST trials...")
        
        # Prepare REST data for model
        X_rest = rest_epochs[..., np.newaxis]  # Add channel dimension
        
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
        
        self.logger.info(f"✅ Neutral threshold calibrated:")
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
            'n_rest_trials': len(rest_epochs)
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
            'neutral_threshold': threshold_info
        }
        
        metadata_path = self.save_dir / f"{self.user_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"💾 Saved metadata: {metadata_path}")
        
        # Register in model registry
        from model_factory import ModelFactory
        ModelFactory.register_personalized_model(
            user_id=self.user_id,
            model_path=str(model_path),
            threshold=threshold_info['threshold'],
            metadata={
                'timestamp': metadata['timestamp'],
                'val_accuracy': metadata['best_val_accuracy'],
                'fine_tuning_strategy': self.strategy,
                'calibration_file': str(self.calibration_file.name)
            }
        )
        self.logger.info(f"📝 Registered in model registry")
        
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
        
        # 4. Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(X, y)
        
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
        default='models/best/eegnet_2class_bci2b.keras',
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
        personal_weight=args.personal_weight
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
