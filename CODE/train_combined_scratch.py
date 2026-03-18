#!/usr/bin/env python3
"""
Train EEGNet from scratch using augmented combined benchmark + personal dataset.

Augmentation techniques:
- Time-shift: ±80ms circular rotations
- Gaussian noise: SNR-appropriate additions
- Temporal mixup: Blend augmented pairs with alpha ∈ [0.3, 0.7]
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from eegnet_model import EEGNet
from normalization_utils import GlobalChannelZScore

# Set deterministic behavior
np.random.seed(42)
tf.random.set_seed(42)


def augment_time_shift(epochs: np.ndarray, shift_range_ms: int = 80, sr: int = 250) -> np.ndarray:
    """Time-shift augmentation: randomly rotate along time dimension."""
    max_shift_samples = int(shift_range_ms * sr / 1000.0)
    n_epochs = epochs.shape[0]
    
    augmented = np.zeros_like(epochs, dtype=np.float32)
    for i in range(n_epochs):
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        augmented[i] = np.roll(epochs[i], shift, axis=0)
    
    return augmented


def augment_gaussian_noise(epochs: np.ndarray, snr_db: float = 25.0) -> np.ndarray:
    """Gaussian noise augmentation: add small SNR-appropriate noise."""
    signal_power = np.mean(epochs ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)
    
    noise = np.random.normal(0, noise_std, epochs.shape)
    augmented = epochs + noise
    
    return augmented.astype(np.float32)


def augment_temporal_mixup(epochs: np.ndarray, alpha_min: float = 0.3, alpha_max: float = 0.7) -> np.ndarray:
    """Temporal mixup: blend adjacent augmented samples."""
    n_epochs = epochs.shape[0]
    augmented = np.zeros_like(epochs, dtype=np.float32)
    
    for i in range(n_epochs):
        alpha = np.random.uniform(alpha_min, alpha_max)
        j = (i + 1) % n_epochs
        augmented[i] = alpha * epochs[i] + (1.0 - alpha) * epochs[j]
    
    return augmented


def augment_compose(epochs: np.ndarray, sr: int = 250, num_augmentations: int = 3) -> np.ndarray:
    """Compose multiple augmentations and replicate dataset."""
    augmented_list = [epochs]  # Original
    
    for aug_idx in range(1, num_augmentations):
        aug = epochs.copy()
        
        # Apply 2-3 augmentation techniques per copy
        if aug_idx % 3 != 0:
            aug = augment_time_shift(aug, shift_range_ms=80, sr=sr)
        if aug_idx % 2 != 0:
            aug = augment_gaussian_noise(aug, snr_db=25.0)
        if aug_idx % 2 == 0:
            aug = augment_temporal_mixup(aug, alpha_min=0.3, alpha_max=0.7)
        
        augmented_list.append(aug)
    
    return np.concatenate(augmented_list, axis=0).astype(np.float32)


def to_model_input(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected 3D input (N, T, C) or (N, C, T), got {X.shape}")

    if X.shape[2] == 3:
        X = X.transpose(0, 2, 1)
    elif X.shape[1] != 3:
        raise ValueError(f"Could not infer channel axis from shape {X.shape}")

    return X[..., np.newaxis].astype(np.float32)


def calibrate_threshold(model: keras.Model, scaler: GlobalChannelZScore, rest_epochs: np.ndarray) -> dict:
    X_rest = to_model_input(rest_epochs)
    X_rest = scaler.transform(X_rest)
    probs = model.predict(X_rest, verbose=0)
    max_conf = np.max(probs, axis=1)
    threshold = float(np.percentile(max_conf, 75))
    return {
        "threshold": threshold,
        "mean": float(np.mean(max_conf)),
        "median": float(np.median(max_conf)),
        "std": float(np.std(max_conf)),
        "p90": float(np.percentile(max_conf, 90)),
        "n_rest_trials": int(len(rest_epochs))
    }


def main():
    parser = argparse.ArgumentParser(description="Train EEGNet from scratch on augmented combined dataset")
    parser.add_argument("--dataset-file", type=str, required=True, help="Path to combined dataset .npz")
    parser.add_argument("--user-id", type=str, required=True, help="User id for output naming")
    parser.add_argument("--config", type=str, default="config_2b.yaml", help="EEGNet config path")
    parser.add_argument("--augmentation-factor", type=int, default=1, help="Augmentation multiplier for personal data")
    parser.add_argument("--personal-weight", type=float, default=0.5, help="Target personal data weight")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    dataset_path = Path(args.dataset_file)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).parent / dataset_path

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info(f"Loading combined dataset: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]  # (N, T, C)
    y = data["y"]
    rest_epochs = data["rest_epochs"] if "rest_epochs" in data else None
    source = data["source"] if "source" in data else None
    
    logger.info(f"Original dataset: X={X.shape}, y={y.shape}, rest={rest_epochs.shape if rest_epochs is not None else None}")
    
    # Separate personal and benchmark data
    if source is not None and args.augmentation_factor > 1:
        logger.info(f"Separating personal vs benchmark data for augmentation")
        personal_mask = np.array([s == "personal" for s in source])
        benchmark_mask = ~personal_mask
        X_personal = X[personal_mask]
        y_personal = y[personal_mask]
        X_benchmark = X[benchmark_mask]
        y_benchmark = y[benchmark_mask]
        logger.info(f"Personal: {len(X_personal)}, Benchmark: {len(X_benchmark)}")
        
        # Augment personal data
        logger.info(f"Augmenting personal data with factor {args.augmentation_factor}x")
        X_personal_aug = augment_compose(X_personal, sr=250, num_augmentations=args.augmentation_factor)
        y_personal_aug = np.repeat(y_personal, args.augmentation_factor)
        logger.info(f"Augmented personal: {X_personal_aug.shape}")
        
        # Recalculate target benchmark count based on new augmented personal size
        target_benchmark = int(len(X_personal_aug) * (1.0 - args.personal_weight) / max(args.personal_weight, 1e-6))
        target_benchmark = min(target_benchmark, len(X_benchmark))
        indices = np.random.choice(len(X_benchmark), size=target_benchmark, replace=False)
        X_benchmark_sampled = X_benchmark[indices]
        y_benchmark_sampled = y_benchmark[indices]
        logger.info(f"Sampled benchmark: {len(X_benchmark_sampled)}")
        
        # Combine augmented personal + resampled benchmark
        X = np.concatenate([X_personal_aug, X_benchmark_sampled], axis=0)
        y = np.concatenate([y_personal_aug, y_benchmark_sampled], axis=0)
        actual_personal_weight = len(X_personal_aug) / len(X)
        logger.info(f"Combined dataset: {X.shape}, actual personal weight: {actual_personal_weight:.2%}")
    
    X_model = to_model_input(X)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_model, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42
    )

    val_fraction_of_trainval = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        stratify=y_trainval,
        random_state=42
    )

    scaler = GlobalChannelZScore().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes=2)
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=2)

    eegnet = EEGNet(config_path=args.config)
    model = eegnet.build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    save_dir = Path(__file__).parent / "models" / "personalized"
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = save_dir / f"{args.user_id}_combined_scratch_best_{ts}.keras"

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(best_path), monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(5, args.patience // 3), min_lr=1e-6, verbose=1)
    ]

    logger.info(f"Training from scratch: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    test_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(test_probs, axis=1)
    weighted_f1 = float(f1_score(y_test, y_pred, average='weighted'))
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['LEFT', 'RIGHT'], output_dict=True)

    model_path = save_dir / f"{args.user_id}_combined_scratch.keras"
    model.save(model_path)

    scaler_path = save_dir / f"{args.user_id}_combined_scaler.json"
    scaler.save(str(scaler_path))

    threshold_info = None
    if rest_epochs is not None and len(rest_epochs) > 0:
        threshold_info = calibrate_threshold(model, scaler, rest_epochs)
        threshold_path = save_dir / f"{args.user_id}_combined_neutral_threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump(threshold_info, f, indent=2)
    else:
        threshold_path = None

    best_val_acc = float(np.max(history.history['val_accuracy']))
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "dataset_file": str(dataset_path),
        "user_id": args.user_id,
        "augmentation_factor": args.augmentation_factor,
        "personal_weight": args.personal_weight,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "epochs_ran": int(len(history.history['loss'])),
        "best_val_accuracy": best_val_acc,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "test_weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "model_path": str(model_path),
        "best_checkpoint": str(best_path),
        "scaler_path": str(scaler_path),
        "threshold_path": str(threshold_path) if threshold_path else None,
        "threshold_info": threshold_info
    }

    metrics_path = save_dir / f"{args.user_id}_combined_training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete")
    logger.info(f"Best val accuracy: {best_val_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    
    # Check target
    if best_val_acc >= 0.76:
        logger.info(f"✓ TARGET ACHIEVED: Val accuracy {best_val_acc:.4f} >= 76%")
    else:
        logger.warning(f"✗ Target not met: Val accuracy {best_val_acc:.4f} < 76%")
    
    logger.info(f"Model: {model_path}")
    logger.info(f"Scaler: {scaler_path}")
    if threshold_path:
        logger.info(f"Threshold: {threshold_path}")
    logger.info(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
