#!/usr/bin/env python3
"""
Train transfer learning model on augmented combined dataset.

Combines BCI IV 2b benchmark + augmented personal calibration data.
Uses data augmentation techniques to increase personal dataset contribution.

Augmentation techniques:
- Time-shift: ±50-100ms rotations in time dimension
- Gaussian noise: SNR-aware small amplitude additions
- Temporal mixup: Blend adjacent augmented samples (alpha ∈ [0.2, 0.8])

Targets: >76% validation accuracy with high personal data weight.

Usage:
    python train_combined_augmented.py \
        --dataset-file=datasets/combined_bci2b_personal_alan.npz \
        --user-id=alan \
        --personal-weight=0.8 \
        --augmentation-factor=3 \
        --strategy=last-block \
        --epochs=100 \
        --patience=20 \
        --learning-rate=5e-4
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, f1_score

from normalization_utils import GlobalChannelZScore

# Set deterministic behavior
np.random.seed(42)
tf.random.set_seed(42)


def to_model_input(X: np.ndarray) -> np.ndarray:
    """Convert (N, T, C) to (N, C, T, 1)."""
    if X.ndim == 4 and X.shape[3] == 1:
        return X
    if X.ndim == 3 and X.shape[2] in [3, 22]:  # 3 channels (personal) or 22 (benchmark)
        return X.transpose(0, 2, 1)[:, :, :, np.newaxis]
    raise ValueError(f"Unexpected shape: {X.shape}")


def augment_time_shift(epochs: np.ndarray, shift_range_ms: int = 100, sr: int = 250) -> np.ndarray:
    """
    Time-shift augmentation: randomly rotate along time dimension (circular shift).
    
    Args:
        epochs: (N, T, C) array
        shift_range_ms: max shift in milliseconds
        sr: sampling rate in Hz
    
    Returns:
        Augmented (N, T, C) array
    """
    max_shift_samples = int(shift_range_ms * sr / 1000.0)
    n_epochs = epochs.shape[0]
    time_dim = epochs.shape[1]
    
    augmented = np.zeros_like(epochs, dtype=np.float32)
    for i in range(n_epochs):
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        augmented[i] = np.roll(epochs[i], shift, axis=0)
    
    return augmented


def augment_gaussian_noise(epochs: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Gaussian noise augmentation: add small SNR-appropriate noise.
    
    Args:
        epochs: (N, T, C) array
        snr_db: signal-to-noise ratio in dB
    
    Returns:
        Augmented (N, T, C) array
    """
    signal_power = np.mean(epochs ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)
    
    noise = np.random.normal(0, noise_std, epochs.shape)
    augmented = epochs + noise
    
    return augmented.astype(np.float32)


def augment_temporal_mixup(epochs: np.ndarray, alpha_min: float = 0.2, alpha_max: float = 0.8) -> np.ndarray:
    """
    Temporal mixup: blend adjacent augmented samples.
    
    Args:
        epochs: (N, T, C) array
        alpha_min, alpha_max: mixup blend range
    
    Returns:
        Augmented (N, T, C) array
    """
    n_epochs = epochs.shape[0]
    augmented = np.zeros_like(epochs, dtype=np.float32)
    
    for i in range(n_epochs):
        # Pick random blend weight
        alpha = np.random.uniform(alpha_min, alpha_max)
        
        # Mix with adjacent sample (wrap around)
        j = (i + 1) % n_epochs
        augmented[i] = alpha * epochs[i] + (1.0 - alpha) * epochs[j]
    
    return augmented


def augment_compose(epochs: np.ndarray, sr: int = 250, num_augmentations: int = 3) -> np.ndarray:
    """
    Compose multiple augmentations and replicate dataset.
    
    Args:
        epochs: (N, T, C) array
        sr: sampling rate
        num_augmentations: number of copies (including original)
    
    Returns:
        (N * num_augmentations, T, C) array with diverse augmentations
    """
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


def calibrate_threshold(model: keras.Model, scaler: GlobalChannelZScore, rest_epochs: np.ndarray) -> dict:
    """Calibrate neutral state threshold from REST confidences."""
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


def apply_freezing_strategy(model: keras.Model, strategy: str, logger: logging.Logger) -> int:
    """Apply layer freezing strategy. Returns number of trainable params."""
    logger.info(f"Applying freezing strategy: {strategy}")

    if strategy == "head-only":
        for layer in model.layers[:-2]:
            layer.trainable = False
        trainable = sum(1 for layer in model.layers if layer.trainable)
        logger.info(f"  Frozen {len(model.layers) - trainable} layers; {trainable} trainable")
    
    elif strategy == "last-block":
        for layer in model.layers[:7]:  # Freeze block 1 and earlier
            layer.trainable = False
        trainable = sum(1 for layer in model.layers if layer.trainable)
        logger.info(f"  Frozen 7 layers; {trainable} trainable")
    
    elif strategy == "full":
        for layer in model.layers:
            layer.trainable = True
        logger.info(f"  All {len(model.layers)} layers trainable (full fine-tuning)")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    total_params = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    logger.info(f"  Total params: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    return trainable_params


def main():
    parser = argparse.ArgumentParser(description="Train on augmented combined dataset")
    parser.add_argument("--dataset-file", type=str, required=True, help="Combined dataset .npz path")
    parser.add_argument("--user-id", type=str, required=True, help="User identifier")
    parser.add_argument("--base-model", type=str, default="models/best/eegnet_2class_bci2b.keras", help="Pretrained model path")
    parser.add_argument("--strategy", type=str, default="last-block", choices=["head-only", "last-block", "full"])
    parser.add_argument("--personal-weight", type=float, default=0.8, help="Target personal data weight")
    parser.add_argument("--augmentation-factor", type=int, default=3, help="Augmentation multiplier")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split (within train)")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test split (from original)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load combined dataset
    logger.info(f"Loading dataset: {args.dataset_file}")
    data = np.load(args.dataset_file, allow_pickle=True)
    X = data["X"]  # (N, T, C) = (398, 1000, 3)
    y = data["y"]  # (N,)
    rest_epochs = data["rest_epochs"] if "rest_epochs" in data else None
    source = data["source"] if "source" in data else None
    logger.info(f"Original dataset shape: X={X.shape}, y={y.shape}, rest={rest_epochs.shape if rest_epochs is not None else None}")

    # Separate personal and benchmark data
    if source is not None:
        personal_mask = np.array([s == "personal" for s in source])
        benchmark_mask = ~personal_mask
        X_personal = X[personal_mask]
        y_personal = y[personal_mask]
        X_benchmark = X[benchmark_mask]
        y_benchmark = y[benchmark_mask]
        logger.info(f"Personal: {len(X_personal)}, Benchmark: {len(X_benchmark)}")
    else:
        raise ValueError("Dataset must have 'source' field to distinguish personal vs benchmark data")

    # Augment personal data
    logger.info(f"Augmenting personal data with factor {args.augmentation_factor}x")
    X_personal_aug = augment_compose(X_personal, sr=250, num_augmentations=args.augmentation_factor)
    y_personal_aug = np.repeat(y_personal, args.augmentation_factor)
    logger.info(f"Augmented personal shape: {X_personal_aug.shape}")

    # Recalculate target benchmark count based on new augmented personal size
    target_benchmark_new = int(len(X_personal_aug) * (1.0 - args.personal_weight) / max(args.personal_weight, 1e-6))
    target_benchmark_new = min(target_benchmark_new, len(X_benchmark))
    indices = np.random.choice(len(X_benchmark), size=target_benchmark_new, replace=False)
    X_benchmark_sampled = X_benchmark[indices]
    y_benchmark_sampled = y_benchmark[indices]
    logger.info(f"Sampled benchmark: {len(X_benchmark_sampled)}")

    # Combine augmented personal + resampled benchmark
    X_combined = np.concatenate([X_personal_aug, X_benchmark_sampled], axis=0)
    y_combined = np.concatenate([y_personal_aug, y_benchmark_sampled], axis=0)

    # Compute actual personal weight
    actual_personal_weight = len(X_personal_aug) / len(X_combined)
    logger.info(f"Combined dataset: {X_combined.shape}, actual personal weight: {actual_personal_weight:.2%}")

    # Train/test split
    n_test = max(1, int(len(X_combined) * args.test_split))
    test_indices = np.random.choice(len(X_combined), size=n_test, replace=False)
    train_mask = np.ones(len(X_combined), dtype=bool)
    train_mask[test_indices] = False

    X_train_full = X_combined[train_mask]
    y_train_full = y_combined[train_mask]
    X_test = X_combined[test_indices]
    y_test = y_combined[test_indices]

    # Fit global normalizer on train split only (no leakage)
    logger.info("Fitting global channel-wise normalizer on train split")
    scaler = GlobalChannelZScore()
    X_train_full_input = to_model_input(X_train_full)
    scaler.fit(X_train_full_input)

    # Train/val split
    n_val = max(1, int(len(X_train_full) * args.val_split))
    val_indices = np.random.choice(len(X_train_full), size=n_val, replace=False)
    train_mask_inner = np.ones(len(X_train_full), dtype=bool)
    train_mask_inner[val_indices] = False

    X_train = X_train_full[train_mask_inner]
    y_train = y_train_full[train_mask_inner]
    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]

    # Normalize all splits
    X_train_input = to_model_input(X_train)
    X_train_input = scaler.transform(X_train_input)
    X_val_input = to_model_input(X_val)
    X_val_input = scaler.transform(X_val_input)
    X_test_input = to_model_input(X_test)
    X_test_input = scaler.transform(X_test_input)

    logger.info(f"Train: {X_train_input.shape}, Val: {X_val_input.shape}, Test: {X_test_input.shape}")

    # Load pretrained model
    logger.info(f"Loading pretrained model: {args.base_model}")
    model = keras.models.load_model(args.base_model, safe_mode=False)
    
    # Apply freezing strategy
    apply_freezing_strategy(model, args.strategy, logger)

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    save_dir = Path(__file__).parent / "models" / "personalized"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / f"{args.user_id}_combined_augmented.keras"
    checkpoint = ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True, verbose=0)
    early_stop = EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # Train
    logger.info(f"Training for up to {args.epochs} epochs with early stopping patience={args.patience}")
    history = model.fit(
        X_train_input, y_train,
        validation_data=(X_val_input, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_loss, test_acc = model.evaluate(X_test_input, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test_input, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Weighted F1: {f1_weighted:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Get best val accuracy from history
    best_val_acc = max(history.history["val_accuracy"])
    best_epoch = np.argmax(history.history["val_accuracy"]) + 1
    logger.info(f"Best Val Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

    # Check if target achieved
    if best_val_acc >= 0.76:
        logger.info(f"✓ TARGET ACHIEVED: Val accuracy {best_val_acc:.4f} >= 76%")
    else:
        logger.warning(f"✗ Target not met: Val accuracy {best_val_acc:.4f} < 76% (consider more augmentation or data)")

    # Calibrate neutral threshold
    if rest_epochs is not None and len(rest_epochs) > 0:
        logger.info("Calibrating neutral state threshold")
        threshold_info = calibrate_threshold(model, scaler, rest_epochs)
        threshold_path = save_dir / f"{args.user_id}_combined_augmented_neutral_threshold.json"
        with open(threshold_path, "w") as f:
            json.dump(threshold_info, f, indent=2)
        logger.info(f"Saved threshold: {threshold_path}")
        logger.info(f"  Neutral confidence threshold: {threshold_info['threshold']:.4f}")

    # Save scaler
    scaler_path = save_dir / f"{args.user_id}_combined_augmented_scaler.json"
    scaler.save(str(scaler_path))
    logger.info(f"Saved scaler: {scaler_path}")

    # Save metrics
    metrics = {
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "test_accuracy": float(test_acc),
        "weighted_f1": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "personal_weight": actual_personal_weight,
        "augmentation_factor": args.augmentation_factor,
        "strategy": args.strategy,
        "dataset_shape": X_combined.shape,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    metrics_path = save_dir / f"{args.user_id}_combined_augmented_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Model:     {model_path}")
    logger.info(f"Scaler:    {scaler_path}")
    logger.info(f"Threshold: {threshold_path if rest_epochs is not None else 'N/A'}")
    logger.info(f"Metrics:   {metrics_path}")
    logger.info("="*70 + "\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
