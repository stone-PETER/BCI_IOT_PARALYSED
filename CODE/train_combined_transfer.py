#!/usr/bin/env python3
"""
Train EEGNet on combined benchmark + personal dataset using transfer learning.
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

from normalization_utils import GlobalChannelZScore


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


def apply_freezing_strategy(model: keras.Model, strategy: str, logger: logging.Logger) -> int:
    logger.info(f"Applying freezing strategy: {strategy}")

    if strategy == "head-only":
        for layer in model.layers[:-2]:
            layer.trainable = False
        logger.info("Frozen Block1+Block2; training Dense head")
    elif strategy == "last-block":
        for layer in model.layers[:7]:
            layer.trainable = False
        logger.info("Frozen Block1; training Block2 + Dense head")
    elif strategy == "full":
        for layer in model.layers:
            layer.trainable = True
        logger.info("Training full model")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    trainable_params = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    total_params = int(model.count_params())
    logger.info(f"Trainable params: {trainable_params:,}/{total_params:,} ({100.0*trainable_params/total_params:.1f}%)")
    return trainable_params


def main():
    parser = argparse.ArgumentParser(description="Train combined dataset with transfer learning")
    parser.add_argument("--dataset-file", type=str, required=True, help="Path to combined dataset .npz")
    parser.add_argument("--user-id", type=str, required=True, help="User id for output naming")
    parser.add_argument("--base-model", type=str, default="models/best/eegnet_2class_bci2b.keras", help="Pretrained EEGNet model")
    parser.add_argument("--strategy", type=str, choices=["head-only", "last-block", "full"], default="last-block")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    root = Path(__file__).parent

    dataset_path = Path(args.dataset_file)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    base_model_path = Path(args.base_model)
    if not base_model_path.is_absolute():
        base_model_path = root / base_model_path
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    logger.info(f"Loading combined dataset: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    rest_epochs = data["rest_epochs"] if "rest_epochs" in data else None

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

    logger.info(f"Loading base model: {base_model_path}")
    model = keras.models.load_model(base_model_path)
    trainable_params = apply_freezing_strategy(model, args.strategy, logger)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    save_dir = root / "models" / "personalized"
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = save_dir / f"{args.user_id}_combined_transfer_best_{ts}.keras"

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(best_path), monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(4, args.patience // 3), min_lr=1e-6, verbose=1)
    ]

    logger.info(f"Transfer training: strategy={args.strategy} train={len(X_train)} val={len(X_val)} test={len(X_test)}")
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
    report = classification_report(y_test, y_pred, target_names=['LEFT', 'RIGHT'], output_dict=True, zero_division=0)

    model_path = save_dir / f"{args.user_id}_combined_transfer.keras"
    model.save(model_path)

    scaler_path = save_dir / f"{args.user_id}_combined_transfer_scaler.json"
    scaler.save(str(scaler_path))

    threshold_info = None
    threshold_path = None
    if rest_epochs is not None and len(rest_epochs) > 0:
        threshold_info = calibrate_threshold(model, scaler, rest_epochs)
        threshold_path = save_dir / f"{args.user_id}_combined_transfer_neutral_threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump(threshold_info, f, indent=2)

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "dataset_file": str(dataset_path),
        "user_id": args.user_id,
        "base_model": str(base_model_path),
        "strategy": args.strategy,
        "trainable_parameters": trainable_params,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "epochs_ran": int(len(history.history['loss'])),
        "best_val_accuracy": float(np.max(history.history['val_accuracy'])),
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

    metrics_path = save_dir / f"{args.user_id}_combined_transfer_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info("Transfer training complete")
    logger.info(f"Best val accuracy: {np.max(history.history['val_accuracy']):.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Scaler: {scaler_path}")
    if threshold_path:
        logger.info(f"Threshold: {threshold_path}")
    logger.info(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
