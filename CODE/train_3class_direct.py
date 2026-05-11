"""
Direct Training over Personal NPG Lite Data (3-Class)

- Loads the 299 calibration trials (Left, Right, Rest).
- Applies Window Cropping augmentation to artificially expand dataset.
- Performs 5-Fold Cross Validation.
- Employs EEGNet with native 3-class target layer.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import class_weight

from eegnet_model import EEGNet
from eeg_augmentation import EEGAugmentationPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Direct 3-class EEGNet Training")
    parser.add_argument("--data", type=str, required=True, help="Path to personal .npz data")
    parser.add_argument("--config", type=str, default="config_3class.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for CV")
    return parser.parse_args()

def to_trials_channels_samples(epochs: np.ndarray) -> np.ndarray:
    if epochs.ndim != 3:
        raise ValueError(f"Expected 3D epochs, got {epochs.shape}")

    # already [trials, channels, samples]
    if epochs.shape[1] == 3:
        return epochs

    # [trials, samples, channels] -> [trials, channels, samples]
    if epochs.shape[2] == 3:
        return epochs.transpose(0, 2, 1)

    raise ValueError(f"Could not infer channel axis from shape {epochs.shape}") 

def load_3class_data(path: str):
    print(f"Loading {path}...")
    pack = np.load(path, allow_pickle=True)

    if all(key in pack for key in ("epochs_left", "epochs_right")):
        epochs_left = to_trials_channels_samples(np.asarray(pack["epochs_left"], dtype=np.float32))
        epochs_right = to_trials_channels_samples(np.asarray(pack["epochs_right"], dtype=np.float32))
        
        # Check for rest
        if "epochs_rest" in pack and len(pack["epochs_rest"]) > 0:
            epochs_rest = to_trials_channels_samples(np.asarray(pack["epochs_rest"], dtype=np.float32))
        else:
            print("Warning: 'epochs_rest' not found or empty! Cannot perform 3-class training without 'Rest' data.")
            epochs_rest = np.zeros((0, epochs_left.shape[1], epochs_left.shape[2]))
            
        print(f"Original sets - Left: {epochs_left.shape}, Right: {epochs_right.shape}, Rest: {epochs_rest.shape}")

        X = np.concatenate([epochs_left, epochs_right, epochs_rest], axis=0)
        y = np.concatenate([
            np.zeros(len(epochs_left), dtype=np.int32),   # 0 = Left
            np.ones(len(epochs_right), dtype=np.int32),   # 1 = Right
            np.full(len(epochs_rest), 2, dtype=np.int32), # 2 = Rest
        ])
        return X, y
    else:
        # Generic fallback
        x_key = [k for k in ["X", "data", "epochs"] if k in pack][0]
        y_key = [k for k in ["y", "labels"] if k in pack][0]
        X = to_trials_channels_samples(np.asarray(pack[x_key], dtype=np.float32))
        y = np.asarray(pack[y_key], dtype=np.int32)
        print(f"Loaded generic dataset: {X.shape}")
        return X, y

def main():
    args = parse_args()

    # Load initial unaugmented data
    X, y = load_3class_data(args.data)
    
    # Needs channel-last for keras (trials, samples, channels) 
    # Current EEGNet expects input shape: (samples, channels, 1) or (channels, samples, 1) depending on config.
    # We will let EEGNet handle it if we look at eegnet_model.py, it expects:
    # (trials, samples, channels, 1) usually if channel_last is True? Wait, usually EEGNet expects (trials, chans, samples, 1)
    # Actually, config says `chans: 3, samples: 500`. Let's just crop FIRST on (trials, chans, samples).
    
    augmenter = EEGAugmentationPipeline()
    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_accuracies = []

    print("\n--- Starting 5-Fold Cross Validation ---")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n--- Fold {fold+1}/{args.folds} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # 1. Window Cropping on Training Set Only (from 1000 to 500 samples)
        # Expected input to window_cropping: (trials, samples, channels)? Wait, `window_cropping` defined earlier
        # assumed timepoints is dim 1: `epochs, timepoints, channels = data.shape`. 
        # But `to_trials_channels_samples` returns (trials, channels, timepoints). We need to transpose to (trials, timepoints, channels) for window_cropping?
        # Let's check window_cropping definition: 
        # `def window_cropping(self, data, labels=None, window_size=500, overlap=0.5): epochs, timepoints, channels = data.shape`
        X_train_trans = X_train.transpose(0, 2, 1) # -> (trials, timepoints, channels)
        X_train_crop, y_train_crop = augmenter.window_cropping(X_train_trans, list(y_train), window_size=500, overlap=0.5)
        X_train_final = X_train_crop.transpose(0, 2, 1) # Back to (trials, channels, timepoints) for EEGNet ?
        
        # 2. Window Cropping on Validation Set (No overlap to avoid evaluating heavily correlated samples)
        X_val_trans = X_val.transpose(0, 2, 1)
        X_val_crop, y_val_crop = augmenter.window_cropping(X_val_trans, list(y_val), window_size=500, overlap=0)
        X_val_final = X_val_crop.transpose(0, 2, 1)
        
        # Ensure correct formatting mapping into TF inputs
        # Models in keras often need: (trials, channels, timepoints, 1) for EEGNet
        X_train_final = np.expand_dims(X_train_final, axis=-1)
        X_val_final = np.expand_dims(X_val_final, axis=-1)
        
        y_train_cat = to_categorical(y_train_crop, num_classes=3)
        y_val_cat = to_categorical(y_val_crop, num_classes=3)
        
        # Determine true class labels (undoing one-hot encoding if present)
        y_train_labels = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train
        
        # Calculate balanced weights
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_labels),
            y=y_train_labels
        )
        weight_dict = {i: w for i, w in enumerate(weights)}
        
        print(f"Applying class weights: {weight_dict}")

        # Instantiate Model
        model_obj = EEGNet(config_path=args.config)
        model = model_obj.build_model()
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
            metrics=['accuracy']
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val_final, y_val_cat),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=weight_dict,     # <-- ADD THIS LINE
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate
        preds = model.predict(X_val_final)
        pred_classes = np.argmax(preds, axis=1)
        
        acc = accuracy_score(y_val_crop, pred_classes)
        fold_accuracies.append(acc)
        
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")
        print(classification_report(y_val_crop, pred_classes, target_names=["Left", "Right", "Rest"]))

    mean_acc = np.mean(fold_accuracies)
    print(f"\n--- Cross Validation Complete ---")
    print(f"Mean Accuracy over {args.folds} Folds: {mean_acc:.4f}")

if __name__ == "__main__":
    main()
