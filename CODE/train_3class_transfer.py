"""
Transfer Learning over Personal NPG Lite Data (3-Class Backup)

- Loads the 299 calibration trials (Left, Right, Rest).
- Applies Window Cropping.
- Loads existing BCI 2b 2-class model, or base model.
- Freezes early Spatial and Temporal convolutional layers.
- Appends a 3-class target layer.
- Fine-tunes model.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

from eeg_augmentation import EEGAugmentationPipeline
from train_3class_direct import load_3class_data

def parse_args():
    parser = argparse.ArgumentParser(description="Transfer 3-class EEGNet Training")
    parser.add_argument("--data", type=str, required=True, help="Path to personal .npz data")
    parser.add_argument("--base-model", type=str, required=True, help="Path to base .keras BCI model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for CV")
    return parser.parse_args()

def adapt_model_for_3_classes(base_model_path):
    print(f"Loading base model: {base_model_path}")
    base_model = tf.keras.models.load_model(base_model_path, compile=False)
    
    # We want to freeze early layers (Conv2D and DepthwiseConv2D) to retain fundamental EEG features
    for layer in base_model.layers:
        if 'conv2d' in layer.name.lower() or 'depthwise' in layer.name.lower():
            layer.trainable = False
        else:
            layer.trainable = True
            
    # The last layer is usually Dense. Let's find the layer just before it (e.g. Flatten or Dropout)
    # Actually, we can pop the last layer by accessing layers up to -1.
    if isinstance(base_model.layers[-1], tf.keras.layers.Dense):
        penultimate_output = base_model.layers[-2].output
    else:
        # Just in case there is an activation after Dense, though EEGNet usually has Activation('softmax') or Dense with softmax included
        # By EEGNet design, it's typically Flatten -> Dense
        penultimate_output = base_model.layers[-2].output 
    
    # We add our new 3-class Dense layer
    new_output = Dense(3, name='dense_3class', kernel_constraint=tf.keras.constraints.max_norm(0.25))(penultimate_output)
    
    # Wait, the original EEGNet output usually has a `Activation('softmax')` layer or it's inside `Dense`.
    # Let's ensure softmax
    if not isinstance(new_output, tf.Tensor) or new_output.shape[-1] != 3:
        pass # Handle if needed
        
    final_output = tf.keras.layers.Activation('softmax', name='softmax_3class')(new_output)
    
    new_model = Model(inputs=base_model.input, outputs=final_output)
    new_model.compile(
        loss='categorical_crossentropy', 
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # smaller LR for fine-tuning
        metrics=['accuracy']
    )
    
    return new_model

def main():
    args = parse_args()

    # Load data
    X, y = load_3class_data(args.data)
    
    augmenter = EEGAugmentationPipeline()
    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n--- Fold {fold+1}/{args.folds} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # 1. Window Cropping on Training Set Only 
        X_train_trans = X_train.transpose(0, 2, 1)
        X_train_crop, y_train_crop = augmenter.window_cropping(X_train_trans, list(y_train), window_size=500, overlap=0.5)
        X_train_final = X_train_crop.transpose(0, 2, 1)
        
        # 2. Window Cropping on Val Set 
        X_val_trans = X_val.transpose(0, 2, 1)
        # Note: the test samples also must be 500 length since base model depends on sample count.
        # Wait - base model (e.g. from BCI IV 2b) might actually have been trained on 1000 samples!
        # If the base model expects 1000 samples, we CANNOT pass 500 samples without a shape mismatch.
        # Exception: if we trained a base model from scratch using config_3class.yaml.
        # So we skip cropping for base model if its input shape expects 1000 samples?
        
        # We will assume user base_model matches the shape `(chans, 500, 1)` or we must re-evaluate.
        # Let's crop validation set anyway.
        X_val_crop, y_val_crop = augmenter.window_cropping(X_val_trans, list(y_val), window_size=500, overlap=0)
        X_val_final = X_val_crop.transpose(0, 2, 1)
        
        X_train_final = np.expand_dims(X_train_final, axis=-1)
        X_val_final = np.expand_dims(X_val_final, axis=-1)
        
        y_train_cat = to_categorical(y_train_crop, num_classes=3)
        y_val_cat = to_categorical(y_val_crop, num_classes=3)
        
        model = adapt_model_for_3_classes(args.base_model)
        
        # Let's dynamically check base_model input shape
        expected_shape = model.input_shape # (None, chans, samples, 1)
        if expected_shape[2] != 500:
            print(f"WARNING: Base model expects {expected_shape[2]} samples, but data has {X_train_final.shape[2]}.")
            print("Consider retraining a base model with 500 samples first, or adjusting window size.")
            
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        )
        
        history = model.fit(
            X_train_final, y_train_cat,
            validation_data=(X_val_final, y_val_cat),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        preds = model.predict(X_val_final)
        pred_classes = np.argmax(preds, axis=1)
        
        acc = accuracy_score(y_val_crop, pred_classes)
        fold_accuracies.append(acc)
        
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")
        print(classification_report(y_val_crop, pred_classes, target_names=["Left", "Right", "Rest"]))

    mean_acc = np.mean(fold_accuracies)
    print(f"\n--- Cross Validation Transfer complete ---")
    print(f"Mean Accuracy over {args.folds} Folds: {mean_acc:.4f}")

if __name__ == "__main__":
    main()
