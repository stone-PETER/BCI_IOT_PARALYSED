import numpy as np
import tensorflow as tf

def validate_model():
    # File paths based on your metadata
    model_path = r"C:\programs\BCI_IOT_PARALYSED\CODE\models\personalized\alan_finetuned_best.keras"
    
    data_path = r"C:\programs\BCI_IOT_PARALYSED\CODE\calibration_data_preprocessed\alan_calibration_20260318_105119_preprocessed.npz"

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    
    # Extract features and labels
    X = data['X']
    y = data['y']
    
    # Safely map any 2 distinct classes to 0 and 1
    unique_labels = np.unique(y)
    print(f"Original unique labels found: {unique_labels}")
    
    if len(unique_labels) != 2:
        print(f"Warning: Expected 2 classes, but found {len(unique_labels)}: {unique_labels}")
    
    # Map the lowest value to 0, and the next to 1
    y_mapped = np.zeros_like(y)
    y_mapped[y == unique_labels[0]] = 0
    if len(unique_labels) > 1:
        y_mapped[y == unique_labels[1]] = 1
    
    y = y_mapped
    print(f"Mapped labels to: {np.unique(y)}")

    # Add channel dimension for 2D convolutions (EEGNet expects 4D: samples, channels, time, 1)
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)
        
    # One-hot encode the labels to match model output (categorical_crossentropy)
    y = tf.keras.utils.to_categorical(y, num_classes=2)
    
    print(f"Adjusted data shape: X={X.shape}, y={y.shape}")

    # Evaluate the model
    loss, accuracy = model.evaluate(X, y, verbose=1)
    
    print(f"\nValidation Results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    validate_model()