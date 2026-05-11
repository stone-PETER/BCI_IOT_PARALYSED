import os
import sys
import argparse
import numpy as np
import mne
import joblib
import json
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score

# Add parent directory to path to allow importing from CODE
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def to_trials_channels_samples(epochs: np.ndarray) -> np.ndarray:
    if epochs.ndim != 3:
        raise ValueError(f"Expected 3D epochs, got {epochs.shape}")
    if epochs.shape[1] == 3:
        return epochs
    if epochs.shape[2] == 3:
        return epochs.transpose(0, 2, 1)
    raise ValueError(f"Could not infer channel axis from shape {epochs.shape}") 

def load_3class_data(path: str):
    print(f"Loading {path}...")
    pack = np.load(path, allow_pickle=True)

    if all(key in pack for key in ("epochs_left", "epochs_right")):
        epochs_left = to_trials_channels_samples(np.asarray(pack["epochs_left"], dtype=np.float64))
        epochs_right = to_trials_channels_samples(np.asarray(pack["epochs_right"], dtype=np.float64))
        
        if "epochs_rest" in pack and len(pack["epochs_rest"]) > 0:
            epochs_rest = to_trials_channels_samples(np.asarray(pack["epochs_rest"], dtype=np.float64))
        else:
            print("Warning: 'epochs_rest' not found or empty! Cannot perform 3-class training without 'Rest' data.")
            epochs_rest = np.zeros((0, epochs_left.shape[1], epochs_left.shape[2]), dtype=np.float64)
            
        print(f"Original sets - Left: {epochs_left.shape}, Right: {epochs_right.shape}, Rest: {epochs_rest.shape}")

        X = np.concatenate([epochs_left, epochs_right, epochs_rest], axis=0)
        y = np.concatenate([
            np.zeros(len(epochs_left), dtype=np.int32),   # 0 = Left
            np.ones(len(epochs_right), dtype=np.int32),   # 1 = Right
            np.full(len(epochs_rest), 2, dtype=np.int32), # 2 = Rest
        ])
        return X, y
    else:
        x_key = [k for k in ["X", "data", "epochs"] if k in pack][0]
        y_key = [k for k in ["y", "labels"] if k in pack][0]
        X = to_trials_channels_samples(np.asarray(pack[x_key], dtype=np.float64))
        y = np.asarray(pack[y_key], dtype=np.int32)
        print(f"Loaded generic dataset: {X.shape}")
        return X, y

class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    Filter Bank Common Spatial Pattern
    Applies multiple bandpass filters, crops the time window, calculates CSP for each, and concatenates the log-variance features.
    """
    def __init__(self, sfreq=250, bands=None, n_components=4, tmin=0.5, tmax=3.5):
        self.sfreq = sfreq
        self.tmin = tmin
        self.tmax = tmax
        # Default Filter Bank: 4-8, 8-12, 12-16, 16-20, 20-24, 24-28, 28-32, 32-36, 36-40 Hz
        if bands is None:
            self.bands = [[4 + i*4, 8 + i*4] for i in range(9)]
        else:
            self.bands = bands
        self.n_components = n_components
        # Using MNE's CSP which natively supports multiclass via one-vs-rest
        self.csps = [mne.decoding.CSP(n_components=self.n_components, log=True, norm_trace=False) for _ in self.bands]
        
    def fit(self, X, y):
        # Suppress MNE info messages during CV
        mne.set_log_level('ERROR')
        start_idx = int(self.tmin * self.sfreq) if self.tmin is not None else 0
        end_idx = int(self.tmax * self.sfreq) if self.tmax is not None else X.shape[2]
        
        for i, band in enumerate(self.bands):
            X_filtered = mne.filter.filter_data(X, self.sfreq, l_freq=band[0], h_freq=band[1], verbose='ERROR')
            X_cropped = X_filtered[:, :, start_idx:end_idx]
            self.csps[i].fit(X_cropped, y)
        return self
        
    def transform(self, X):
        mne.set_log_level('ERROR')
        start_idx = int(self.tmin * self.sfreq) if self.tmin is not None else 0
        end_idx = int(self.tmax * self.sfreq) if self.tmax is not None else X.shape[2]
        
        features = []
        for i, band in enumerate(self.bands):
            X_filtered = mne.filter.filter_data(X, self.sfreq, l_freq=band[0], h_freq=band[1], verbose='ERROR')
            X_cropped = X_filtered[:, :, start_idx:end_idx]
            features.append(self.csps[i].transform(X_cropped))
        return np.concatenate(features, axis=1)

def main():
    svm_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = os.path.join(svm_dir, f"fbcsp_svm_{timestamp_str}.joblib")
    registry_path = os.path.join(os.path.dirname(svm_dir), "model_registry.json")

    parser = argparse.ArgumentParser(description="FBCSP + SVM 3-class BCI Training")
    parser.add_argument("--data", type=str, required=True, help="Path to personal .npz data")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--k-best", type=int, default=10, help="Number of features to select")
    parser.add_argument("--output", type=str, default=default_output, help="Output model path")
    parser.add_argument("--user", type=str, default="default_user", help="User name for registry")
    args = parser.parse_args()

    X, y = load_3class_data(args.data)
    
    # Define pipeline: Filter Bank CSP -> Feature Selection -> SVM
    # SVM must have probability=True for API endpoints that perform thresholding
    pipeline = Pipeline([
        ('fbcsp', FilterBankCSP(sfreq=250, n_components=4, tmin=0.5, tmax=3.5)),
        ('feat_select', SelectKBest(score_func=mutual_info_classif)),
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
    ])

    # Define hyperparameter grid
    param_grid = {
        'feat_select__k': [10, 18, 27],  # Changed max from 36 to 27
        'svm__C': [1, 10, 100, 1000],  # Increased C pushes the SVM to fit harder
        'svm__gamma': ['scale', 'auto', 0.1, 1]
    }

    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    print("\n--- Starting Grid Search CV for FBCSP + SVM ---")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=kfold, 
        scoring='accuracy', 
        n_jobs=-1,  # Uses all available CPU cores to speed it up drastically
        verbose=3   # Prints progress of each fit
    )
    
    grid_search.fit(X, y)
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

    print("\nComputing Aggregated Classification Report using Best Estimator...")
    y_pred = cross_val_predict(grid_search.best_estimator_, X, y, cv=kfold)
    print(classification_report(y, y_pred, target_names=["Left", "Right", "Rest"]))
    
    # Save the model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(grid_search.best_estimator_, args.output)
    print(f"Model successfully trained & saved to: {args.output}")

    # Update Registry
    try:
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                registry = json.load(f)
        else:
            registry = {"personalized_models": {}}

        if "personalized_models" not in registry:
            registry["personalized_models"] = {}

        model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_key = f"{args.user}_fbcsp_svm_{model_timestamp}"
        registry["personalized_models"][model_key] = {
            "model_path": os.path.abspath(args.output),
            "model_type": "FBCSP_SVM",
            "threshold": 0.5,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "val_accuracy": float(grid_search.best_score_),
                "k_components": 4,
                "k_best_features": grid_search.best_estimator_.named_steps['feat_select'].k,
                "svm_C": grid_search.best_params_["svm__C"],
                "svm_gamma": grid_search.best_params_["svm__gamma"],
                "calibration_file": os.path.basename(args.data)
            }
        }

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"Model registry updated successfully at {registry_path}")

    except Exception as e:
        print(f"Failed to update registry: {e}")

if __name__ == "__main__":
    main()
