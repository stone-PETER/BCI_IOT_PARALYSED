import json
from pathlib import Path
import numpy as np


class GlobalChannelZScore:
    """Global train-set channel-wise z-score normalizer."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        if X.ndim != 4:
            raise ValueError(f"Expected 4D input (N, C, T, 1), got {X.shape}")

        x = X[..., 0]
        self.mean_ = x.mean(axis=(0, 2), keepdims=True)
        self.std_ = x.std(axis=(0, 2), keepdims=True)
        self.std_ = np.maximum(self.std_, self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Normalizer must be fitted before transform")
        if X.ndim != 4:
            raise ValueError(f"Expected 4D input (N, C, T, 1), got {X.shape}")

        x = X[..., 0]
        x = (x - self.mean_) / self.std_
        return x[..., np.newaxis]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, file_path: str):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Cannot save unfitted normalizer")

        payload = {
            "eps": float(self.eps),
            "mean": self.mean_.reshape(-1).tolist(),
            "std": self.std_.reshape(-1).tolist()
        }

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r') as f:
            payload = json.load(f)

        normalizer = cls(eps=float(payload.get("eps", 1e-8)))
        mean = np.array(payload["mean"], dtype=np.float32)
        std = np.array(payload["std"], dtype=np.float32)
        normalizer.mean_ = mean.reshape(1, -1, 1)
        normalizer.std_ = std.reshape(1, -1, 1)
        return normalizer
