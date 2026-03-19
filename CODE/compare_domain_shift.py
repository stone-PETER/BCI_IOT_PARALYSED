"""
Compare domain shift between BCI Competition IV 2b and personal NPG Lite data.

- BCI IV 2b is loaded from .mat files using BCI4_2B_Loader_v2.
- Personal data is loaded from .npz (generic X/y or calibration format).

Supported personal .npz keys:
- Generic: X (or data/epochs), y (or labels), fs (or sampling_rate)
- Calibration: epochs_left, epochs_right, optional epochs_rest

Data shape must be 3D and include C3/Cz/C4 (or first 3 channels fallback).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import welch, resample

from bci4_2b_loader_v2 import BCI4_2B_Loader


BANDS = {
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "motor": (8.0, 30.0),
    "line50": (49.0, 51.0),
}


def _first_existing_key(payload: Dict, candidates) -> Optional[str]:
    for key in candidates:
        if key in payload:
            return key
    return None


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


def load_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float], list]:
    pack = np.load(path, allow_pickle=True)

    x_key = _first_existing_key(pack, ["X", "data", "epochs"])
    if x_key is None:
        raise ValueError(f"{path} does not contain data key (X/data/epochs)")

    X = np.asarray(pack[x_key], dtype=np.float32)
    X = to_trials_channels_samples(X)

    y_key = _first_existing_key(pack, ["y", "labels"])
    y = np.asarray(pack[y_key], dtype=np.int32) if y_key else None

    fs_key = _first_existing_key(pack, ["fs", "sampling_rate"])
    fs = float(np.asarray(pack[fs_key]).reshape(-1)[0]) if fs_key else None

    ch_key = _first_existing_key(pack, ["ch_names", "channels", "channel_names"])
    ch_names = list(pack[ch_key]) if ch_key else [f"ch{i}" for i in range(X.shape[1])]

    return X, y, fs, ch_names


def load_personal_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float], list]:
    """
    Supports two personal formats:
    1) generic npz with X/y
    2) calibration npz with epochs_left/epochs_right/epochs_rest
    """
    pack = np.load(path, allow_pickle=True)

    if all(key in pack for key in ("epochs_left", "epochs_right")):
        epochs_left = to_trials_channels_samples(np.asarray(pack["epochs_left"], dtype=np.float32))
        epochs_right = to_trials_channels_samples(np.asarray(pack["epochs_right"], dtype=np.float32))

        X = np.concatenate([epochs_left, epochs_right], axis=0)
        y = np.concatenate([
            np.zeros(len(epochs_left), dtype=np.int32),
            np.ones(len(epochs_right), dtype=np.int32),
        ])

        fs_key = _first_existing_key(pack, ["fs", "sampling_rate"])
        fs = float(np.asarray(pack[fs_key]).reshape(-1)[0]) if fs_key else 250.0

        ch_key = _first_existing_key(pack, ["ch_names", "channels", "channel_names"])
        ch_names = list(pack[ch_key]) if ch_key else ["C3", "Cz", "C4"]

        return X, y, fs, ch_names

    return load_npz(path)


def select_c3_cz_c4(X: np.ndarray, ch_names: list) -> Tuple[np.ndarray, list]:
    idx_map = {str(name).upper(): idx for idx, name in enumerate(ch_names)}
    needed = ["C3", "CZ", "C4"]

    if all(name in idx_map for name in needed):
        idx = [idx_map["C3"], idx_map["CZ"], idx_map["C4"]]
        return X[:, idx, :], ["C3", "Cz", "C4"]

    if X.shape[1] < 3:
        raise ValueError("Need at least 3 channels to compare C3/Cz/C4 domain shift")

    return X[:, :3, :], ["C3", "Cz", "C4"]


def dc_center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=2, keepdims=True)


def maybe_resample(X: np.ndarray, fs_in: Optional[float], fs_out: float) -> np.ndarray:
    if fs_in is None or abs(fs_in - fs_out) < 1e-6:
        return X

    target_samples = int(round(X.shape[2] * fs_out / fs_in))
    return resample(X, target_samples, axis=2)


def integrate_band(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return np.zeros(psd.shape[:-1], dtype=np.float32)
    return np.trapz(psd[..., mask], freqs[mask], axis=-1)


def extract_features(X: np.ndarray, fs: float) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    freqs, pxx = welch(X, fs=fs, nperseg=min(256, X.shape[2]), axis=2)

    bandpowers = {}
    for name, (lo, hi) in BANDS.items():
        bandpowers[name] = integrate_band(pxx, freqs, lo, hi)

    mu = np.log(bandpowers["mu"] + 1e-12)
    beta = np.log(bandpowers["beta"] + 1e-12)
    motor = np.log(bandpowers["motor"] + 1e-12)

    feature_vector = np.stack(
        [
            mu[:, 0], mu[:, 2],
            beta[:, 0], beta[:, 2],
            motor[:, 0], motor[:, 2],
        ],
        axis=1,
    )
    return bandpowers, feature_vector


def class_separation(feature_vector: np.ndarray, y: Optional[np.ndarray]) -> Optional[dict]:
    if y is None:
        return None

    classes = np.unique(y)
    if classes.size != 2:
        return None

    a = feature_vector[y == classes[0]]
    b = feature_vector[y == classes[1]]
    if len(a) < 2 or len(b) < 2:
        return None

    mean_diff = np.abs(a.mean(axis=0) - b.mean(axis=0))
    pooled_std = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2.0 + 1e-12)
    effect = mean_diff / pooled_std

    return {
        "classes": [int(classes[0]), int(classes[1])],
        "effect_size_per_feature": effect.tolist(),
        "mean_effect_size": float(effect.mean()),
    }


def frechet_diag(features_a: np.ndarray, features_b: np.ndarray) -> float:
    mean_a = features_a.mean(axis=0)
    mean_b = features_b.mean(axis=0)
    cov_a = np.cov(features_a, rowvar=False)
    cov_b = np.cov(features_b, rowvar=False)

    diag_a = np.diag(cov_a)
    diag_b = np.diag(cov_b)

    mean_term = np.sum((mean_a - mean_b) ** 2)
    cov_term = np.sum(diag_a + diag_b - 2.0 * np.sqrt(diag_a * diag_b + 1e-12))
    return float(mean_term + cov_term)


def summarize(
    name: str,
    X: np.ndarray,
    y: Optional[np.ndarray],
    fs: float,
    bandpowers: Dict[str, np.ndarray],
    feature_vector: np.ndarray,
) -> dict:
    line_ratio = (bandpowers["line50"] / (bandpowers["motor"] + 1e-12)).mean()

    return {
        "name": name,
        "shape": list(X.shape),
        "sampling_rate": fs,
        "channel_std_mean": X.std(axis=2).mean(axis=0).tolist(),
        "mu_power_mean": bandpowers["mu"].mean(axis=0).tolist(),
        "beta_power_mean": bandpowers["beta"].mean(axis=0).tolist(),
        "line50_to_motor_ratio": float(line_ratio),
        "class_separation": class_separation(feature_vector, y),
    }


def tune_recommendations(bci: dict, personal: dict, fd: float, target_fs: float) -> list:
    notes = []

    if abs(float(personal["sampling_rate"]) - target_fs) > 1e-6:
        notes.append("Resample personal stream to 250 Hz before inference.")

    if personal["line50_to_motor_ratio"] > 1.5 * bci["line50_to_motor_ratio"]:
        notes.append("Powerline noise is higher in personal data; keep notch enabled and improve grounding/contact.")

    bci_std = float(np.mean(bci["channel_std_mean"]))
    per_std = float(np.mean(personal["channel_std_mean"]))
    if per_std > 2.0 * bci_std or per_std < 0.5 * bci_std:
        notes.append("Amplitude scale mismatch detected; test z-score normalization ON for realtime.")

    bci_sep = bci.get("class_separation")
    per_sep = personal.get("class_separation")
    if bci_sep and per_sep and per_sep["mean_effect_size"] < 0.7 * bci_sep["mean_effect_size"]:
        notes.append("Personal class separability is weaker; increase cue quality and collect more personal training epochs.")

    if fd > 1.0:
        notes.append("Domain shift is high; fine-tune final dense layer on personal labeled data.")

    notes.append("Evaluate both --laplacian and --no-laplacian using the same session and keep whichever gives better class separation.")
    notes.append("Use class-gap decision with a safer margin (start 0.05, try 0.08 if toggling remains unstable).")

    return notes


def main():
    parser = argparse.ArgumentParser(description="Compare BCI 4 2b (.mat via loader) vs personal NPG (.npz) domain shift")
    parser.add_argument("--personal", required=True, help="Path to personal .npz (generic or calibration format)")
    parser.add_argument("--config", default="config_2b.yaml", help="Config path for BCI4_2B_Loader_v2")
    parser.add_argument("--sessions", nargs='+', default=["T", "E"], help="BCI2b sessions to load (default: T E)")
    parser.add_argument("--target-fs", type=float, default=250.0, help="Common fs for comparison (default 250)")
    parser.add_argument("--out", default="CODE/analysis/domain_shift_report.json", help="Output JSON path")

    args = parser.parse_args()

    loader = BCI4_2B_Loader(args.config)
    Xb, yb = loader.load_all_subjects(sessions=args.sessions)
    Xb = to_trials_channels_samples(np.asarray(Xb, dtype=np.float32))
    fsb = float(loader.sampling_rate)
    chb = ["C3", "Cz", "C4"]

    Xp, yp, fsp, chp = load_personal_npz(Path(args.personal))

    Xb, _ = select_c3_cz_c4(Xb, chb)
    Xp, _ = select_c3_cz_c4(Xp, chp)

    Xb = dc_center(Xb)
    Xp = dc_center(Xp)

    Xb = maybe_resample(Xb, fsb, args.target_fs)
    Xp = maybe_resample(Xp, fsp if fsp is not None else args.target_fs, args.target_fs)

    fs_common = float(args.target_fs)

    bp_b, fv_b = extract_features(Xb, fs_common)
    bp_p, fv_p = extract_features(Xp, fs_common)

    summary_b = summarize("bci4_2b", Xb, yb, fs_common, bp_b, fv_b)
    summary_p = summarize("personal_npg", Xp, yp, fs_common, bp_p, fv_p)

    distance = frechet_diag(fv_b, fv_p)

    report = {
        "bci": summary_b,
        "personal": summary_p,
        "feature_frechet_distance": distance,
        "recommendations": tune_recommendations(summary_b, summary_p, distance, fs_common),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
