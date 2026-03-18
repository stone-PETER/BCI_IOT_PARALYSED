#!/usr/bin/env python3
"""
Create a combined training dataset from BCI IV 2b and personal calibration data.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from bci4_2b_loader_v2 import BCI4_2B_Loader


def to_samples_channels(epochs: np.ndarray) -> np.ndarray:
    if epochs.ndim != 3:
        raise ValueError(f"Expected 3D epochs, got {epochs.shape}")

    if epochs.shape[2] == 3:
        return epochs
    if epochs.shape[1] == 3:
        return epochs.transpose(0, 2, 1)

    raise ValueError(f"Could not infer channel axis from shape {epochs.shape}")


def apply_fixed_filters(epochs: np.ndarray,
                        sampling_rate: float,
                        notch_freq: float,
                        bp_low: float,
                        bp_high: float) -> np.ndarray:
    nyq = sampling_rate / 2.0
    b_notch, a_notch = iirnotch(w0=notch_freq / nyq, Q=30)
    b_band, a_band = butter(4, [bp_low / nyq, bp_high / nyq], btype='band')

    filtered = np.zeros_like(epochs, dtype=np.float32)
    for i in range(epochs.shape[0]):
        for c in range(epochs.shape[2]):
            x = epochs[i, :, c]
            x = filtfilt(b_notch, a_notch, x)
            x = filtfilt(b_band, a_band, x)
            filtered[i, :, c] = x
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Create combined BCI2b + personal dataset")
    parser.add_argument("--personal-file", type=str, required=True, help="Path to personal calibration .npz")
    parser.add_argument("--output-file", type=str, default="datasets/combined_bci2b_personal.npz", help="Output dataset path")
    parser.add_argument("--config", type=str, default="config_2b.yaml", help="Config path for BCI2b loader")
    parser.add_argument("--sessions", nargs='+', default=["T", "E"], help="BCI2b sessions to include")
    parser.add_argument("--personal-weight", type=float, default=0.5, help="Target share of personal epochs")
    parser.add_argument("--sampling-rate", type=float, default=250.0, help="Sampling rate for filtering")
    parser.add_argument("--notch", type=float, default=50.0, help="Notch frequency")
    parser.add_argument("--bp-low", type=float, default=8.0, help="Bandpass low cutoff")
    parser.add_argument("--bp-high", type=float, default=30.0, help="Bandpass high cutoff")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    loader = BCI4_2B_Loader(args.config)

    logger.info("Loading personal calibration data")
    personal_epochs, personal_labels, rest_epochs = loader.load_personal_calibration(args.personal_file)
    personal_epochs = to_samples_channels(personal_epochs)
    rest_epochs = to_samples_channels(rest_epochs)

    logger.info("Loading BCI IV 2b benchmark data")
    benchmark_epochs, benchmark_labels = loader.load_all_subjects(sessions=args.sessions)
    benchmark_epochs = to_samples_channels(benchmark_epochs)

    target_benchmark = int(len(personal_epochs) * (1.0 - args.personal_weight) / max(args.personal_weight, 1e-6))
    target_benchmark = min(target_benchmark, len(benchmark_epochs))
    indices = np.random.choice(len(benchmark_epochs), size=target_benchmark, replace=False)
    benchmark_epochs = benchmark_epochs[indices]
    benchmark_labels = benchmark_labels[indices]

    logger.info("Applying fixed filtering to both sources")
    personal_epochs = apply_fixed_filters(personal_epochs, args.sampling_rate, args.notch, args.bp_low, args.bp_high)
    benchmark_epochs = apply_fixed_filters(benchmark_epochs, args.sampling_rate, args.notch, args.bp_low, args.bp_high)
    rest_epochs = apply_fixed_filters(rest_epochs, args.sampling_rate, args.notch, args.bp_low, args.bp_high)

    X = np.concatenate([personal_epochs, benchmark_epochs], axis=0)
    y = np.concatenate([personal_labels, benchmark_labels], axis=0)

    source = np.concatenate([
        np.array(["personal"] * len(personal_epochs), dtype=object),
        np.array(["benchmark"] * len(benchmark_epochs), dtype=object)
    ])

    out_path = Path(args.output_file)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        y=y.astype(np.int32),
        rest_epochs=rest_epochs.astype(np.float32),
        source=source,
        sampling_rate=np.array([args.sampling_rate], dtype=np.float32),
        channels=np.array(["C3", "Cz", "C4"], dtype=object),
        metadata=json.dumps({
            "personal_file": str(args.personal_file),
            "sessions": args.sessions,
            "personal_weight": args.personal_weight,
            "notch": args.notch,
            "bp_low": args.bp_low,
            "bp_high": args.bp_high,
            "n_personal": int(len(personal_epochs)),
            "n_benchmark": int(len(benchmark_epochs)),
            "n_total": int(len(X))
        })
    )

    logger.info(f"Saved combined dataset: {out_path}")
    logger.info(f"Shape X={X.shape}, y={y.shape}, rest={rest_epochs.shape}")
    logger.info(f"Class counts LEFT={np.sum(y==0)}, RIGHT={np.sum(y==1)}")


if __name__ == "__main__":
    main()
