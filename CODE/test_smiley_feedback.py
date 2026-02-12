#!/usr/bin/env python3
"""
Smiley Feedback Demo
Demonstrates the running integral classifier that reduces prediction flickering

Run with default settings (Leaky Accumulator):
    python test_smiley_feedback.py

Run with Smiley Feedback:
    python test_smiley_feedback.py --smiley-feedback

Compare both:
    python test_smiley_feedback.py --compare
"""

import numpy as np
import time
import logging
from pathlib import Path
import sys

# Add CODE directory to path
sys.path.insert(0, str(Path(__file__).parent))

from npg_inference import NPGInferenceEngine, SmileyFeedback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_synthetic_epoch(class_label=0, strength=0.8, noise_level=0.2):
    """
    Generate synthetic EEG-like epoch data.
    
    Args:
        class_label: 0 for left, 1 for right
        strength: Signal strength (0-1)
        noise_level: Noise strength (0-1)
    
    Returns:
        Preprocessed data ready for model (1, 3, 1000, 1)
    """
    # Generate random data with some structure
    n_channels = 3
    n_timepoints = 1000
    
    # Base oscillations (simulating brain rhythms)
    t = np.linspace(0, 4, n_timepoints)
    alpha = np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    beta = np.sin(2 * np.pi * 20 * t)   # 20 Hz beta
    
    data = np.zeros((n_timepoints, n_channels))
    
    for ch in range(n_channels):
        # Different channels have different patterns
        if class_label == 0:  # Left hand
            if ch == 0:  # C3 suppression
                signal = -strength * alpha + 0.5 * beta
            else:
                signal = 0.3 * alpha + 0.3 * beta
        else:  # Right hand
            if ch == 2:  # C4 suppression
                signal = -strength * alpha + 0.5 * beta
            else:
                signal = 0.3 * alpha + 0.3 * beta
        
        # Add noise
        noise = np.random.randn(n_timepoints) * noise_level * 10
        data[:, ch] = signal + noise
    
    # Reshape to model input format
    # (1, 3, 1000, 1) - batch, channels, timepoints, depth
    data_reshaped = data.T.reshape(1, n_channels, n_timepoints, 1)
    
    return data_reshaped


def test_accumulator_mode():
    """Test with Leaky Accumulator (default mode)."""
    print("\n" + "="*70)
    print("Testing LEAKY ACCUMULATOR Mode (Default)")
    print("="*70)
    
    engine = NPGInferenceEngine(
        use_accumulator=True,
        accumulator_threshold=2.0,
        accumulator_decay=0.15,
        use_smiley_feedback=False
    )
    
    print("\nSimulating 20 predictions with varying confidence...")
    print("(Left hand dominant with some noise)\n")
    
    for i in range(20):
        # Generate mostly left-hand epochs with some random variation
        if i % 5 == 0:  # Occasional uncertainty
            data = generate_synthetic_epoch(class_label=np.random.randint(2), strength=0.4)
        else:
            data = generate_synthetic_epoch(class_label=0, strength=0.7 + np.random.rand()*0.2)
        
        # Predict with accumulator
        class_idx, confidence, class_name, is_triggered = engine.predict_with_accumulator(data)
        
        # Get bucket status
        status = engine.get_accumulator_status()
        left_pct = status['buckets']['LEFT_HAND']['percent_full']
        right_pct = status['buckets']['RIGHT_HAND']['percent_full']
        
        trigger_icon = "🎯" if is_triggered else "  "
        print(f"{i+1:2}. {trigger_icon} {class_name:12} | Conf: {confidence:5.1%} | "
              f"Buckets: L={left_pct:3.0f}% R={right_pct:3.0f}%")
        
        time.sleep(0.1)
    
    print(f"\nStatistics:\n{engine.get_statistics()}")


def test_smiley_feedback_mode():
    """Test with Smiley Feedback running integral."""
    print("\n" + "="*70)
    print("Testing SMILEY FEEDBACK Mode (BCI Competition Strategy)")
    print("="*70)
    
    engine = NPGInferenceEngine(
        use_smiley_feedback=True,
        smiley_window_duration=2.0,
        smiley_threshold=3.5,
        smiley_prediction_rate=2.2,
        use_accumulator=False
    )
    
    print("\nSimulating 20 predictions with varying confidence...")
    print("(Left hand dominant with some noise)")
    print("Window: 2s (~4-5 predictions), Threshold: 3.5\n")
    
    for i in range(20):
        # Generate mostly left-hand epochs with some random variation
        if i % 5 == 0:  # Occasional uncertainty
            data = generate_synthetic_epoch(class_label=np.random.randint(2), strength=0.4)
        else:
            data = generate_synthetic_epoch(class_label=0, strength=0.7 + np.random.rand()*0.2)
        
        # Predict with smiley feedback
        class_idx, confidence, class_name, is_triggered, winning_sum = \
            engine.predict_with_smiley_feedback(data)
        
        # Get integral status
        status = engine.get_smiley_feedback_status()
        sums = status['integral_sums']
        left_pct = sums['LEFT_HAND']['percent_to_trigger']
        right_pct = sums['RIGHT_HAND']['percent_to_trigger']
        buffer_fill = status['buffer_fill_count']
        
        trigger_icon = "🎯" if is_triggered else "  "
        print(f"{i+1:2}. {trigger_icon} {class_name:12} | Conf: {confidence:5.1%} | "
              f"Buffer: {buffer_fill}/5 | Integral: L={left_pct:3.0f}% R={right_pct:3.0f}%")
        
        time.sleep(0.1)
    
    print(f"\nStatistics:\n{engine.get_statistics()}")


def compare_modes():
    """Compare both modes side-by-side."""
    print("\n" + "="*70)
    print("COMPARING ACCUMULATOR vs SMILEY FEEDBACK")
    print("="*70)
    
    engine_acc = NPGInferenceEngine(
        use_accumulator=True,
        use_smiley_feedback=False
    )
    
    engine_smiley = NPGInferenceEngine(
        use_smiley_feedback=True,
        use_accumulator=False,
        smiley_threshold=3.5
    )
    
    print("\nGenerating same sequence for both...\n")
    print(f"{'#':>3} | {'Accumulator':<25} | {'Smiley Feedback':<25}")
    print("-" * 70)
    
    # Generate fixed seed for reproducibility
    np.random.seed(42)
    
    for i in range(15):
        # Generate consistent epochs
        if i % 5 == 0:
            data = generate_synthetic_epoch(class_label=np.random.randint(2), strength=0.4)
        else:
            data = generate_synthetic_epoch(class_label=0, strength=0.7 + np.random.rand()*0.2)
        
        # Accumulator prediction
        _, conf_acc, name_acc, trig_acc = engine_acc.predict_with_accumulator(data)
        icon_acc = "🎯" if trig_acc else "  "
        
        # Smiley prediction
        _, conf_smiley, name_smiley, trig_smiley, sum_smiley = \
            engine_smiley.predict_with_smiley_feedback(data)
        icon_smiley = "🎯" if trig_smiley else "  "
        
        print(f"{i+1:2}. | {icon_acc} {name_acc:12} {conf_acc:5.1%} | "
              f"{icon_smiley} {name_smiley:12} {conf_smiley:5.1%}")
        
        time.sleep(0.1)
    
    print("\n" + "="*70)
    print("Key Differences:")
    print("- Accumulator: Gradually fills 'buckets', decays over time")
    print("- Smiley Feedback: Integrates probabilities over fixed window")
    print("- Both reduce flickering compared to frame-by-frame classification")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Smiley Feedback vs Accumulator')
    parser.add_argument('--smiley-feedback', action='store_true',
                       help='Test Smiley Feedback mode')
    parser.add_argument('--accumulator', action='store_true',
                       help='Test Accumulator mode')
    parser.add_argument('--compare', action='store_true',
                       help='Compare both modes')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_modes()
    elif args.smiley_feedback:
        test_smiley_feedback_mode()
    elif args.accumulator:
        test_accumulator_mode()
    else:
        # Default: show both
        test_accumulator_mode()
        test_smiley_feedback_mode()
        compare_modes()
