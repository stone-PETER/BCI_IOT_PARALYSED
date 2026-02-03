#!/usr/bin/env python3
"""
Test script to demonstrate baseline calibration workflow
"""

import numpy as np
import sys
from pathlib import Path

# Add CODE directory to path
sys.path.insert(0, str(Path(__file__).parent))

from npg_inference import NPGInferenceEngine
from npg_preprocessor import NPGPreprocessor

def test_baseline_calibration():
    """Test baseline calibration and bias correction."""
    print("\n" + "="*70)
    print("Testing Baseline Calibration for 2-Class Model Bias Correction")
    print("="*70)
    
    # Create engine
    print("\n1. Creating inference engine...")
    engine = NPGInferenceEngine(
        confidence_threshold=0.65,
        smoothing_window=8,
        use_accumulator=False  # Disable for clearer testing
    )
    
    # Create preprocessor
    preprocessor = NPGPreprocessor()
    
    # Simulate rest state data (biased toward right hand)
    print("\n2. Simulating REST state (60 predictions)...")
    print("   (Model naturally biased toward RIGHT_HAND)")
    
    engine.start_calibration()
    
    for i in range(60):
        # Generate synthetic rest data
        n_samples = int(4.0 * 256)
        n_channels = 6
        t = np.arange(n_samples) / 256
        
        # Rest = mostly noise with slight bias pattern
        test_data = np.random.randn(n_samples, n_channels) * 5
        # Add slight pattern that biases toward right
        for ch in range(n_channels):
            test_data[:, ch] += 2 * np.sin(2 * np.pi * 15 * t)  # 15 Hz
        
        # Preprocess and add to calibration
        preprocessed = preprocessor.preprocess_for_model(test_data)
        engine.add_calibration_sample(preprocessed)
    
    # Finalize calibration
    print("\n3. Finalizing calibration...")
    success = engine.finalize_calibration()
    
    if not success:
        print("Calibration failed!")
        return False
    
    bias = engine.get_baseline_bias()
    print(f"\n   Baseline bias calculated: {bias}")
    print(f"   LEFT bias: {bias[0]:+.3f}")
    print(f"   RIGHT bias: {bias[1]:+.3f}")
    
    # Test predictions with and without bias correction
    print("\n4. Testing predictions...")
    
    # Generate test imagery data
    test_data = np.random.randn(n_samples, n_channels) * 5
    for ch in range(n_channels):
        alpha = 10 * np.sin(2 * np.pi * 10 * t)
        beta = 8 * np.sin(2 * np.pi * 20 * t)
        test_data[:, ch] = alpha + beta + np.random.randn(n_samples) * 2
    
    preprocessed = preprocessor.preprocess_for_model(test_data)
    
    # Test with bias correction
    class_idx, confidence, class_name = engine.predict(preprocessed)
    
    print(f"\n   Prediction with bias correction:")
    print(f"   Class: {class_name}")
    print(f"   Confidence: {confidence:.1%}")
    
    # Test without bias correction
    original_bias = engine.get_baseline_bias()
    engine.clear_baseline_bias()
    
    class_idx_no_bias, conf_no_bias, name_no_bias = engine.predict(preprocessed)
    
    print(f"\n   Prediction WITHOUT bias correction (for comparison):")
    print(f"   Class: {name_no_bias}")
    print(f"   Confidence: {conf_no_bias:.1%}")
    
    # Restore bias
    engine.set_baseline_bias(original_bias)
    
    # Show statistics
    print("\n5. Statistics:")
    stats = engine.get_statistics()
    if stats['baseline_bias']['calibrated']:
        print(f"   Baseline bias active")
        print(f"   LEFT bias: {stats['baseline_bias']['left_bias']:+.3f}")
        print(f"   RIGHT bias: {stats['baseline_bias']['right_bias']:+.3f}")
    else:
        print(f"   No baseline bias")
    
    print("\n" + "="*70)
    print("Baseline calibration test completed!")
    print("="*70)
    
    print("\n📝 To use in real system:")
    print("   1. Run calibration: python npg_realtime_bci.py --simulate --calibrate")
    print("   2. Run BCI: python npg_realtime_bci.py --simulate")
    
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        test_baseline_calibration()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
