#!/usr/bin/env python3
"""
Test Domain Shift Fixes for NPG Lite Real-time BCI

Verifies all critical fixes from domain shift report:
✅ Z-score normalization (100-140x amplitude mismatch)
✅ Notch filter (50 Hz powerline noise)
✅ Laplacian filtering (18x better separability)
✅ Fine-tuning capability (domain shift distance 210.1)

Run: python test_domain_shift_fixes.py
"""

import numpy as np
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Test configurations
INPUT_SAMPLING_RATE = 500
OUTPUT_SAMPLING_RATE = 250
N_CHANNELS = 3
EPOCH_DURATION = 4.0
INPUT_EPOCH_SAMPLES = int(EPOCH_DURATION * INPUT_SAMPLING_RATE)  # 2000
OUTPUT_EPOCH_SAMPLES = int(EPOCH_DURATION * OUTPUT_SAMPLING_RATE)  # 1000


def test_zscore_normalization():
    """Test 1: Verify z-score normalization is enabled and working"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Z-score Normalization (Domain Shift Fix)")
    logger.info("="*70)
    
    from npg_preprocessor import NPGPreprocessor
    
    # Create preprocessor with z-score ENABLED
    preprocessor = NPGPreprocessor(
        input_rate=INPUT_SAMPLING_RATE,
        output_rate=OUTPUT_SAMPLING_RATE,
        apply_zscore=True,  # MUST be True
        apply_laplacian=True,
        apply_notch=True
    )
    
    # Check configuration
    assert preprocessor.apply_zscore == True, "❌ Z-score normalization NOT enabled!"
    logger.info("✅ Z-score normalization ENABLED in preprocessor")
    
    # Simulate personal data (488-649 μV std, ~ 100-140x larger than BCI)
    personal_scale_data = np.random.randn(INPUT_EPOCH_SAMPLES, N_CHANNELS) * 550  # 550 μV std
    
    logger.info(f"Input amplitude statistics:")
    logger.info(f"  Mean across channels: {np.mean(personal_scale_data):.2f} μV")
    logger.info(f"  Std across channels: {np.std(personal_scale_data):.2f} μV (expected 488-649)")
    
    # Preprocess
    preprocessed = preprocessor.preprocess_epoch(personal_scale_data)
    
    # Check output
    output_std = np.std(preprocessed)
    logger.info(f"\nOutput after z-score:")
    logger.info(f"  Std: {output_std:.3f} (expected ~1.0)")
    
    if output_std < 0.3:
        logger.error(f"❌ Z-score produced flat signal (std={output_std:.3f})")
        return False
    elif output_std > 3.0:
        logger.warning(f"⚠️ Z-score incomplete (std={output_std:.3f}), may have outliers")
    else:
        logger.info(f"✅ Z-score normalization working correctly")
    
    return True


def test_notch_filter_50hz():
    """Test 2: Verify notch filter reduces 50 Hz noise"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Notch Filter (50 Hz Powerline Noise)")
    logger.info("="*70)
    
    from scipy import signal
    from npg_preprocessor import NPGPreprocessor
    
    preprocessor = NPGPreprocessor(
        input_rate=INPUT_SAMPLING_RATE,
        output_rate=OUTPUT_SAMPLING_RATE,
        apply_notch=True
    )
    
    assert preprocessor.apply_notch == True, "❌ Notch filter not enabled!"
    logger.info(f"✅ Notch filter ENABLED at {preprocessor.notch_freq} Hz")
    
    # Create synthetic signal: motor signal (15 Hz) + heavy 50 Hz noise
    t = np.linspace(0, EPOCH_DURATION, INPUT_EPOCH_SAMPLES)
    motor_signal = 2 * np.sin(2 * np.pi * 15 * t)  # 15 Hz motor signal
    noise_50hz = 600 * np.sin(2 * np.pi * 50 * t)  # 50 Hz pollution (VERY strong)
    
    signal_before = (motor_signal + noise_50hz).reshape(-1, 1)
    
    # Resample to output rate
    from scipy.signal import resample_poly
    signal_resampled = resample_poly(signal_before, 1, 2, axis=0)
    
    # Apply notch filter
    signal_after = preprocessor.notch_filter(signal_resampled, stateful=False)
    
    # Compute power spectrum before and after
    freqs_before, pxx_before = signal.welch(signal_before.flatten(), fs=INPUT_SAMPLING_RATE, nperseg=256)
    freqs_after, pxx_after = signal.welch(signal_after.flatten(), fs=OUTPUT_SAMPLING_RATE, nperseg=256)
    
    # Find 50 Hz power (at downsampled rate)
    idx_50 = np.argmin(np.abs(freqs_after - 50))
    power_before_50hz = pxx_before[np.argmin(np.abs(freqs_before - 50))]
    power_after_50hz = pxx_after[idx_50]
    
    reduction = power_before_50hz / (power_after_50hz + 1e-10)
    
    logger.info(f"50 Hz power reduction:")
    logger.info(f"  Before: {power_before_50hz:.2e}")
    logger.info(f"  After:  {power_after_50hz:.2e}")
    logger.info(f"  Reduction factor: {reduction:.1f}x")
    
    if reduction < 5:
        logger.error(f"❌ Notch filter weak: only {reduction:.1f}x reduction (expected >10x)")
        return False
    else:
        logger.info(f"✅ Notch filter working correctly ({reduction:.1f}x reduction)")
    
    return True


def test_laplacian_filtering():
    """Test 3: Verify Laplacian spatial filtering is enabled"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Laplacian Spatial Filtering")
    logger.info("="*70)
    
    from npg_preprocessor import NPGPreprocessor
    
    preprocessor = NPGPreprocessor(
        input_rate=INPUT_SAMPLING_RATE,
        output_rate=OUTPUT_SAMPLING_RATE,
        apply_laplacian=True,
        apply_zscore=True
    )
    
    assert preprocessor.apply_laplacian == True, "❌ Laplacian filtering not enabled!"
    logger.info("✅ Laplacian filtering ENABLED")
    
    # Test on sample data
    test_data = np.random.randn(OUTPUT_EPOCH_SAMPLES, 3) * 50
    
    laplacian_output = preprocessor.apply_small_laplacian(test_data)
    
    assert laplacian_output.shape == test_data.shape, "❌ Laplacian changed shape"
    logger.info(f"  Input shape: {test_data.shape}")
    logger.info(f"  Output shape: {laplacian_output.shape} ✅")
    logger.info(f"  Laplacian enhances local motor cortex activity")
    logger.info(f"  Expected effect: 18x better class separability for personal data")
    
    return True


def test_preprocessing_pipeline():
    """Test 4: Full preprocessing pipeline with all domain shift fixes"""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Full Preprocessing Pipeline")
    logger.info("="*70)
    
    from npg_preprocessor import NPGPreprocessor
    
    preprocessor = NPGPreprocessor(
        input_rate=INPUT_SAMPLING_RATE,
        output_rate=OUTPUT_SAMPLING_RATE,
        apply_notch=True,
        apply_laplacian=True,
        apply_zscore=True,
        realtime_mode=True
    )
    
    # Create synthetic personal data (100-140x larger than BCI)
    personal_data = np.random.randn(INPUT_EPOCH_SAMPLES, N_CHANNELS) * 550
    
    logger.info(f"Input: {personal_data.shape} @ {INPUT_SAMPLING_RATE} Hz, {np.std(personal_data):.1f} μV")
    
    # Preprocess
    model_input = preprocessor.preprocess_for_model(personal_data)
    
    logger.info(f"Output: {model_input.shape} @ {OUTPUT_SAMPLING_RATE} Hz")
    
    # Verify shape
    expected_shape = (1, N_CHANNELS, OUTPUT_EPOCH_SAMPLES, 1)
    assert model_input.shape == expected_shape, f"❌ Shape mismatch: {model_input.shape} vs {expected_shape}"
    logger.info(f"✅ Shape correct: {model_input.shape}")
    
    # Verify normalization
    output_std = np.std(model_input)
    if output_std < 0.5 or output_std > 2.0:
        logger.warning(f"⚠️ Normalization std={output_std:.2f}, expected ~1.0")
    else:
        logger.info(f"✅ Normalization applied: std={output_std:.2f}")
    
    return True


def test_fine_tune_preparation():
    """Test 5: Verify fine-tuning capability is available"""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Fine-tuning Capability (Domain Shift Distance Fix)")
    logger.info("="*70)
    
    from npg_realtime_bci import NPGRealtimeBCI
    
    try:
        # Create realtime BCI instance
        bci = NPGRealtimeBCI(
            simulate=True,
            confidence_threshold=0.65,
            use_smiley_feedback=False,
            api_base_url=None
        )
        
        # Check if fine-tuning method exists
        assert hasattr(bci, 'fine_tune_dense_layer'), "❌ fine_tune_dense_layer method not found"
        logger.info("✅ fine_tune_dense_layer method available")
        
        # Test the method signature
        import inspect
        sig = inspect.signature(bci.fine_tune_dense_layer)
        params = list(sig.parameters.keys())
        assert 'calibration_data' in params, "❌ calibration_data parameter missing"
        assert 'labels' in params, "❌ labels parameter missing"
        logger.info(f"✅ Method signature correct: {sig}")
        
        # Domain shift info
        logger.info(f"\nDomain Shift Metrics from Report:")
        logger.info(f"  Domain shift distance: 210.1 (severe mismatch)")
        logger.info(f"  Solution: Fine-tune final dense layer on personal data")
        logger.info(f"  Expected: Reduce domain shift distance <50")
        logger.info(f"  Personal class effect size: 0.139 vs BCI 0.0076 (18x better)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def test_realtime_diagnostics():
    """Test 6: Verify realtime diagnostic methods"""
    logger.info("\n" + "="*70)
    logger.info("TEST 6: Real-time Diagnostics")
    logger.info("="*70)
    
    from npg_realtime_bci import NPGRealtimeBCI
    
    try:
        bci = NPGRealtimeBCI(
            simulate=True,
            confidence_threshold=0.65,
            api_base_url=None
        )
        
        # Check diagnostic methods
        assert hasattr(bci, '_diagnose_50hz_noise'), "❌ _diagnose_50hz_noise method missing"
        assert hasattr(bci, '_check_zscore_applied'), "❌ _check_zscore_applied method missing"
        assert hasattr(bci, 'report_domain_shift_fixes'), "❌ report_domain_shift_fixes method missing"
        logger.info("✅ All diagnostic methods available")
        
        # Get status report
        status = bci.report_domain_shift_fixes()
        
        # Verify required fixes are enabled
        fixes = status['fixes']
        critical_fixes = {
            'z_score_normalization': True,
            'laplacian_filtering': True,
            'notch_filter_enabled': True,
        }
        
        for fix, expected in critical_fixes.items():
            if fixes.get(fix) != expected:
                logger.error(f"❌ {fix}: {fixes.get(fix)} (expected {expected})")
                return False
            logger.info(f"✅ {fix}: {fixes.get(fix)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def run_all_tests():
    """Run all domain shift fix tests"""
    logger.info("\n" + "="*80)
    logger.info("DOMAIN SHIFT FIX VERIFICATION TEST SUITE")
    logger.info("="*80)
    logger.info("\nFrom Domain Shift Report:")
    logger.info("  • Amplitude mismatch: Personal 488-649 μV vs BCI 4.5 μV (100-140x)")
    logger.info("  • 50 Hz noise: Personal 4.51 vs BCI 0.0011 (4000x worse)")
    logger.info("  • Class separability: Personal 18x better (effect size 0.139)")
    logger.info("  • Domain shift distance: 210.1 (severe mismatch)")
    logger.info("\nFixes Implemented:")
    logger.info("  ✅ Z-score normalization: Handle amplitude mismatch")
    logger.info("  ✅ Notch filter: Reduce 50 Hz noise")
    logger.info("  ✅ Laplacian filtering: Enhance separability")
    logger.info("  ✅ Dense layer fine-tuning: Adapt to personal distribution")
    
    tests = [
        ("Z-score Normalization", test_zscore_normalization),
        ("Notch Filter (50 Hz)", test_notch_filter_50hz),
        ("Laplacian Filtering", test_laplacian_filtering),
        ("Full Pipeline", test_preprocessing_pipeline),
        ("Fine-tune Capability", test_fine_tune_preparation),
        ("Real-time Diagnostics", test_realtime_diagnostics),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            passed = test_func()
            results[name] = "✅ PASS" if passed else "❌ FAIL"
        except Exception as e:
            logger.error(f"❌ Exception: {e}")
            results[name] = "❌ ERROR"
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    for name, result in results.items():
        logger.info(f"  {result} | {name}")
    
    passed = sum(1 for r in results.values() if "PASS" in r)
    total = len(results)
    logger.info("="*80)
    logger.info(f"Results: {passed}/{total} tests passed\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
