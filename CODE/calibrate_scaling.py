#!/usr/bin/env python3
"""
NPG Lite Scaling Factor Calibration Tool

This script helps determine the correct scaling factor to convert
NPG Lite output to microvolts (µV) to match the BCI training data.

Usage:
    python calibrate_scaling.py --simulate  # Test with simulator
    python calibrate_scaling.py --direct --port COM6  # Test with hardware
"""

import numpy as np
import sys
import argparse
import time
from pathlib import Path

# Add CODE directory to path
sys.path.insert(0, str(Path(__file__).parent))

from npg_lite_adapter import NPGLiteAdapter, NPGLiteSimulator, NPGLiteDirectSerial


def analyze_signal_amplitude(data: np.ndarray, channel_names: list = None):
    """
    Analyze signal amplitude characteristics.
    
    Args:
        data: EEG data (n_samples, n_channels)
        channel_names: Optional channel labels
    """
    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(data.shape[1])]
    
    print("\n" + "="*70)
    print("Signal Amplitude Analysis")
    print("="*70)
    
    for ch_idx in range(min(data.shape[1], 6)):  # Analyze first 6 channels
        ch_data = data[:, ch_idx]
        
        # Calculate statistics
        rms = np.sqrt(np.mean(ch_data**2))
        peak_to_peak = np.max(ch_data) - np.min(ch_data)
        mean_val = np.mean(ch_data)
        std_val = np.std(ch_data)
        
        print(f"\n{channel_names[ch_idx]}:")
        print(f"  Mean:         {mean_val:12.6f}")
        print(f"  Std Dev:      {std_val:12.6f}")
        print(f"  RMS:          {rms:12.6f}")
        print(f"  Peak-to-Peak: {peak_to_peak:12.6f}")
    
    # Overall statistics
    overall_rms = np.sqrt(np.mean(data**2))
    
    print("\n" + "-"*70)
    print(f"Overall RMS across all channels: {overall_rms:.6f}")
    
    return overall_rms


def suggest_scaling_factor(measured_rms: float):
    """
    Suggest scaling factor based on measured RMS.
    
    Expected EEG at rest: 10-50 µV RMS
    """
    print("\n" + "="*70)
    print("Scaling Factor Recommendations")
    print("="*70)
    
    # Expected range
    expected_min = 10   # µV
    expected_max = 50   # µV
    expected_typical = 25  # µV
    
    print(f"\nExpected EEG RMS at rest: {expected_min}-{expected_max} µV")
    print(f"Measured RMS: {measured_rms:.6f} (in current units)")
    
    # Calculate suggested scaling factors
    print("\n" + "-"*70)
    print("Possible Scenarios:")
    print("-"*70)
    
    # Scenario 1: Already in microvolts
    if 5 <= measured_rms <= 100:
        print("\n✅ SCENARIO 1: Data already in microvolts")
        print(f"   Measured: {measured_rms:.1f} µV (within expected range)")
        print("   Recommended scaling_factor: 1.0")
        print("   No scaling needed!")
        return 1.0
    
    # Scenario 2: Data in volts
    if 0.000005 <= measured_rms <= 0.0001:
        scaling = 1e6
        scaled_rms = measured_rms * scaling
        print("\n📊 SCENARIO 2: Data appears to be in Volts")
        print(f"   Measured: {measured_rms:.9f} V")
        print(f"   After scaling: {scaled_rms:.1f} µV")
        print(f"   Recommended scaling_factor: {scaling:.0e}")
        return scaling
    
    # Scenario 3: Data in millivolts
    if 0.005 <= measured_rms <= 0.1:
        scaling = 1000
        scaled_rms = measured_rms * scaling
        print("\n📊 SCENARIO 3: Data appears to be in millivolts")
        print(f"   Measured: {measured_rms:.6f} mV")
        print(f"   After scaling: {scaled_rms:.1f} µV")
        print(f"   Recommended scaling_factor: {scaling}")
        return scaling
    
    # Scenario 4: Raw ADC values (needs more info)
    if measured_rms > 100:
        print("\n⚠️  SCENARIO 4: Data appears to be raw ADC values")
        print(f"   Measured: {measured_rms:.0f} (ADC units)")
        print("\n   To calculate scaling factor, you need:")
        print("   1. ADC resolution (bits): e.g., 12-bit = 4096 levels")
        print("   2. Reference voltage: e.g., 3.3V")
        print("   3. Gain stage: Check NPG Lite hardware specs")
        print("\n   Formula: scaling_factor = (V_ref / ADC_max) * gain * 1e6")
        print("\n   Example for 12-bit ADC, 3.3V ref, gain=24:")
        print(f"   scaling_factor = (3.3 / 4096) * 24 * 1e6 = {(3.3/4096)*24*1e6:.0f}")
        return None
    
    # Scenario 5: Unknown
    print("\n❓ SCENARIO 5: Cannot determine unit")
    print(f"   Measured RMS ({measured_rms:.6f}) outside expected ranges")
    print("   Manual investigation required")
    print("\n   Check NPG Lite documentation for output unit")
    return None


def main():
    parser = argparse.ArgumentParser(description='NPG Lite Scaling Factor Calibration')
    parser.add_argument('--simulate', action='store_true',
                       help='Use simulator for testing')
    parser.add_argument('--direct', action='store_true',
                       help='Direct serial connection to NPG Lite')
    parser.add_argument('--port', type=str, default='COM6',
                       help='Serial port (default: COM6)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Recording duration in seconds (default: 10)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("NPG Lite Scaling Factor Calibration Tool")
    print("="*70)
    
    # Create adapter
    if args.simulate:
        print("\nUsing SIMULATOR mode")
        adapter = NPGLiteSimulator()
    elif args.direct:
        print(f"\nUsing DIRECT mode (port: {args.port})")
        adapter = NPGLiteDirectSerial(port=args.port)
    else:
        print("\nUsing LSL mode (Chords-Python)")
        adapter = NPGLiteAdapter()
    
    # Connect
    print(f"\nConnecting...")
    if not adapter.connect():
        print("❌ Failed to connect!")
        print("\nTroubleshooting:")
        if args.direct:
            print("  - Check COM port with Device Manager")
            print("  - Try different baud rate: --baudrate 115200")
        else:
            print("  - Make sure Chords-Python is running")
            print("  - Run: python -m chordspy.connection --protocol usb")
        sys.exit(1)
    
    print("✅ Connected!")
    
    # Instructions
    print("\n" + "="*70)
    print("CALIBRATION INSTRUCTIONS")
    print("="*70)
    print(f"\n1. Sit still and relax for {args.duration} seconds")
    print("2. Look at a fixed point")
    print("3. Breathe normally")
    print("4. Do NOT imagine any movement")
    print("\nPress ENTER when ready...")
    input()
    
    # Start recording
    adapter.start_streaming()
    print(f"\n⏱️  Recording for {args.duration} seconds...")
    
    # Collect data
    time.sleep(1)  # Initial buffer fill
    
    samples_needed = int(args.duration * 256)  # Assuming 256 Hz
    data_buffer = []
    
    start_time = time.time()
    while time.time() - start_time < args.duration:
        chunk = adapter.get_latest_data(n_samples=256)
        if chunk is not None and len(chunk) > 0:
            data_buffer.append(chunk)
        time.sleep(0.2)
    
    adapter.stop_streaming()
    
    # Combine data
    if not data_buffer:
        print("\n❌ No data received!")
        sys.exit(1)
    
    data = np.vstack(data_buffer)
    print(f"\n✅ Recorded {len(data)} samples from {data.shape[1]} channels")
    
    # Analyze
    channel_names = ['C3', 'Cz', 'C4', 'Ch4', 'Ch5', 'Ch6'][:data.shape[1]]
    measured_rms = analyze_signal_amplitude(data, channel_names)
    
    # Suggest scaling
    suggested_scaling = suggest_scaling_factor(measured_rms)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY & NEXT STEPS")
    print("="*70)
    
    if suggested_scaling:
        print(f"\n1. Update your preprocessor code:")
        print(f"\n   preprocessor = NPGPreprocessor(")
        print(f"       scaling_factor={suggested_scaling},")
        print(f"       use_bipolar=True")
        print(f"   )")
        
        print(f"\n2. Run baseline calibration:")
        print(f"   python npg_realtime_bci.py --simulate --calibrate")
        
        print(f"\n3. Test BCI system:")
        print(f"   python npg_realtime_bci.py --simulate")
    else:
        print("\n⚠️  Could not determine scaling factor automatically")
        print("   Consult NPG Lite documentation for output unit")
        print("   Or contact hardware manufacturer for specifications")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
