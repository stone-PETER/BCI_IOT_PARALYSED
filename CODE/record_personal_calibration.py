#!/usr/bin/env python3
"""
Personal EEG Calibration Data Recorder

Guided session to collect labeled motor imagery trials for model fine-tuning:
- LEFT HAND imagery: 100 trials
- RIGHT HAND imagery: 100 trials  
- REST (neutral state): 50 trials

Total duration: ~25 minutes (6 seconds per trial)

Usage:
    # With real NPG Lite device
    python record_personal_calibration.py --user-id=john_doe
    
    # With simulator for testing
    python record_personal_calibration.py --user-id=test_user --simulate
    
    # Custom trial counts
    python record_personal_calibration.py --user-id=john_doe --left=50 --right=50 --rest=30
"""

import numpy as np
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys
from typing import List, Tuple
import os

# Add CODE directory to path
sys.path.insert(0, str(Path(__file__).parent))

from npg_lite_adapter import NPGLiteAdapter, NPGLiteSimulator
from npg_preprocessor import NPGPreprocessor


# Calibration parameters
TRIAL_DURATION = 4.0  # seconds of motor imagery
REST_DURATION = 2.0   # seconds between trials
SAMPLING_RATE = 500   # Hz (NPG Lite)
OUTPUT_SAMPLING_RATE = 250  # Hz (after preprocessing)
N_CHANNELS = 3        # C3, Cz, C4
EPOCH_SAMPLES = int(TRIAL_DURATION * OUTPUT_SAMPLING_RATE)  # 1000 samples @ 250Hz


class CalibrationRecorder:
    """Records personal calibration data with guided visual cues."""
    
    def __init__(self, 
                 user_id: str,
                 simulate: bool = False,
                 save_dir: str = "calibration_data"):
        """
        Initialize calibration recorder.
        
        Args:
            user_id: Unique identifier for user
            simulate: Use simulator instead of real device
            save_dir: Directory to save calibration data
        """
        self.user_id = user_id
        self.simulate = simulate
        self.save_dir = Path(__file__).parent / save_dir
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        if simulate:
            self.adapter = NPGLiteSimulator()
            self.logger.info("Using NPG Lite SIMULATOR")
        else:
            self.adapter = NPGLiteAdapter()
            self.logger.info("Using real NPG Lite device")
        
        self.preprocessor = NPGPreprocessor(
            input_rate=SAMPLING_RATE,
            output_rate=OUTPUT_SAMPLING_RATE,
            apply_bandpass=True,
            bandpass_low=8.0,
            bandpass_high=30.0,
            apply_notch=True,
            notch_freq=50.0,
            use_car=False,  # Use Small Laplacian for 3 channels
            apply_zscore=True
        )
        
        # Data storage
        self.epochs_left = []
        self.epochs_right = []
        self.epochs_rest = []
        
        self.logger.info(f"Calibration recorder initialized for user: {user_id}")
    
    def connect(self) -> bool:
        """Connect to EEG device."""
        self.logger.info("Connecting to EEG device...")
        
        if self.simulate:
            success = self.adapter.connect()
        else:
            success = self.adapter.connect()
        
        if success:
            self.logger.info("✅ Connected successfully")
            return True
        else:
            self.logger.error("❌ Connection failed")
            return False
    
    def show_instructions(self, n_left: int, n_right: int, n_rest: int):
        """Display calibration instructions."""
        total_trials = n_left + n_right + n_rest
        estimated_time = (total_trials * (TRIAL_DURATION + REST_DURATION)) / 60
        
        print("\n" + "="*70)
        print("PERSONAL EEG CALIBRATION SESSION")
        print("="*70)
        print(f"\nUser ID: {self.user_id}")
        print(f"\nTotal trials: {total_trials}")
        print(f"  • LEFT HAND imagery:  {n_left} trials")
        print(f"  • RIGHT HAND imagery: {n_right} trials")
        print(f"  • REST (neutral):     {n_rest} trials")
        print(f"\nEstimated time: {estimated_time:.1f} minutes")
        print("\n" + "="*70)
        print("INSTRUCTIONS:")
        print("="*70)
        print("\n1. You will see text cues on screen:")
        print("   - 'IMAGINE LEFT HAND'  → Imagine clenching your LEFT fist")
        print("   - 'IMAGINE RIGHT HAND' → Imagine clenching your RIGHT fist")
        print("   - 'STAY NEUTRAL'       → Relax, no motor imagery")
        print("\n2. DO NOT actually move your hands - only IMAGINE the movement")
        print("\n3. Focus on the kinesthetic feeling of the movement")
        print("\n4. Each trial lasts 4 seconds, then 2 seconds rest")
        print("\n5. Try to maintain consistent imagery throughout each trial")
        print("\n6. You can take breaks between blocks if needed")
        print("="*70 + "\n")
    
    def display_cue(self, cue_type: str, trial_num: int, total_trials: int):
        """
        Display visual cue for motor imagery.
        
        Args:
            cue_type: 'LEFT', 'RIGHT', or 'REST'
            trial_num: Current trial number
            total_trials: Total number of trials
        """
        # Clear screen (works on Windows and Unix)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" * 5)
        print("="*70)
        
        if cue_type == "LEFT":
            icon = "👈" * 5
            text = "IMAGINE LEFT HAND"
            instruction = "Imagine clenching your LEFT fist"
        elif cue_type == "RIGHT":
            icon = "👉" * 5
            text = "IMAGINE RIGHT HAND"
            instruction = "Imagine clenching your RIGHT fist"
        else:  # REST
            icon = "⚪" * 5
            text = "STAY NEUTRAL"
            instruction = "Relax - no motor imagery"
        
        print(f"{icon:^70}")
        print(f"{text:^70}")
        print(f"{instruction:^70}")
        print("="*70)
        print(f"\nTrial: {trial_num}/{total_trials}")
        print(f"Recording for {TRIAL_DURATION:.1f} seconds...")
        print("\n" * 5)
    
    def record_trial(self, cue_type: str, trial_num: int, total_trials: int) -> np.ndarray:
        """
        Record a single trial with visual cue.
        
        Args:
            cue_type: 'LEFT', 'RIGHT', or 'REST'
            trial_num: Current trial number
            total_trials: Total trials in session
        
        Returns:
            Preprocessed epoch (3, 1000) or None if failed
        """
        # Display cue
        self.display_cue(cue_type, trial_num, total_trials)
        
        # Wait for initial samples to buffer
        time.sleep(0.5)
        
        # Record data
        samples_needed = int(TRIAL_DURATION * SAMPLING_RATE)  # 2000 @ 500Hz
        start_time = time.time()
        
        # Collect data
        raw_data = self.adapter.get_latest_data(n_samples=samples_needed)
        
        if raw_data is None or len(raw_data) < samples_needed * 0.8:
            self.logger.warning(f"⚠️  Insufficient data collected for trial {trial_num}")
            return None
        
        # Take exactly the samples we need
        raw_data = raw_data[-samples_needed:]
        
        # Preprocess: 500Hz → 250Hz, bandpass, normalize
        # Input shape: (2000, 3) @ 500Hz
        # Output shape: (3, 1000) @ 250Hz
        preprocessed = self.preprocessor.preprocess_epoch(raw_data)
        
        if preprocessed is None:
            self.logger.warning(f"⚠️  Preprocessing failed for trial {trial_num}")
            return None
        
        # Remove model format dimension if present
        if preprocessed.ndim == 4:  # (1, 3, 1000, 1)
            preprocessed = preprocessed[0, :, :, 0]  # → (3, 1000)
        
        elapsed = time.time() - start_time
        self.logger.debug(f"Trial {trial_num} recorded in {elapsed:.2f}s")
        
        return preprocessed
    
    def inter_trial_rest(self):
        """Display rest period between trials."""
        print("\n" + "-"*70)
        print(f"Rest for {REST_DURATION:.1f} seconds...")
        print("-"*70)
        time.sleep(REST_DURATION)
    
    def record_session(self, 
                      n_left: int = 100,
                      n_right: int = 100,
                      n_rest: int = 50) -> bool:
        """
        Record complete calibration session.
        
        Args:
            n_left: Number of left hand trials
            n_right: Number of right hand trials
            n_rest: Number of rest trials
        
        Returns:
            True if session completed successfully
        """
        # Show instructions
        self.show_instructions(n_left, n_right, n_rest)
        
        input("Press ENTER when ready to start calibration...")
        
        # Start streaming
        self.adapter.start_streaming()
        time.sleep(2.0)  # Warm-up period
        
        # Create randomized trial sequence
        trial_sequence = (
            [('LEFT', i+1) for i in range(n_left)] +
            [('RIGHT', i+1) for i in range(n_right)] +
            [('REST', i+1) for i in range(n_rest)]
        )
        np.random.shuffle(trial_sequence)
        
        total_trials = len(trial_sequence)
        successful_trials = {'LEFT': 0, 'RIGHT': 0, 'REST': 0}
        
        self.logger.info(f"Starting calibration: {total_trials} trials")
        
        try:
            for idx, (cue_type, _) in enumerate(trial_sequence, 1):
                # Record trial
                epoch = self.record_trial(cue_type, idx, total_trials)
                
                if epoch is not None:
                    # Store epoch
                    if cue_type == 'LEFT':
                        self.epochs_left.append(epoch)
                    elif cue_type == 'RIGHT':
                        self.epochs_right.append(epoch)
                    else:  # REST
                        self.epochs_rest.append(epoch)
                    
                    successful_trials[cue_type] += 1
                
                # Inter-trial rest (except for last trial)
                if idx < total_trials:
                    self.inter_trial_rest()
                
                # Progress update every 10 trials
                if idx % 10 == 0:
                    self.logger.info(
                        f"Progress: {idx}/{total_trials} trials | "
                        f"LEFT: {successful_trials['LEFT']}, "
                        f"RIGHT: {successful_trials['RIGHT']}, "
                        f"REST: {successful_trials['REST']}"
                    )
        
        except KeyboardInterrupt:
            self.logger.warning("\n⚠️  Calibration interrupted by user")
            return False
        
        finally:
            self.adapter.stop_streaming()
        
        # Show final statistics
        print("\n" + "="*70)
        print("CALIBRATION COMPLETE")
        print("="*70)
        print(f"\nSuccessful trials collected:")
        print(f"  • LEFT HAND:  {successful_trials['LEFT']}/{n_left}")
        print(f"  • RIGHT HAND: {successful_trials['RIGHT']}/{n_right}")
        print(f"  • REST:       {successful_trials['REST']}/{n_rest}")
        print("="*70 + "\n")
        
        # Check minimum requirements
        min_required = min(n_left, n_right) * 0.8  # At least 80% success rate
        if (successful_trials['LEFT'] < min_required or 
            successful_trials['RIGHT'] < min_required):
            self.logger.error("❌ Insufficient data collected for fine-tuning")
            return False
        
        return True
    
    def save_data(self) -> str:
        """
        Save calibration data to .npz file.
        
        Returns:
            Path to saved file
        """
        # Convert to numpy arrays
        epochs_left = np.array(self.epochs_left)   # (n_left, 3, 1000)
        epochs_right = np.array(self.epochs_right) # (n_right, 3, 1000)
        epochs_rest = np.array(self.epochs_rest)   # (n_rest, 3, 1000)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.user_id}_calibration_{timestamp}.npz"
        filepath = self.save_dir / filename
        
        # Save data
        np.savez_compressed(
            filepath,
            epochs_left=epochs_left,
            epochs_right=epochs_right,
            epochs_rest=epochs_rest,
            sampling_rate=OUTPUT_SAMPLING_RATE,
            channel_names=['C3', 'Cz', 'C4'],
            user_id=self.user_id,
            timestamp=timestamp,
            n_left=len(epochs_left),
            n_right=len(epochs_right),
            n_rest=len(epochs_rest)
        )
        
        self.logger.info(f"✅ Calibration data saved: {filepath}")
        
        # Print summary
        print(f"\n📁 Saved to: {filepath}")
        print(f"   Size: {filepath.stat().st_size / 1024:.1f} KB")
        print(f"   LEFT trials:  {len(epochs_left)}")
        print(f"   RIGHT trials: {len(epochs_right)}")
        print(f"   REST trials:  {len(epochs_rest)}")
        
        return str(filepath)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Record personal EEG calibration data for BCI fine-tuning'
    )
    parser.add_argument(
        '--user-id',
        type=str,
        required=True,
        help='Unique identifier for user (e.g., john_doe)'
    )
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Use simulator instead of real NPG Lite device'
    )
    parser.add_argument(
        '--left',
        type=int,
        default=100,
        help='Number of LEFT hand trials (default: 100)'
    )
    parser.add_argument(
        '--right',
        type=int,
        default=100,
        help='Number of RIGHT hand trials (default: 100)'
    )
    parser.add_argument(
        '--rest',
        type=int,
        default=50,
        help='Number of REST trials (default: 50)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='calibration_data',
        help='Directory to save calibration data'
    )
    
    args = parser.parse_args()
    
    # Create recorder
    recorder = CalibrationRecorder(
        user_id=args.user_id,
        simulate=args.simulate,
        save_dir=args.save_dir
    )
    
    # Connect to device
    if not recorder.connect():
        print("❌ Failed to connect to EEG device")
        return 1
    
    # Record session
    success = recorder.record_session(
        n_left=args.left,
        n_right=args.right,
        n_rest=args.rest
    )
    
    if not success:
        print("❌ Calibration session failed")
        return 1
    
    # Save data
    filepath = recorder.save_data()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print(f"\n1. Fine-tune the model:")
    print(f"   python fine_tune_personal_model.py --calibration-file={filepath}")
    print(f"\n2. Validate the personalized system:")
    print(f"   python validate_3state_system.py --user-id={args.user_id}")
    print(f"\n3. Use personalized model in real-time:")
    print(f"   python npg_realtime_bci.py --user-id={args.user_id}")
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
