#!/usr/bin/env python3
"""
3-State BCI System Validation

Validates personalized model with LEFT/RIGHT/NEUTRAL classification.
Tests each state with 10 trials to ensure >80% accuracy before deployment.

Usage:
    # With real NPG Lite device
    python validate_3state_system.py --user-id=john_doe
    
    # With simulator
    python validate_3state_system.py --user-id=john_doe --simulate
    
    # Adjust threshold if needed
    python validate_3state_system.py --user-id=john_doe --threshold=0.70
"""

import numpy as np
import argparse
import logging
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import Counter

# Add CODE directory to path
sys.path.insert(0, str(Path(__file__).parent))

from npg_lite_adapter import NPGLiteAdapter, NPGLiteSimulator
from npg_preprocessor import NPGPreprocessor
from npg_inference import NPGInferenceEngine
from model_factory import ModelFactory


# Validation parameters
TRIALS_PER_STATE = 10
TRIAL_DURATION = 4.0  # seconds
REST_BETWEEN_TRIALS = 2.0  # seconds
TARGET_ACCURACY = 0.80  # 80% threshold


class ThreeStateValidator:
    """Validates 3-state BCI system (LEFT/RIGHT/NEUTRAL)."""
    
    def __init__(self,
                 user_id: str,
                 simulate: bool = False,
                 threshold_override: float = None):
        """
        Initialize validator.
        
        Args:
            user_id: User ID for personalized model
            simulate: Use simulator instead of real device
            threshold_override: Override neutral threshold (for testing)
        """
        self.user_id = user_id
        self.simulate = simulate
        self.threshold_override = threshold_override
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load model info from registry
        self.model_info = ModelFactory.load_from_registry(user_id)
        
        # Initialize components
        self.adapter = None
        self.preprocessor = None
        self.inference = None
        
        # Results storage
        self.results = {
            'LEFT': {'predictions': [], 'correct': 0, 'total': 0},
            'RIGHT': {'predictions': [], 'correct': 0, 'total': 0},
            'NEUTRAL': {'predictions': [], 'correct': 0, 'total': 0}
        }
        
        self.confusion_matrix = np.zeros((3, 3), dtype=int)  # [true_state, predicted_state]
        self.state_labels = ['LEFT', 'RIGHT', 'NEUTRAL']
        
        self.logger.info(f"3-State Validator initialized for user: {user_id}")
    
    def setup(self) -> bool:
        """Setup hardware and inference engine."""
        self.logger.info("Setting up validation system...")
        
        # Initialize adapter
        if self.simulate:
            self.adapter = NPGLiteSimulator()
            self.logger.info("Using NPG Lite SIMULATOR")
        else:
            self.adapter = NPGLiteAdapter()
            self.logger.info("Using real NPG Lite device")
        
        # Connect
        if not self.adapter.connect():
            self.logger.error("Failed to connect to EEG device")
            return False
        
        # Initialize preprocessor
        self.preprocessor = NPGPreprocessor(
            input_rate=500,
            output_rate=250,
            apply_bandpass=True,
            bandpass_low=8.0,
            bandpass_high=30.0,
            apply_notch=True,
            notch_freq=50.0,
            use_car=False,
            apply_zscore=True
        )
        
        # Initialize inference engine with personalized model
        model_path = self.model_info['path']
        
        # Determine neutral threshold
        if self.threshold_override is not None:
            neutral_threshold = self.threshold_override
            self.logger.info(f"Using override neutral threshold: {neutral_threshold:.4f}")
        else:
            neutral_threshold = self.model_info.get('neutral_threshold')
            self.logger.info(f"Using calibrated neutral threshold: {neutral_threshold:.4f}")
        
        self.inference = NPGInferenceEngine(
            model_path=model_path,
            confidence_threshold=0.5,  # Lower for validation
            neutral_threshold=neutral_threshold
        )
        
        self.logger.info(f"✅ Loaded personalized model: {model_path}")
        self.logger.info(f"✅ 3-state mode enabled: LEFT/RIGHT/NEUTRAL")
        
        return True
    
    def display_instruction(self, state: str, trial_num: int, total_trials: int):
        """
        Display instruction for current trial.
        
        Args:
            state: 'LEFT', 'RIGHT', or 'NEUTRAL'
            trial_num: Current trial number
            total_trials: Total trials for this state
        """
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" * 3)
        print("="*70)
        print(f"VALIDATION BLOCK: {state}")
        print("="*70)
        
        if state == "LEFT":
            icon = "👈" * 5
            instruction = "IMAGINE LEFT HAND - Imagine clenching your LEFT fist"
        elif state == "RIGHT":
            icon = "👉" * 5
            instruction = "IMAGINE RIGHT HAND - Imagine clenching your RIGHT fist"
        else:  # NEUTRAL
            icon = "⚪" * 5
            instruction = "STAY NEUTRAL - Relax, no motor imagery"
        
        print(f"\n{icon:^70}")
        print(f"{instruction:^70}")
        print(f"\nTrial: {trial_num}/{total_trials}")
        print(f"Duration: {TRIAL_DURATION:.1f} seconds")
        print("\n" + "="*70 + "\n")
    
    def run_trial(self, expected_state: str, trial_num: int, total_trials: int) -> str:
        """
        Run a single validation trial.
        
        Args:
            expected_state: Expected state ('LEFT', 'RIGHT', 'NEUTRAL')
            trial_num: Current trial number
            total_trials: Total trials for this state
        
        Returns:
            Predicted state
        """
        # Display instruction
        self.display_instruction(expected_state, trial_num, total_trials)
        
        # Wait for data buffering
        time.sleep(0.5)
        
        # Collect predictions over trial duration
        predictions_in_trial = []
        start_time = time.time()
        
        while time.time() - start_time < TRIAL_DURATION:
            # Get data
            data = self.adapter.get_latest_data(n_samples=2000)  # 4s @ 500Hz
            
            if data is None or len(data) < 1600:  # Need at least 80% of samples
                time.sleep(0.1)
                continue
            
            # Preprocess
            preprocessed = self.preprocessor.preprocess_for_model(data[-2000:])
            
            if preprocessed is not None:
                # Predict
                class_idx, confidence, class_name = self.inference.predict(preprocessed)
                predictions_in_trial.append(class_name)
                
                # Show real-time feedback
                print(f"  Prediction: {class_name:12} (confidence: {confidence:.2%})", end='\r')
            
            time.sleep(0.2)  # Update at ~5Hz
        
        # Determine final prediction (majority vote)
        if predictions_in_trial:
            vote_counts = Counter(predictions_in_trial)
            predicted_state = vote_counts.most_common(1)[0][0]
            
            # Handle 2-state predictions
            if predicted_state not in ['LEFT_HAND', 'RIGHT_HAND', 'NEUTRAL']:
                if predicted_state == 'LEFT':
                    predicted_state = 'LEFT_HAND'
                elif predicted_state == 'RIGHT':
                    predicted_state = 'RIGHT_HAND'
        else:
            predicted_state = 'UNKNOWN'
        
        # Convert to simple labels
        if predicted_state == 'LEFT_HAND':
            final_state = 'LEFT'
        elif predicted_state == 'RIGHT_HAND':
            final_state = 'RIGHT'
        else:
            final_state = predicted_state
        
        # Show result
        is_correct = (final_state == expected_state)
        result_icon = "✅" if is_correct else "❌"
        
        print(f"\n\n  Result: {result_icon} Predicted: {final_state:8} | Expected: {expected_state:8}")
        print(f"  All predictions: {vote_counts}")
        
        return final_state
    
    def inter_trial_rest(self):
        """Rest period between trials."""
        print(f"\n  Rest for {REST_BETWEEN_TRIALS:.1f} seconds...")
        time.sleep(REST_BETWEEN_TRIALS)
    
    def validate_state(self, state: str) -> Dict:
        """
        Validate a specific state with multiple trials.
        
        Args:
            state: 'LEFT', 'RIGHT', or 'NEUTRAL'
        
        Returns:
            Results dict with accuracy and predictions
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"VALIDATING STATE: {state}")
        self.logger.info(f"{'='*70}\n")
        
        input(f"Press ENTER to start {state} validation block...")
        
        # Start streaming
        self.adapter.start_streaming()
        time.sleep(2.0)  # Warm-up
        
        predictions = []
        
        for trial in range(1, TRIALS_PER_STATE + 1):
            predicted = self.run_trial(state, trial, TRIALS_PER_STATE)
            predictions.append(predicted)
            
            # Update results
            self.results[state]['predictions'].append(predicted)
            self.results[state]['total'] += 1
            if predicted == state:
                self.results[state]['correct'] += 1
            
            # Update confusion matrix
            true_idx = self.state_labels.index(state)
            pred_idx = self.state_labels.index(predicted) if predicted in self.state_labels else true_idx
            self.confusion_matrix[true_idx, pred_idx] += 1
            
            # Inter-trial rest (except last trial)
            if trial < TRIALS_PER_STATE:
                self.inter_trial_rest()
        
        # Stop streaming
        self.adapter.stop_streaming()
        
        # Calculate accuracy
        accuracy = self.results[state]['correct'] / self.results[state]['total']
        
        # Show block results
        print(f"\n{'='*70}")
        print(f"BLOCK RESULTS: {state}")
        print(f"{'='*70}")
        print(f"Accuracy: {accuracy:.1%} ({self.results[state]['correct']}/{self.results[state]['total']})")
        print(f"Predictions: {Counter(predictions)}")
        
        if accuracy >= TARGET_ACCURACY:
            print(f"✅ PASS - Meets {TARGET_ACCURACY:.0%} threshold")
        else:
            print(f"❌ FAIL - Below {TARGET_ACCURACY:.0%} threshold")
        
        print(f"{'='*70}\n")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'target': TARGET_ACCURACY,
            'passed': accuracy >= TARGET_ACCURACY
        }
    
    def run_validation(self) -> Dict:
        """
        Run full 3-state validation.
        
        Returns:
            Validation results dict
        """
        print("\n" + "="*70)
        print("3-STATE BCI SYSTEM VALIDATION")
        print("="*70)
        print(f"\nUser: {self.user_id}")
        print(f"Trials per state: {TRIALS_PER_STATE}")
        print(f"Total trials: {TRIALS_PER_STATE * 3}")
        print(f"Target accuracy: {TARGET_ACCURACY:.0%}")
        print("\n" + "="*70 + "\n")
        
        # Setup
        if not self.setup():
            return {'success': False, 'error': 'Setup failed'}
        
        # Validate each state
        state_results = {}
        
        for state in ['LEFT', 'RIGHT', 'NEUTRAL']:
            result = self.validate_state(state)
            state_results[state] = result
        
        # Calculate overall metrics
        total_correct = sum(self.results[s]['correct'] for s in ['LEFT', 'RIGHT', 'NEUTRAL'])
        total_trials = sum(self.results[s]['total'] for s in ['LEFT', 'RIGHT', 'NEUTRAL'])
        overall_accuracy = total_correct / total_trials if total_trials > 0 else 0.0
        
        # Check neutral false trigger rate
        neutral_false_triggers = (
            self.confusion_matrix[2, 0] +  # NEUTRAL → LEFT
            self.confusion_matrix[2, 1]     # NEUTRAL → RIGHT
        )
        neutral_total = self.results['NEUTRAL']['total']
        neutral_false_rate = neutral_false_triggers / neutral_total if neutral_total > 0 else 0.0
        
        # Display final results
        self.display_final_results(overall_accuracy, neutral_false_rate, state_results)
        
        # Save results
        self.save_results(overall_accuracy, neutral_false_rate, state_results)
        
        # Overall pass/fail
        all_passed = all(r['passed'] for r in state_results.values())
        
        return {
            'success': True,
            'overall_accuracy': overall_accuracy,
            'neutral_false_rate': neutral_false_rate,
            'state_results': state_results,
            'all_passed': all_passed,
            'confusion_matrix': self.confusion_matrix.tolist()
        }
    
    def display_final_results(self, overall_accuracy: float, neutral_false_rate: float, state_results: Dict):
        """Display comprehensive validation results."""
        print("\n" + "="*70)
        print("FINAL VALIDATION RESULTS")
        print("="*70)
        
        print(f"\nOverall Accuracy: {overall_accuracy:.1%}")
        print(f"\nPer-State Accuracy:")
        for state in ['LEFT', 'RIGHT', 'NEUTRAL']:
            acc = state_results[state]['accuracy']
            passed = "✅" if state_results[state]['passed'] else "❌"
            print(f"  {state:8}: {acc:.1%} {passed}")
        
        print(f"\nConfusion Matrix:")
        print("              Predicted →")
        print("  True ↓   LEFT    RIGHT   NEUTRAL")
        for i, true_state in enumerate(self.state_labels):
            row = f"  {true_state:8}"
            for j in range(3):
                row += f"  {self.confusion_matrix[i, j]:4}"
            print(row)
        
        print(f"\nNeutral False Trigger Rate: {neutral_false_rate:.1%}")
        
        if neutral_false_rate > 0.20:
            current_threshold = self.inference.get_neutral_threshold()
            suggested = current_threshold + 0.05
            print(f"⚠️  WARNING: High false trigger rate (>{20}%)")
            print(f"   Suggestion: Increase threshold from {current_threshold:.4f} to {suggested:.4f}")
        
        # Final recommendation
        all_passed = all(r['passed'] for r in state_results.values())
        
        print("\n" + "="*70)
        if all_passed and neutral_false_rate <= 0.20:
            print("✅ SYSTEM VALIDATED - Ready for real-time use")
        else:
            print("❌ VALIDATION FAILED - System needs adjustment")
            if not all_passed:
                print("   • Some states below 80% accuracy threshold")
            if neutral_false_rate > 0.20:
                print("   • Neutral false trigger rate too high")
            print("   • Consider re-recording calibration data")
            print("   • Or adjust neutral threshold manually")
        print("="*70 + "\n")
    
    def save_results(self, overall_accuracy: float, neutral_false_rate: float, state_results: Dict):
        """Save validation results to JSON."""
        save_dir = Path(__file__).parent / "models" / "personalized"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'user_id': self.user_id,
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_info['path'],
            'neutral_threshold': self.inference.get_neutral_threshold(),
            'overall_accuracy': float(overall_accuracy),
            'neutral_false_rate': float(neutral_false_rate),
            'state_results': {
                state: {
                    'accuracy': float(result['accuracy']),
                    'passed': result['passed'],
                    'predictions': result['predictions']
                }
                for state, result in state_results.items()
            },
            'confusion_matrix': self.confusion_matrix.tolist(),
            'trials_per_state': TRIALS_PER_STATE,
            'target_accuracy': TARGET_ACCURACY
        }
        
        filepath = save_dir / f"{self.user_id}_validation.json"
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"💾 Validation results saved: {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate 3-state BCI system (LEFT/RIGHT/NEUTRAL)'
    )
    parser.add_argument(
        '--user-id',
        type=str,
        required=True,
        help='User ID for personalized model'
    )
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Use simulator instead of real NPG Lite device'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Override neutral threshold for testing'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = ThreeStateValidator(
        user_id=args.user_id,
        simulate=args.simulate,
        threshold_override=args.threshold
    )
    
    # Run validation
    try:
        results = validator.run_validation()
        
        if results['success'] and results['all_passed']:
            print("\n🎉 Validation complete! System ready for deployment.")
            print(f"\nTo use in real-time:")
            print(f"  python npg_realtime_bci.py --user-id={args.user_id}")
            return 0
        elif results['success']:
            print("\n⚠️  Validation complete but system needs improvement.")
            return 2
        else:
            print("\n❌ Validation failed.")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
