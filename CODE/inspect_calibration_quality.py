#!/usr/bin/env python3
"""
Inspect calibration data quality for personal model fine-tuning.

Analyzes:
- Class separability (LEFT vs RIGHT)
- Signal quality metrics
- Class balance
- Feature distributions
- Recommendations for fine-tuning

Usage:
    python inspect_calibration_quality.py --calibration-file=calibration_data/alan_*.npz
"""

import numpy as np
import argparse
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix
from scipy import signal

# Add CODE directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bci4_2b_loader_v2 import BCI4_2B_Loader


class CalibrationQualityInspector:
    """Inspect personal calibration data quality."""
    
    def __init__(self, calibration_file: str):
        self.calibration_file = Path(calibration_file)
        if not self.calibration_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    
    def load_data(self):
        """Load calibration data."""
        print(f"Loading calibration data: {self.calibration_file.name}")
        
        loader = BCI4_2B_Loader("config_2b.yaml")
        X, y, rest_epochs = loader.load_personal_calibration(str(self.calibration_file))
        
        print(f"✅ Loaded {len(X)} training epochs")
        print(f"   LEFT (0):  {np.sum(y == 0)} epochs")
        print(f"   RIGHT (1): {np.sum(y == 1)} epochs")
        print(f"   REST: {len(rest_epochs)} epochs\n")
        
        return X, y, rest_epochs
    
    def analyze_signal_quality(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Analyze basic signal quality metrics."""
        print("="*70)
        print("SIGNAL QUALITY ANALYSIS")
        print("="*70)
        
        metrics = {}
        
        # Per-class statistics
        for class_idx, class_name in [(0, "LEFT"), (1, "RIGHT")]:
            class_mask = y == class_idx
            X_class = X[class_mask]
            
            mean_std = np.mean([np.std(trial) for trial in X_class])
            mean_peak = np.mean([np.max(np.abs(trial)) for trial in X_class])
            mean_power = np.mean([np.mean(trial**2) for trial in X_class])
            
            print(f"\n{class_name} ({np.sum(class_mask)} trials):")
            print(f"  Mean channel std: {mean_std:.2f} µV")
            print(f"  Mean peak amplitude: {mean_peak:.2f} µV")
            print(f"  Mean power: {mean_power:.2f}")
            
            metrics[class_name] = {
                'std': mean_std,
                'peak': mean_peak,
                'power': mean_power,
                'n_trials': int(np.sum(class_mask))
            }
        
        return metrics
    
    def analyze_class_separability(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Analyze class separability using various metrics."""
        print("\n" + "="*70)
        print("CLASS SEPARABILITY ANALYSIS")
        print("="*70)
        
        separability = {}
        
        # Feature extraction: mean power in frequency bands
        left_features = []
        right_features = []
        
        for trial, label in zip(X, y):
            features = self._extract_features(trial)
            if label == 0:
                left_features.append(features)
            else:
                right_features.append(features)
        
        left_features = np.array(left_features)
        right_features = np.array(right_features)
        
        # Effect size (Cohen's d) for each feature
        effect_sizes = []
        for feat_idx in range(left_features.shape[1]):
            left_feat = left_features[:, feat_idx]
            right_feat = right_features[:, feat_idx]
            
            mean_diff = np.mean(left_feat) - np.mean(right_feat)
            pooled_std = np.sqrt((np.std(left_feat)**2 + np.std(right_feat)**2) / 2)
            
            if pooled_std > 0:
                cohens_d = mean_diff / pooled_std
            else:
                cohens_d = 0
            
            effect_sizes.append(abs(cohens_d))
        
        mean_effect_size = np.mean(effect_sizes)
        
        print(f"\nCohen's d (effect size) for each feature:")
        for i, d in enumerate(effect_sizes):
            print(f"  Feature {i}: {d:.4f}")
        
        print(f"\nMean effect size: {mean_effect_size:.4f}")
        
        if mean_effect_size < 0.2:
            print("  ⚠️  Small effect size - classes may not be well separated")
        elif mean_effect_size < 0.5:
            print("  ⚠️  Moderate effect size - separation present but weak")
        else:
            print("  ✅ Good effect size - classes are well separated")
        
        separability['mean_effect_size'] = float(mean_effect_size)
        separability['per_feature_effect_size'] = [float(d) for d in effect_sizes]
        
        return separability
    
    def analyze_class_balance(self, y: np.ndarray) -> dict:
        """Analyze class balance."""
        print("\n" + "="*70)
        print("CLASS BALANCE ANALYSIS")
        print("="*70)
        
        left_count = np.sum(y == 0)
        right_count = np.sum(y == 1)
        total = len(y)
        
        left_pct = left_count / total * 100
        right_pct = right_count / total * 100
        
        print(f"\nClass distribution:")
        print(f"  LEFT:  {left_count:3d} ({left_pct:5.1f}%)")
        print(f"  RIGHT: {right_count:3d} ({right_pct:5.1f}%)")
        
        imbalance_ratio = max(left_count, right_count) / min(left_count, right_count)
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio < 1.2:
            print("  ✅ Well balanced")
        elif imbalance_ratio < 1.5:
            print("  ⚠️  Slightly imbalanced")
        else:
            print("  ⚠️  Significantly imbalanced - consider class weights")
        
        return {
            'left_count': int(left_count),
            'right_count': int(right_count),
            'imbalance_ratio': float(imbalance_ratio),
            'left_pct': float(left_pct),
            'right_pct': float(right_pct)
        }
    
    def analyze_rest_trials(self, rest_epochs: np.ndarray) -> dict:
        """Analyze REST trial quality."""
        print("\n" + "="*70)
        print("REST TRIAL ANALYSIS")
        print("="*70)
        
        rest_power = np.mean([np.mean(trial**2) for trial in rest_epochs])
        rest_std = np.mean([np.std(trial) for trial in rest_epochs])
        
        print(f"\nREST trials: {len(rest_epochs)} epochs")
        print(f"  Mean power: {rest_power:.2f}")
        print(f"  Mean std: {rest_std:.2f} µV")
        print(f"  ✅ Sufficient for neutral threshold calibration" if len(rest_epochs) >= 30 else "  ⚠️  May need more REST trials")
        
        return {
            'n_rest': int(len(rest_epochs)),
            'power': float(rest_power),
            'std': float(rest_std)
        }
    
    def _extract_features(self, trial: np.ndarray) -> np.ndarray:
        """Extract frequency-domain features from a trial."""
        features = []
        
        # Per-channel features
        for ch_idx in range(trial.shape[0]):
            signal_ch = trial[ch_idx]
            
            # Skip if signal too short for Welch
            if len(signal_ch) < 256:
                # Use simple std and peak instead
                features.extend([np.std(signal_ch), np.max(np.abs(signal_ch))])
            else:
                # Compute power spectrum
                freqs, psd = signal.welch(signal_ch, fs=250, nperseg=min(256, len(signal_ch)))
                
                # Mu band (8-12 Hz)
                mu_mask = (freqs >= 8) & (freqs <= 12)
                mu_power = np.sum(psd[mu_mask]) if np.any(mu_mask) else 0
                
                # Beta band (15-30 Hz)
                beta_mask = (freqs >= 15) & (freqs <= 30)
                beta_power = np.sum(psd[beta_mask]) if np.any(beta_mask) else 0
                
                features.extend([mu_power, beta_power])
        
        return np.array(features)
    
    def generate_recommendations(self, metrics: dict, separability: dict, 
                                balance: dict, rest_info: dict) -> list:
        """Generate recommendations based on analysis."""
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70 + "\n")
        
        recommendations = []
        
        # Signal quality
        if any(m['std'] > 100 for m in metrics.values()):
            recommendations.append("✅ Signal amplitude is good (> 100 µV standard deviation)")
        else:
            recommendations.append("⚠️  Signal amplitude is low - check electrode contact and impedance")
        
        # Class separability
        if separability['mean_effect_size'] > 0.5:
            recommendations.append("✅ Classes are well separated - fine-tuning should work well")
        else:
            recommendations.append("⚠️  Classes have weak separation - may need more data or better electrode placement")
        
        # Class balance
        if balance['imbalance_ratio'] < 1.5:
            recommendations.append("✅ Classes are well balanced - no special handling needed")
        else:
            recommendations.append(f"⚠️  Classes are imbalanced ({balance['imbalance_ratio']:.1f}:1) - use class weights during fine-tuning")
        
        # REST trials
        if rest_info['n_rest'] >= 50:
            recommendations.append("✅ Plenty of REST trials for threshold calibration")
        else:
            recommendations.append(f"⚠️  Only {rest_info['n_rest']} REST trials - collect more for robust threshold")
        
        # Filter recommendation
        recommendations.append("\n💡 Fine-tuning recommendation:")
        if separability['mean_effect_size'] > 0.3:
            recommendations.append("   • Start with --no-filter (skip Laplacian) for better performance")
            recommendations.append("   • Use --strategy=full to train all layers")
        else:
            recommendations.append("   • Try both --filter and --no-filter to see which works better")
            recommendations.append("   • Consider collecting more calibration data")
        
        for rec in recommendations:
            print(rec)
        
        return recommendations
    
    def run(self):
        """Run complete inspection."""
        print("\n" + "="*70)
        print("CALIBRATION DATA QUALITY INSPECTION")
        print("="*70 + "\n")
        
        # Load data
        X, y, rest_epochs = self.load_data()
        
        # Analyses
        signal_metrics = self.analyze_signal_quality(X, y)
        separability = self.analyze_class_separability(X, y)
        balance = self.analyze_class_balance(y)
        rest_info = self.analyze_rest_trials(rest_epochs)
        
        # Recommendations
        recommendations = self.generate_recommendations(
            signal_metrics, separability, balance, rest_info
        )
        
        print("\n" + "="*70 + "\n")
        
        return {
            'signal_metrics': signal_metrics,
            'separability': separability,
            'balance': balance,
            'rest_info': rest_info,
            'recommendations': recommendations
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Inspect personal calibration data quality'
    )
    parser.add_argument(
        '--calibration-file',
        type=str,
        required=True,
        help='Path to calibration NPZ file'
    )
    
    args = parser.parse_args()
    
    try:
        inspector = CalibrationQualityInspector(args.calibration_file)
        results = inspector.run()
        return 0
    except Exception as e:
        print(f"\n❌ Inspection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
