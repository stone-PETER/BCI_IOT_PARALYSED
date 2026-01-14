"""
Channel Importance Analysis for BCI Motor Imagery
Analyzes which EEG channels are most significant for motor imagery classification

Methods:
1. Variance-based analysis (inter-class variance, signal-to-noise ratio)
2. Common Spatial Patterns (CSP) analysis
3. Model gradient-based importance
4. Correlation with motor cortex channels
"""

import numpy as np
import yaml
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.linalg import eigh
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")


class ChannelImportanceAnalyzer:
    """
    Analyzer for determining channel importance in motor imagery BCI.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the analyzer."""
        # Get script directory
        script_dir = Path(__file__).parent.resolve()
        config_path = script_dir / config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # BCI IV 2a channel names (22 EEG channels)
        self.channel_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2',
            'POz'
        ]
        
        # Motor cortex channel indices (central electrodes)
        self.motor_cortex_channels = {
            'C3': 7,   # Left motor cortex
            'Cz': 9,   # Central
            'C4': 11,  # Right motor cortex
            'CP3': 13, # Left centro-parietal
            'CP4': 17  # Right centro-parietal
        }
        
        # Class names
        self.class_names = ['Left Hand', 'Right Hand', 'Foot', 'Tongue']
        
        self.results = {}
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load preprocessed data for analysis.
        
        Returns:
            epochs: EEG epochs (trials, samples, channels)
            labels: Class labels (0-3)
        """
        self.logger.info("Loading data for channel importance analysis...")
        
        from bci4_2a_loader import BCI4_2A_Loader
        
        loader = BCI4_2A_Loader()
        
        # Load all subjects' training data
        all_epochs = []
        all_labels = []
        
        for subject_num in range(1, 10):  # A01 to A09
            subject_id = f"A{subject_num:02d}"
            
            try:
                epochs, labels = loader.load_subject(subject_id, 'T')
                all_epochs.append(epochs)
                all_labels.append(labels)
                self.logger.info(f"Loaded {subject_id}: {len(epochs)} epochs")
            except Exception as e:
                self.logger.warning(f"Failed to load {subject_id}: {e}")
                continue
        
        # Combine all data
        combined_epochs = np.vstack(all_epochs)
        combined_labels = np.concatenate(all_labels)
        
        self.logger.info(f"Total data: {combined_epochs.shape[0]} epochs")
        self.logger.info(f"Epochs shape: {combined_epochs.shape}")
        
        return combined_epochs, combined_labels
    
    def analyze_variance_based_importance(self, epochs: np.ndarray, 
                                         labels: np.ndarray) -> Dict:
        """
        Analyze channel importance based on variance metrics.
        
        Args:
            epochs: EEG epochs (trials, samples, channels)
            labels: Class labels
            
        Returns:
            Dictionary with variance-based importance metrics
        """
        self.logger.info("Computing variance-based channel importance...")
        
        n_channels = epochs.shape[2]
        
        # 1. Inter-class variance (how much each channel varies between classes)
        inter_class_variance = np.zeros(n_channels)
        
        for ch in range(n_channels):
            class_means = []
            for class_idx in range(4):
                class_mask = labels == class_idx
                class_mean = np.mean(epochs[class_mask, :, ch])
                class_means.append(class_mean)
            
            inter_class_variance[ch] = np.var(class_means)
        
        # 2. Signal-to-Noise Ratio per channel
        signal_to_noise = np.zeros(n_channels)
        
        for ch in range(n_channels):
            # Signal power (variance across trials)
            signal_power = np.mean(np.var(epochs[:, :, ch], axis=1))
            
            # Noise estimate (high-frequency variations)
            noise_estimate = np.mean(np.abs(np.diff(epochs[:, :, ch], axis=1)))
            
            signal_to_noise[ch] = signal_power / (noise_estimate + 1e-10)
        
        # 3. Class separability (within-class vs between-class variance)
        class_separability = np.zeros(n_channels)
        
        for ch in range(n_channels):
            within_class_var = 0
            between_class_var = 0
            
            overall_mean = np.mean(epochs[:, :, ch])
            
            for class_idx in range(4):
                class_mask = labels == class_idx
                class_data = epochs[class_mask, :, ch]
                class_mean = np.mean(class_data)
                
                # Within-class variance
                within_class_var += np.sum((class_data - class_mean) ** 2)
                
                # Between-class variance
                n_class = np.sum(class_mask)
                between_class_var += n_class * (class_mean - overall_mean) ** 2
            
            class_separability[ch] = between_class_var / (within_class_var + 1e-10)
        
        # 4. Motor cortex correlation
        motor_cortex_correlation = np.zeros(n_channels)
        
        # Average signal from key motor cortex channels
        motor_channels_idx = list(self.motor_cortex_channels.values())
        motor_signal = np.mean(epochs[:, :, motor_channels_idx], axis=2)
        
        for ch in range(n_channels):
            correlation = np.corrcoef(
                epochs[:, :, ch].flatten(),
                motor_signal.flatten()
            )[0, 1]
            motor_cortex_correlation[ch] = abs(correlation)
        
        results = {
            'inter_class_variance': inter_class_variance,
            'signal_to_noise': signal_to_noise,
            'class_separability': class_separability,
            'motor_cortex_correlation': motor_cortex_correlation
        }
        
        self.logger.info("Variance-based analysis completed")
        
        return results
    
    def analyze_power_spectral_density(self, epochs: np.ndarray, 
                                      labels: np.ndarray) -> Dict:
        """
        Analyze channel importance based on power spectral density in motor imagery bands.
        
        Args:
            epochs: EEG epochs
            labels: Class labels
            
        Returns:
            PSD-based importance metrics
        """
        self.logger.info("Computing PSD-based channel importance...")
        
        n_channels = epochs.shape[2]
        fs = 250  # Sampling rate
        
        # Frequency bands of interest for motor imagery
        mu_band = (8, 13)    # Mu rhythm
        beta_band = (13, 30)  # Beta rhythm
        
        mu_power_diff = np.zeros(n_channels)
        beta_power_diff = np.zeros(n_channels)
        
        for ch in range(n_channels):
            # Calculate PSD for left hand vs right hand (most discriminative)
            left_hand = epochs[labels == 0, :, ch]
            right_hand = epochs[labels == 1, :, ch]
            
            # Average PSD across trials
            freq_left, psd_left = welch(left_hand, fs=fs, nperseg=256, axis=1)
            freq_right, psd_right = welch(right_hand, fs=fs, nperseg=256, axis=1)
            
            psd_left_avg = np.mean(psd_left, axis=0)
            psd_right_avg = np.mean(psd_right, axis=0)
            
            # Mu band power difference
            mu_idx = (freq_left >= mu_band[0]) & (freq_left <= mu_band[1])
            mu_power_diff[ch] = abs(np.mean(psd_left_avg[mu_idx]) - np.mean(psd_right_avg[mu_idx]))
            
            # Beta band power difference
            beta_idx = (freq_left >= beta_band[0]) & (freq_left <= beta_band[1])
            beta_power_diff[ch] = abs(np.mean(psd_left_avg[beta_idx]) - np.mean(psd_right_avg[beta_idx]))
        
        results = {
            'mu_power_diff': mu_power_diff,
            'beta_power_diff': beta_power_diff,
            'combined_band_power': mu_power_diff + beta_power_diff
        }
        
        self.logger.info("PSD-based analysis completed")
        
        return results
    
    def analyze_csp_importance(self, epochs: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Analyze channel importance using Common Spatial Patterns (CSP).
        
        Args:
            epochs: EEG epochs
            labels: Class labels
            
        Returns:
            CSP-based importance scores
        """
        self.logger.info("Computing CSP-based channel importance...")
        
        n_channels = epochs.shape[2]
        
        # Calculate covariance matrices for each class
        cov_matrices = []
        for class_idx in range(4):
            class_mask = labels == class_idx
            class_epochs = epochs[class_mask]
            
            # Average covariance across trials
            cov = np.zeros((n_channels, n_channels))
            for trial in class_epochs:
                trial_centered = trial - trial.mean(axis=0)
                cov += np.dot(trial_centered.T, trial_centered) / trial.shape[0]
            cov /= len(class_epochs)
            cov_matrices.append(cov)
        
        # CSP analysis for each class pair
        channel_scores = np.zeros(n_channels)
        
        for i in range(4):
            for j in range(i + 1, 4):
                # Composite covariance
                composite_cov = cov_matrices[i] + cov_matrices[j]
                
                # Eigenvalue decomposition
                eigenvalues, eigenvectors = eigh(cov_matrices[i], composite_cov)
                
                # Channel importance from spatial filters
                # Take the magnitude of contributions from top and bottom eigenvectors
                top_filters = eigenvectors[:, -3:]  # Top 3
                bottom_filters = eigenvectors[:, :3]  # Bottom 3
                
                for ch in range(n_channels):
                    channel_scores[ch] += (np.abs(top_filters[ch, :]).sum() + 
                                          np.abs(bottom_filters[ch, :]).sum())
        
        # Normalize
        channel_scores /= channel_scores.max()
        
        results = {
            'csp_importance': channel_scores
        }
        
        self.logger.info("CSP-based analysis completed")
        
        return results
    
    def analyze_model_importance(self, model_path: str, epochs: np.ndarray) -> Dict:
        """
        Analyze channel importance using trained model gradients.
        
        Args:
            model_path: Path to trained model
            epochs: Sample epochs for gradient analysis
            
        Returns:
            Model-based importance scores
        """
        self.logger.info("Computing model-based channel importance...")
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            
            # Load model
            model = load_model(model_path)
            n_channels = epochs.shape[2]
            
            # Prepare data (take subset for efficiency)
            sample_epochs = epochs[:100]  # Use first 100 samples
            X = sample_epochs.transpose(0, 2, 1)[..., np.newaxis]
            X_tensor = tf.Variable(X, dtype=tf.float32)
            
            # Calculate gradients
            channel_gradients = np.zeros(n_channels)
            
            with tf.GradientTape() as tape:
                predictions = model(X_tensor)
                loss = tf.reduce_mean(predictions)
            
            # Get gradients with respect to input
            grads = tape.gradient(loss, X_tensor)
            
            if grads is not None:
                # Average gradient magnitude per channel
                grads_np = grads.numpy()
                for ch in range(n_channels):
                    channel_gradients[ch] = np.mean(np.abs(grads_np[:, ch, :, :]))
            
            # Normalize
            if channel_gradients.max() > 0:
                channel_gradients /= channel_gradients.max()
            
            results = {
                'gradient_importance': channel_gradients
            }
            
            self.logger.info("Model-based analysis completed")
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Model-based analysis failed: {e}")
            return {'gradient_importance': np.zeros(epochs.shape[2])}
    
    def compute_combined_importance(self, all_results: Dict) -> Dict:
        """
        Compute combined importance scores from all methods.
        
        Args:
            all_results: Dictionary containing results from all analysis methods
            
        Returns:
            Combined importance scores and rankings
        """
        self.logger.info("Computing combined importance scores...")
        
        n_channels = len(self.channel_names)
        
        # Normalize all metrics to [0, 1]
        def normalize(x):
            x_min, x_max = x.min(), x.max()
            if x_max - x_min > 0:
                return (x - x_min) / (x_max - x_min)
            return x
        
        # Extract and normalize metrics
        variance_results = all_results['variance_based']
        psd_results = all_results['psd_based']
        csp_results = all_results['csp_based']
        
        normalized_scores = {
            'inter_class_var': normalize(variance_results['inter_class_variance']),
            'snr': normalize(variance_results['signal_to_noise']),
            'separability': normalize(variance_results['class_separability']),
            'motor_corr': normalize(variance_results['motor_cortex_correlation']),
            'mu_power': normalize(psd_results['mu_power_diff']),
            'beta_power': normalize(psd_results['beta_power_diff']),
            'csp': normalize(csp_results['csp_importance'])
        }
        
        # Weighted combination (higher weights for more reliable metrics)
        weights = {
            'inter_class_var': 1.5,
            'snr': 1.0,
            'separability': 2.0,
            'motor_corr': 1.5,
            'mu_power': 2.0,
            'beta_power': 2.0,
            'csp': 2.5
        }
        
        combined_score = np.zeros(n_channels)
        total_weight = sum(weights.values())
        
        for metric, weight in weights.items():
            combined_score += weight * normalized_scores[metric]
        
        combined_score /= total_weight
        
        # Rank channels
        ranked_indices = np.argsort(combined_score)[::-1]
        
        results = {
            'combined_score': combined_score,
            'ranked_indices': ranked_indices,
            'top_10_channels': ranked_indices[:10],
            'top_10_names': [self.channel_names[i] for i in ranked_indices[:10]],
            'top_10_scores': combined_score[ranked_indices[:10]],
            'normalized_scores': normalized_scores
        }
        
        self.logger.info("Combined importance computed")
        self.logger.info(f"Top 10 channels: {results['top_10_names']}")
        
        return results
    
    def visualize_results(self, all_results: Dict, combined_results: Dict):
        """
        Create comprehensive visualizations of channel importance.
        
        Args:
            all_results: All analysis results
            combined_results: Combined importance scores
        """
        self.logger.info("Creating visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        n_channels = len(self.channel_names)
        x_pos = np.arange(n_channels)
        
        # 1. Inter-class variance
        ax1 = fig.add_subplot(gs[0, 0])
        variance_results = all_results['variance_based']
        ax1.bar(x_pos, variance_results['inter_class_variance'], alpha=0.7)
        ax1.set_title('Inter-Class Variance', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Variance')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.channel_names, rotation=90, fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Signal-to-Noise Ratio
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(x_pos, variance_results['signal_to_noise'], alpha=0.7, color='orange')
        ax2.set_title('Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('SNR')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.channel_names, rotation=90, fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Class Separability
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(x_pos, variance_results['class_separability'], alpha=0.7, color='green')
        ax3.set_title('Class Separability', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Channel')
        ax3.set_ylabel('Separability')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(self.channel_names, rotation=90, fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Mu band power difference
        ax4 = fig.add_subplot(gs[1, 0])
        psd_results = all_results['psd_based']
        ax4.bar(x_pos, psd_results['mu_power_diff'], alpha=0.7, color='purple')
        ax4.set_title('Mu Band (8-13 Hz) Power Difference', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Channel')
        ax4.set_ylabel('Power Diff')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(self.channel_names, rotation=90, fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Beta band power difference
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.bar(x_pos, psd_results['beta_power_diff'], alpha=0.7, color='red')
        ax5.set_title('Beta Band (13-30 Hz) Power Difference', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Channel')
        ax5.set_ylabel('Power Diff')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(self.channel_names, rotation=90, fontsize=8)
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. CSP importance
        ax6 = fig.add_subplot(gs[1, 2])
        csp_results = all_results['csp_based']
        ax6.bar(x_pos, csp_results['csp_importance'], alpha=0.7, color='brown')
        ax6.set_title('CSP-Based Importance', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Channel')
        ax6.set_ylabel('Importance')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(self.channel_names, rotation=90, fontsize=8)
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Combined importance score
        ax7 = fig.add_subplot(gs[2, :2])
        combined_score = combined_results['combined_score']
        top_10 = combined_results['top_10_channels']
        colors = ['red' if i in top_10 else 'steelblue' for i in range(n_channels)]
        
        bars = ax7.bar(x_pos, combined_score, color=colors, alpha=0.7)
        ax7.set_title('Combined Importance Score (Top 10 in Red)', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Channel', fontsize=11)
        ax7.set_ylabel('Combined Score', fontsize=11)
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(self.channel_names, rotation=90, fontsize=9)
        ax7.grid(axis='y', alpha=0.3)
        ax7.axhline(y=np.mean(combined_score), color='black', linestyle='--', 
                    alpha=0.5, label='Mean')
        ax7.legend()
        
        # 8. Top 10 channels ranking
        ax8 = fig.add_subplot(gs[2, 2])
        top_10_names = combined_results['top_10_names']
        top_10_scores = combined_results['top_10_scores']
        
        y_pos = np.arange(len(top_10_names))
        ax8.barh(y_pos, top_10_scores, alpha=0.7, color='darkgreen')
        ax8.set_yticks(y_pos)
        ax8.set_yticklabels(top_10_names, fontsize=10)
        ax8.invert_yaxis()
        ax8.set_xlabel('Importance Score', fontsize=11)
        ax8.set_title('Top 10 Channels', fontsize=12, fontweight='bold')
        ax8.grid(axis='x', alpha=0.3)
        
        plt.suptitle('BCI Motor Imagery Channel Importance Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        output_path = 'channel_importance_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Visualization saved to: {output_path}")
        
        plt.show()
    
    def generate_report(self, all_results: Dict, combined_results: Dict) -> str:
        """
        Generate text report of channel importance analysis.
        
        Args:
            all_results: All analysis results
            combined_results: Combined importance scores
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 80)
        report.append("BCI MOTOR IMAGERY CHANNEL IMPORTANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Top 10 channels
        report.append("TOP 10 MOST IMPORTANT CHANNELS FOR MOTOR IMAGERY:")
        report.append("-" * 80)
        
        for rank, (idx, name, score) in enumerate(zip(
            combined_results['top_10_channels'],
            combined_results['top_10_names'],
            combined_results['top_10_scores']
        ), 1):
            report.append(f"{rank:2d}. {name:6s} (Index: {idx:2d}) - Score: {score:.4f}")
        
        report.append("")
        report.append("MOTOR CORTEX CHANNELS (Expected to be important):")
        report.append("-" * 80)
        
        for name, idx in self.motor_cortex_channels.items():
            score = combined_results['combined_score'][idx]
            rank = np.where(combined_results['ranked_indices'] == idx)[0][0] + 1
            status = "✓" if idx in combined_results['top_10_channels'] else " "
            report.append(f"{status} {name:6s} (Index: {idx:2d}) - Score: {score:.4f} - Rank: {rank}")
        
        report.append("")
        report.append("ANALYSIS METHOD CONTRIBUTIONS:")
        report.append("-" * 80)
        
        # Show top channel from each method
        variance_results = all_results['variance_based']
        psd_results = all_results['psd_based']
        csp_results = all_results['csp_based']
        
        methods = [
            ("Inter-Class Variance", variance_results['inter_class_variance']),
            ("Signal-to-Noise Ratio", variance_results['signal_to_noise']),
            ("Class Separability", variance_results['class_separability']),
            ("Mu Band Power", psd_results['mu_power_diff']),
            ("Beta Band Power", psd_results['beta_power_diff']),
            ("CSP Importance", csp_results['csp_importance'])
        ]
        
        for method_name, scores in methods:
            top_idx = np.argmax(scores)
            top_channel = self.channel_names[top_idx]
            top_score = scores[top_idx]
            report.append(f"{method_name:25s} - Top: {top_channel:6s} ({top_score:.4f})")
        
        report.append("")
        report.append("RECOMMENDATIONS:")
        report.append("-" * 80)
        report.append("1. For optimal motor imagery classification, focus on the top 10 channels")
        report.append("2. Central channels (C3, Cz, C4) should be highly ranked for motor imagery")
        report.append("3. Parietal channels (CP3, CP4, P3, P4) are also important for motor planning")
        report.append("4. If using fewer channels, select from the top 5-8 ranked channels")
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save to file
        output_path = 'channel_importance_report.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        self.logger.info(f"Report saved to: {output_path}")
        
        return report_text
    
    def run_complete_analysis(self, model_path: str = None):
        """
        Run complete channel importance analysis pipeline.
        
        Args:
            model_path: Path to trained model (optional, for gradient analysis)
        """
        self.logger.info("Starting complete channel importance analysis...")
        
        # 1. Load data
        epochs, labels = self.load_data()
        
        # 2. Run all analysis methods
        self.logger.info("Running variance-based analysis...")
        variance_results = self.analyze_variance_based_importance(epochs, labels)
        
        self.logger.info("Running PSD-based analysis...")
        psd_results = self.analyze_power_spectral_density(epochs, labels)
        
        self.logger.info("Running CSP-based analysis...")
        csp_results = self.analyze_csp_importance(epochs, labels)
        
        all_results = {
            'variance_based': variance_results,
            'psd_based': psd_results,
            'csp_based': csp_results
        }
        
        # 3. Model-based analysis (if model available)
        if model_path and os.path.exists(model_path):
            self.logger.info("Running model-based analysis...")
            model_results = self.analyze_model_importance(model_path, epochs)
            all_results['model_based'] = model_results
        
        # 4. Compute combined importance
        combined_results = self.compute_combined_importance(all_results)
        
        # 5. Visualize results
        self.visualize_results(all_results, combined_results)
        
        # 6. Generate report
        report = self.generate_report(all_results, combined_results)
        print("\n" + report)
        
        self.logger.info("Analysis complete!")
        
        return {
            'all_results': all_results,
            'combined_results': combined_results
        }


def main():
    """Main function to run channel importance analysis."""
    print("BCI Motor Imagery Channel Importance Analysis")
    print("=" * 80)
    
    try:
        # Initialize analyzer
        analyzer = ChannelImportanceAnalyzer()
        
        # Check for trained model
        model_path = "models/best_eegnet_4class_motor_imagery.keras"
        if not os.path.exists(model_path):
            model_path = "models/eegnet_4class_motor_imagery.keras"
        
        if os.path.exists(model_path):
            print(f"Using trained model: {model_path}")
        else:
            print("No trained model found, skipping gradient analysis")
            model_path = None
        
        # Run complete analysis
        results = analyzer.run_complete_analysis(model_path=model_path)
        
        print("\n" + "=" * 80)
        print("Analysis completed successfully!")
        print("Generated files:")
        print("  - channel_importance_analysis.png")
        print("  - channel_importance_report.txt")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
