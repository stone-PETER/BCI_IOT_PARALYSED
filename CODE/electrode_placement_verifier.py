"""
Electrode Placement Verification Tool for NPG Lite
Helps verify proper positioning of C3, Cz, C4 electrodes

Features:
- Real-time signal quality monitoring
- Alpha rhythm detection (8-12 Hz)
- Hemisphere balance check (C3 vs C4)
- Visual and text feedback on electrode contact
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from scipy.fft import rfft, rfftfreq
import logging
import time
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from npg_lite_adapter import NPGLiteAdapter, NPGLiteSimulator


class ElectrodePlacementVerifier:
    """
    Real-time electrode placement verification tool.
    Displays signal quality, spectral analysis, and placement guidance.
    """
    
    def __init__(self, npg_adapter: NPGLiteAdapter, use_simulator: bool = False):
        """
        Initialize verifier.
        
        Args:
            npg_adapter: NPG Lite adapter instance
            use_simulator: Use simulator instead of real hardware
        """
        self.npg = npg_adapter
        self.use_simulator = use_simulator
        self.sampling_rate = npg_adapter.sampling_rate
        
        # Channel indices (assuming: C3=0, Cz=1, C4=2)
        self.target_channels = {
            'C3': 0,
            'Cz': 1,
            'C4': 2
        }
        
        # Analysis parameters
        self.window_duration = 2.0  # seconds
        self.window_samples = int(self.window_duration * self.sampling_rate)
        
        # Quality thresholds
        self.snr_threshold = 5.0  # dB
        self.alpha_power_threshold = 0.3  # normalized
        self.balance_threshold = 0.3  # 30% difference allowed
        
        # Results
        self.results = {
            'C3': {'quality': 0.0, 'snr': 0.0, 'alpha_power': 0.0, 'status': 'UNKNOWN'},
            'Cz': {'quality': 0.0, 'snr': 0.0, 'alpha_power': 0.0, 'status': 'UNKNOWN'},
            'C4': {'quality': 0.0, 'snr': 0.0, 'alpha_power': 0.0, 'status': 'UNKNOWN'},
            'balance': 0.0,
            'overall': 'NOT_READY'
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def calculate_snr(self, signal_data: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio.
        
        Args:
            signal_data: 1D array of signal
        
        Returns:
            SNR in dB
        """
        # Signal power (8-30 Hz - EEG band)
        b, a = signal.butter(4, [8, 30], btype='band', fs=self.sampling_rate)
        signal_filtered = signal.filtfilt(b, a, signal_data)
        signal_power = np.var(signal_filtered)
        
        # Noise power (50-100 Hz - high frequency noise)
        b_noise, a_noise = signal.butter(4, [50, 100], btype='band', fs=self.sampling_rate)
        noise_filtered = signal.filtfilt(b_noise, a_noise, signal_data)
        noise_power = np.var(noise_filtered)
        
        # SNR in dB
        if noise_power < 1e-10:
            return 100.0  # Very high SNR
        
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return snr_db
    
    def calculate_alpha_power(self, signal_data: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate alpha band (8-12 Hz) power.
        
        Args:
            signal_data: 1D array of signal
        
        Returns:
            Tuple of (normalized_alpha_power, frequencies, psd)
        """
        # Calculate power spectral density
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, nperseg=min(512, len(signal_data)))
        
        # Alpha band power (8-12 Hz)
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        alpha_power = np.sum(psd[alpha_mask])
        
        # Total power (1-40 Hz)
        total_mask = (freqs >= 1) & (freqs <= 40)
        total_power = np.sum(psd[total_mask])
        
        # Normalized alpha power
        alpha_normalized = alpha_power / (total_power + 1e-10)
        
        return alpha_normalized, freqs, psd
    
    def check_hemisphere_balance(self, c3_data: np.ndarray, c4_data: np.ndarray) -> float:
        """
        Check balance between C3 and C4 (left/right hemisphere).
        
        Args:
            c3_data: C3 channel data
            c4_data: C4 channel data
        
        Returns:
            Balance score (0 = perfect balance, 1 = completely imbalanced)
        """
        c3_power = np.var(c3_data)
        c4_power = np.var(c4_data)
        
        # Avoid division by zero
        if c3_power + c4_power < 1e-10:
            return 1.0
        
        # Balance: abs difference / sum
        balance = abs(c3_power - c4_power) / (c3_power + c4_power + 1e-10)
        
        return balance
    
    def analyze_channels(self) -> Dict:
        """
        Analyze current signal quality for all channels.
        
        Returns:
            Dictionary with analysis results
        """
        # Get latest data
        data = self.npg.get_latest_data(self.window_samples)
        
        if data is None:
            self.logger.warning("Not enough data for analysis")
            return self.results
        
        # Analyze each target channel
        for ch_name, ch_idx in self.target_channels.items():
            if ch_idx >= data.shape[1]:
                continue
            
            ch_data = data[:, ch_idx]
            
            # Calculate metrics
            snr = self.calculate_snr(ch_data)
            alpha_power, freqs, psd = self.calculate_alpha_power(ch_data)
            
            # Overall quality score (0-1)
            snr_score = min(1.0, max(0.0, snr / 20.0))  # SNR: 0-20 dB maps to 0-1
            alpha_score = min(1.0, alpha_power / 0.5)  # Alpha: 0-0.5 maps to 0-1
            quality = (snr_score + alpha_score) / 2
            
            # Status
            if quality > 0.7 and snr > self.snr_threshold:
                status = 'GOOD'
            elif quality > 0.4:
                status = 'ADJUST'
            else:
                status = 'POOR'
            
            # Update results
            self.results[ch_name] = {
                'quality': quality,
                'snr': snr,
                'alpha_power': alpha_power,
                'status': status,
                'std': np.std(ch_data),
                'mean': np.mean(ch_data)
            }
        
        # Check hemisphere balance (C3 vs C4)
        if 'C3' in self.target_channels and 'C4' in self.target_channels:
            c3_idx = self.target_channels['C3']
            c4_idx = self.target_channels['C4']
            
            if c3_idx < data.shape[1] and c4_idx < data.shape[1]:
                balance = self.check_hemisphere_balance(
                    data[:, c3_idx],
                    data[:, c4_idx]
                )
                self.results['balance'] = balance
        
        # Overall status
        all_good = all(
            self.results[ch]['status'] == 'GOOD' 
            for ch in self.target_channels.keys()
        )
        
        if all_good and self.results['balance'] < self.balance_threshold:
            self.results['overall'] = 'READY'
        elif any(self.results[ch]['status'] == 'POOR' for ch in self.target_channels.keys()):
            self.results['overall'] = 'POOR'
        else:
            self.results['overall'] = 'ADJUST'
        
        return self.results
    
    def print_status(self):
        """Print current status to console."""
        print("\n" + "="*70)
        print("📊 ELECTRODE PLACEMENT STATUS")
        print("="*70)
        
        for ch_name in ['C3', 'Cz', 'C4']:
            if ch_name in self.results:
                result = self.results[ch_name]
                
                # Status emoji
                if result['status'] == 'GOOD':
                    emoji = "✅"
                elif result['status'] == 'ADJUST':
                    emoji = "⚠️ "
                else:
                    emoji = "❌"
                
                print(f"\n{emoji} {ch_name}:")
                print(f"   Status: {result['status']}")
                print(f"   Quality: {result['quality']:.2f}")
                print(f"   SNR: {result['snr']:.1f} dB")
                print(f"   Alpha Power: {result['alpha_power']:.2f}")
                print(f"   Signal Std: {result['std']:.2f} µV")
        
        # Balance
        print(f"\n🔄 Hemisphere Balance (C3 vs C4):")
        balance = self.results['balance']
        balance_status = "✅ GOOD" if balance < self.balance_threshold else "⚠️  ADJUST"
        print(f"   {balance_status} - Difference: {balance:.1%}")
        
        # Overall
        print(f"\n🎯 OVERALL STATUS: {self.results['overall']}")
        
        if self.results['overall'] == 'READY':
            print("\n✅ All electrodes positioned correctly! Ready for BCI.")
        elif self.results['overall'] == 'ADJUST':
            print("\n⚠️  Adjust electrode positioning:")
            for ch_name in ['C3', 'Cz', 'C4']:
                if self.results[ch_name]['status'] != 'GOOD':
                    print(f"   - {ch_name}: {self._get_adjustment_tip(ch_name)}")
        else:
            print("\n❌ Poor signal quality. Check all electrodes:")
            print("   1. Ensure electrodes are in contact with skin")
            print("   2. Add conductive gel if needed")
            print("   3. Adjust headband tension")
            print("   4. Clean skin with alcohol wipe before placement")
        
        print("="*70)
    
    def _get_adjustment_tip(self, channel: str) -> str:
        """Get adjustment tip for specific channel."""
        tips = {
            'C3': "Move headband slightly right or adjust left electrode contact",
            'Cz': "Center the headband front-to-back, ensure top electrode contact",
            'C4': "Move headband slightly left or adjust right electrode contact"
        }
        return tips.get(channel, "Check electrode contact and positioning")
    
    def run_verification(self, duration: int = 30):
        """
        Run verification for specified duration.
        
        Args:
            duration: How long to run verification (seconds)
        """
        print("\n" + "="*70)
        print("🔍 STARTING ELECTRODE PLACEMENT VERIFICATION")
        print("="*70)
        print(f"\nRunning for {duration} seconds...")
        print("Keep still and relax during verification.")
        print("\nPress Ctrl+C to stop early.\n")
        
        try:
            start_time = time.time()
            last_print = 0
            
            while time.time() - start_time < duration:
                # Wait for buffer to fill
                buffer_size = self.npg.get_buffer_size()
                if buffer_size < self.window_samples:
                    remaining = self.window_samples - buffer_size
                    print(f"⏳ Buffering... {remaining} samples needed", end='\r')
                    time.sleep(0.5)
                    continue
                
                # Analyze
                self.analyze_channels()
                
                # Print status every 5 seconds
                elapsed = time.time() - start_time
                if elapsed - last_print >= 5:
                    self.print_status()
                    last_print = elapsed
                    
                    # Show progress
                    remaining = duration - elapsed
                    print(f"\n⏱️  Time remaining: {remaining:.0f} seconds")
                
                time.sleep(1)
            
            # Final status
            print("\n\n" + "="*70)
            print("🏁 VERIFICATION COMPLETE")
            print("="*70)
            self.print_status()
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Verification stopped by user")
            self.print_status()
    
    def run_live_monitor(self):
        """
        Run live monitoring with matplotlib visualization.
        Shows real-time waveforms and power spectra.
        """
        print("\n🎬 Starting live visualization...")
        print("Close the window to stop monitoring.\n")
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('NPG Lite Electrode Placement Verification', fontsize=16, fontweight='bold')
        
        # Initialize plots
        time_axes = axes[:, 0]
        freq_axes = axes[:, 1]
        
        channel_names = ['C3', 'Cz', 'C4']
        colors = ['blue', 'green', 'red']
        
        lines_time = []
        lines_freq = []
        status_texts = []
        
        for i, (ch_name, color) in enumerate(zip(channel_names, colors)):
            # Time domain
            line_t, = time_axes[i].plot([], [], color=color, linewidth=0.5)
            time_axes[i].set_title(f'{ch_name} - Time Domain', fontweight='bold')
            time_axes[i].set_xlabel('Time (s)')
            time_axes[i].set_ylabel('Amplitude (µV)')
            time_axes[i].set_xlim(0, 2)
            time_axes[i].set_ylim(-50, 50)
            time_axes[i].grid(True, alpha=0.3)
            lines_time.append(line_t)
            
            # Frequency domain
            line_f, = freq_axes[i].plot([], [], color=color, linewidth=1.5)
            freq_axes[i].set_title(f'{ch_name} - Frequency Domain', fontweight='bold')
            freq_axes[i].set_xlabel('Frequency (Hz)')
            freq_axes[i].set_ylabel('Power (dB)')
            freq_axes[i].set_xlim(0, 40)
            freq_axes[i].set_ylim(-20, 20)
            freq_axes[i].grid(True, alpha=0.3)
            freq_axes[i].axvspan(8, 12, alpha=0.2, color='yellow', label='Alpha (8-12 Hz)')
            freq_axes[i].legend(loc='upper right')
            lines_freq.append(line_f)
            
            # Status text
            status_text = time_axes[i].text(0.02, 0.95, '', transform=time_axes[i].transAxes,
                                           verticalalignment='top', fontsize=10,
                                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            status_texts.append(status_text)
        
        plt.tight_layout()
        
        def update(frame):
            """Update animation frame."""
            # Get latest data
            data = self.npg.get_latest_data(self.window_samples)
            
            if data is None:
                return lines_time + lines_freq + status_texts
            
            # Analyze
            self.analyze_channels()
            
            # Update plots
            t = np.arange(len(data)) / self.sampling_rate
            
            for i, ch_name in enumerate(channel_names):
                ch_idx = self.target_channels[ch_name]
                
                if ch_idx >= data.shape[1]:
                    continue
                
                ch_data = data[:, ch_idx]
                
                # Time domain
                lines_time[i].set_data(t, ch_data)
                
                # Frequency domain
                alpha_power, freqs, psd = self.calculate_alpha_power(ch_data)
                psd_db = 10 * np.log10(psd + 1e-10)
                lines_freq[i].set_data(freqs, psd_db)
                
                # Status text
                result = self.results[ch_name]
                status_color = {'GOOD': 'green', 'ADJUST': 'orange', 'POOR': 'red'}
                status_text = (f"Status: {result['status']}\n"
                             f"Quality: {result['quality']:.2f}\n"
                             f"SNR: {result['snr']:.1f} dB\n"
                             f"Alpha: {result['alpha_power']:.2f}")
                
                status_texts[i].set_text(status_text)
                status_texts[i].set_bbox(dict(boxstyle='round', 
                                             facecolor=status_color.get(result['status'], 'gray'),
                                             alpha=0.5))
            
            return lines_time + lines_freq + status_texts
        
        # Animate
        anim = FuncAnimation(fig, update, interval=500, blit=True)
        plt.show()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NPG Lite Electrode Placement Verifier')
    parser.add_argument('--simulate', action='store_true', help='Use simulator instead of real device')
    parser.add_argument('--duration', type=int, default=30, help='Verification duration (seconds)')
    parser.add_argument('--live', action='store_true', help='Show live visualization')
    parser.add_argument('--port', type=str, help='Serial port (e.g., COM3, /dev/ttyUSB0). Auto-detect if not specified')
    parser.add_argument('--baudrate', type=int, default=115200, help='Serial baud rate')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("🧠 NPG LITE ELECTRODE PLACEMENT VERIFIER")
    print("="*70)
    
    # Create adapter
    if args.simulate:
        print("\n⚠️  Running in SIMULATION mode")
        npg = NPGLiteSimulator()
    else:
        print("\n📡 Connecting to NPG Lite...")
        npg = NPGLiteAdapter(port=args.port, baudrate=args.baudrate)
    
    # Connect
    if not npg.connect():
        print("\n❌ Failed to connect to NPG Lite")
        print("   - Check device is powered on and connected via USB")
        print("   - Try specifying port: --port COM3")
        print("   - Try using --simulate flag for testing")
        return
    
    # Start streaming
    npg.start_streaming()
    
    # Wait for initial buffer
    print("\n⏳ Buffering initial data...")
    time.sleep(3)
    
    # Create verifier
    verifier = ElectrodePlacementVerifier(npg, use_simulator=args.simulate)
    
    try:
        if args.live:
            # Live visualization
            verifier.run_live_monitor()
        else:
            # Text-based verification
            verifier.run_verification(duration=args.duration)
    
    finally:
        # Cleanup
        npg.stop_streaming()
        npg.disconnect()
        print("\n✅ Verification complete. Disconnected from NPG Lite.\n")


if __name__ == "__main__":
    main()
