"""
NPG Lite Electrode Placement Checker
Real-time signal viewer to verify electrode contact and placement
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

from npg_lite_adapter import NPGLiteAdapter, NPGLiteDirectSerial

class ElectrodePlacementChecker:
    """Real-time EEG signal viewer for checking electrode placement."""
    
    def __init__(self, adapter, window_seconds=5, sampling_rate=500):
        """
        Initialize checker.
        
        Args:
            adapter: NPGLiteAdapter or NPGLiteDirectSerial
            window_seconds: Time window to display (seconds)
            sampling_rate: Sampling rate (Hz)
        """
        self.adapter = adapter
        self.window_seconds = window_seconds
        self.sampling_rate = sampling_rate
        self.window_samples = int(window_seconds * sampling_rate)
        
        # Data buffers for each channel
        self.channel_names = ['C3', 'Cz', 'C4']
        self.n_channels = len(self.channel_names)
        self.buffers = [deque(maxlen=self.window_samples) for _ in range(self.n_channels)]
        
        # Initialize with zeros
        for buffer in self.buffers:
            for _ in range(self.window_samples):
                buffer.append(0.0)
        
        # Time axis
        self.time_axis = np.linspace(-window_seconds, 0, self.window_samples)
        
        # Setup plot
        self.fig, self.axes = plt.subplots(self.n_channels, 1, figsize=(12, 8))
        self.fig.suptitle('NPG Lite - Electrode Placement Check', fontsize=16, fontweight='bold')
        
        self.lines = []
        self.quality_texts = []
        
        for i, (ax, name) in enumerate(zip(self.axes, self.channel_names)):
            line, = ax.plot(self.time_axis, np.zeros(self.window_samples), 'b-', linewidth=0.8)
            self.lines.append(line)
            
            ax.set_ylabel(f'{name}\n(µV)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-window_seconds, 0)
            ax.set_ylim(-200, 200)  # Typical EEG range
            
            # Quality indicator text
            quality_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                                  verticalalignment='top', fontsize=9,
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            self.quality_texts.append(quality_text)
        
        self.axes[-1].set_xlabel('Time (seconds)', fontsize=10)
        
        plt.tight_layout()
        
    def assess_signal_quality(self, data):
        """
        Assess signal quality for each channel.
        
        Args:
            data: Array of recent samples (last 250 samples = 0.5 seconds)
            
        Returns:
            tuple: (quality_label, color)
        """
        if len(data) < 10:
            return "NO DATA", "red"
        
        # Calculate metrics
        amplitude = np.ptp(data)  # Peak-to-peak amplitude
        std = np.std(data)
        mean = np.mean(data)
        
        # Quality assessment
        if amplitude < 5:
            return f"POOR - Too Flat\nAmp: {amplitude:.1f}µV", "red"
        elif amplitude > 500:
            return f"POOR - Too Noisy\nAmp: {amplitude:.1f}µV", "red"
        elif std < 2:
            return f"POOR - No Activity\nStd: {std:.1f}µV", "orange"
        elif std > 200:
            return f"POOR - High Noise\nStd: {std:.1f}µV", "orange"
        else:
            return f"GOOD ✓\nAmp: {amplitude:.1f}µV\nStd: {std:.1f}µV", "green"
    
    def update(self, frame):
        """Update plot with new data."""
        # Get latest data
        data = self.adapter.get_latest_data(n_samples=50)  # Get 50 samples (~0.1s)
        
        if data is not None and len(data) > 0:
            # Update buffers
            for ch in range(self.n_channels):
                for sample in data[:, ch]:
                    self.buffers[ch].append(sample)
        
        # Update plots
        for i in range(self.n_channels):
            # Convert buffer to array
            y_data = np.array(list(self.buffers[i]))
            self.lines[i].set_ydata(y_data)
            
            # Auto-scale y-axis based on data
            if len(y_data) > 0:
                data_min, data_max = np.min(y_data), np.max(y_data)
                margin = (data_max - data_min) * 0.2 + 10
                self.axes[i].set_ylim(data_min - margin, data_max + margin)
            
            # Assess quality on recent data (last 0.5 seconds)
            recent_data = y_data[-250:] if len(y_data) >= 250 else y_data
            quality_label, color = self.assess_signal_quality(recent_data)
            self.quality_texts[i].set_text(quality_label)
            self.quality_texts[i].get_bbox_patch().set_facecolor(color)
            self.quality_texts[i].get_bbox_patch().set_alpha(0.7)
        
        return self.lines + self.quality_texts
    
    def run(self):
        """Start real-time monitoring."""
        print("=" * 70)
        print("🧠 NPG Lite Electrode Placement Checker")
        print("=" * 70)
        print("\nChecking electrode placement for:")
        print("  • C3  - Left motor cortex (left hand movement)")
        print("  • Cz  - Central reference")
        print("  • C4  - Right motor cortex (right hand movement)")
        print("\n📊 Signal Quality Indicators:")
        print("  🟢 GREEN  - Good contact, proper amplitude")
        print("  🟠 ORANGE - Marginal quality, may need adjustment")
        print("  🔴 RED    - Poor contact or noise, adjust electrodes")
        print("\n✅ What to look for:")
        print("  • Smooth, continuous waveforms")
        print("  • Amplitude 10-200 µV (typical EEG range)")
        print("  • All channels showing activity")
        print("  • Minimal high-frequency noise")
        print("\n❌ Problems to fix:")
        print("  • Flat lines (no contact)")
        print("  • Very noisy signals (poor contact)")
        print("  • One channel very different (electrode issue)")
        print("\n⚠️  Close the plot window to stop")
        print("=" * 70)
        
        # Start animation
        anim = FuncAnimation(self.fig, self.update, interval=100, blit=True, cache_frame_data=False)
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='NPG Lite Electrode Placement Checker')
    parser.add_argument('--direct', action='store_true',
                       help='Direct serial connection (no LSL)')
    parser.add_argument('--port', type=str, default='COM9',
                       help='Serial port for direct connection (default: COM9)')
    parser.add_argument('--baudrate', type=int, default=230400,
                       help='Serial baud rate (default: 230400)')
    parser.add_argument('--window', type=float, default=5.0,
                       help='Time window to display in seconds (default: 5)')
    
    args = parser.parse_args()
    
    # Create adapter
    if args.direct:
        print(f"🔌 Using direct serial connection: {args.port} @ {args.baudrate} baud")
        adapter = NPGLiteDirectSerial(port=args.port, baudrate=args.baudrate)
        
        if not adapter.connect():
            print(f"❌ Failed to connect to {args.port}")
            print("\nTroubleshooting:")
            print("  1. Run: python test_npg_connection.py")
            print("  2. Check NPG Lite is powered on")
            print("  3. Try different port: --port COM6")
            print("  4. Try different baud rate: --baudrate 115200")
            sys.exit(1)
    else:
        print("🌐 Using LSL connection (requires Chords-Python)")
        adapter = NPGLiteAdapter()
        
        if not adapter.connect():
            print("❌ Failed to find LSL stream")
            print("\nMake sure Chords-Python is running:")
            print("  Terminal 1: chordspy")
            print("  Terminal 2: python check_electrodes.py")
            print("\nOr use direct mode:")
            print("  python check_electrodes.py --direct --port COM9")
            sys.exit(1)
    
    # Start streaming
    adapter.start_streaming()
    
    # Give some time for data to accumulate
    import time
    print("\n⏳ Collecting initial data...")
    time.sleep(2)
    
    # Create and run checker
    checker = ElectrodePlacementChecker(adapter, window_seconds=args.window)
    
    try:
        checker.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")
    finally:
        adapter.stop_streaming()
        adapter.disconnect()
        print("\n✅ Disconnected")


if __name__ == "__main__":
    main()
