"""
Realtime Channel Display for NPG Lite

Display channel 5 (or specified channel) in realtime with amplitude information.
Shows live signal data with rolling plot and statistics.

Usage:
    python display_channel_realtime.py --channel 5
    python display_channel_realtime.py --channel 2 --simulate
"""

import numpy as np
import argparse
import time
import sys
from pathlib import Path
from collections import deque

try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for better interactivity
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.ion()  # Enable interactive mode
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from npg_lite_adapter import NPGLiteAdapter, NPGLiteSimulator


class ChannelDisplay:
    """Realtime channel display with rolling buffer and statistics."""
    
    def __init__(self, channel: int = 2, window_duration: float = 4.0, 
                 sampling_rate: int = 500, simulate: bool = False,
                 use_plot: bool = True):
        """
        Initialize channel display.
        
        Args:
            channel: Channel index to display (0-based)
            window_duration: Rolling window duration in seconds
            sampling_rate: Sampling rate in Hz
            simulate: Use simulator instead of real device
            use_plot: Use matplotlib for plotting (vs console only)
        """
        self.channel = channel
        self.window_duration = window_duration
        self.sampling_rate = sampling_rate
        self.window_size = int(window_duration * sampling_rate)
        self.use_plot = use_plot and MATPLOTLIB_AVAILABLE
        
        # Connect to device
        self.adapter = NPGLiteSimulator(sampling_rate=sampling_rate) if simulate else \
                      NPGLiteAdapter(sampling_rate=sampling_rate)
        
        # Data buffer
        self.data_buffer = deque(maxlen=self.window_size)
        self.timestamps = deque(maxlen=self.window_size)
        
        # Statistics
        self.start_time = None
        self.sample_count = 0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
        # Setup plot if enabled
        if self.use_plot:
            self._setup_plot()
        else:
            print("Matplotlib not available. Using console mode.")
    
    def _setup_plot(self):
        """Setup matplotlib figure and axis."""
        self.fig, (self.ax_waveform, self.ax_spectrum) = plt.subplots(
            2, 1, figsize=(12, 8)
        )
        self.fig.suptitle(f"Channel {self.channel} - Realtime EEG Signal (Press Ctrl+C to stop)", fontsize=14)
        
        # Waveform plot
        self.ax_waveform.set_ylabel("Amplitude (μV)", fontsize=12)
        self.ax_waveform.set_ylim(-200, 200)
        self.line_waveform, = self.ax_waveform.plot([], [], lw=0.8, color='blue')
        self.ax_waveform.grid(True, alpha=0.3)
        
        # Spectrum plot
        self.ax_spectrum.set_title("Power Spectrum", fontsize=12)
        self.ax_spectrum.set_xlabel("Frequency (Hz)", fontsize=12)
        self.ax_spectrum.set_ylabel("Power (dB)", fontsize=12)
        self.line_spectrum, = self.ax_spectrum.plot([], [], lw=1, color='green')
        self.ax_spectrum.set_xlim(0, 50)
        self.ax_spectrum.set_ylim(-40, 20)
        self.ax_spectrum.grid(True, alpha=0.3)
        
        # Status text
        self.status_text = self.ax_waveform.text(
            0.02, 0.95, "", transform=self.ax_waveform.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    def _compute_spectrum(self, data: np.ndarray) -> tuple:
        """Compute power spectrum."""
        from scipy import signal
        
        # Compute FFT
        fft_vals = np.fft.fft(data - np.mean(data))
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.fftfreq(len(data), 1/self.sampling_rate)
        
        # Take positive frequencies only
        positive_freq_idx = freqs > 0
        freqs = freqs[positive_freq_idx]
        power = power[positive_freq_idx]
        
        # Convert to dB
        power_db = 10 * np.log10(power + 1e-10)
        
        return freqs, power_db
    
    def _update_plot(self):
        """Update matplotlib plot."""
        if not self.use_plot or not self.data_buffer:
            return
        
        data = np.array(self.data_buffer)
        times = np.array(self.timestamps)
        
        try:
            # Update waveform
            if len(times) > 0:
                time_offset = times[0]
                relative_times = times - time_offset
                self.line_waveform.set_data(relative_times, data)
                
                # Dynamic y-axis
                if len(data) > 0:
                    data_max = np.max(np.abs(data)) * 1.2
                    self.ax_waveform.set_ylim(-data_max, data_max)
                
                self.ax_waveform.set_xlim(relative_times[0], relative_times[-1] if len(relative_times) > 1 else relative_times[0] + 1)
            
            # Update spectrum every 10 updates (for performance)
            if self.sample_count % 10 == 0:
                if len(data) > 64:
                    freqs, power_db = self._compute_spectrum(data)
                    # Limit to 50 Hz for visualization
                    freq_limit = freqs <= 50
                    self.line_spectrum.set_data(freqs[freq_limit], power_db[freq_limit])
            
            # Update status text
            if self.data_buffer:
                stats = self._compute_stats()
                status = (
                    f"Channel: {self.channel}  |  "
                    f"Samples: {self.sample_count:6d}\n"
                    f"Mean: {stats['mean']:8.2f} μV  |  "
                    f"Std:  {stats['std']:8.2f} μV\n"
                    f"Min:  {stats['min']:8.2f} μV  |  "
                    f"Max:  {stats['max']:8.2f} μV\n"
                    f"Range: {stats['range']:8.2f} μV"
                )
                self.status_text.set_text(status)
            
            # Update canvas with small pause for event processing
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        
        except Exception as e:
            print(f"\n⚠️  Plot update error: {e}")
    
    def _compute_stats(self) -> dict:
        """Compute signal statistics."""
        if not self.data_buffer:
            return {
                'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'range': 0
            }
        
        data = np.array(self.data_buffer)
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data)
        }
    
    def _display_console(self):
        """Display stats in console mode."""
        stats = self._compute_stats()
        print(
            f"\r[Channel {self.channel}] "
            f"Samples: {self.sample_count:6d} | "
            f"Mean: {stats['mean']:8.2f} | "
            f"Std:  {stats['std']:8.2f} | "
            f"Min:  {stats['min']:8.2f} | "
            f"Max:  {stats['max']:8.2f} | "
            f"Range: {stats['range']:8.2f}",
            end='', flush=True
        )
    
    def run(self):
        """Start realtime display."""
        print("="*80)
        print(f"Channel {self.channel} Realtime Display")
        print("="*80)
        
        # Connect
        if not self.adapter.connect():
            print("❌ Failed to connect to device")
            return False
        
        print("✅ Connected to device")
        print(f"Sampling rate: {self.sampling_rate} Hz")
        print(f"Display window: {self.window_duration}s ({self.window_size} samples)")
        print("Press Ctrl+C to stop\n")
        
        # Start streaming
        self.adapter.start_streaming()
        self.start_time = time.time()
        
        try:
            chunk_size = 128
            update_counter = 0
            
            while True:
                # Get data
                data = self.adapter.get_latest_data(n_samples=chunk_size)
                
                if data is None or len(data) == 0:
                    # Still pump events even with no data
                    if self.use_plot:
                        plt.pause(0.01)
                    else:
                        time.sleep(0.01)
                    continue
                
                # Extract channel
                if data.shape[1] <= self.channel:
                    print(f"\n❌ Channel {self.channel} not available (only {data.shape[1]} channels)")
                    break
                
                channel_data = data[:, self.channel]
                
                # Add to buffer
                current_time = time.time()
                for i, sample in enumerate(channel_data):
                    self.data_buffer.append(sample)
                    self.timestamps.append(current_time + i / self.sampling_rate)
                    self.sample_count += 1
                    self.min_val = min(self.min_val, sample)
                    self.max_val = max(self.max_val, sample)
                
                # Update display (every 5 chunks = 640 samples ≈ 1.3 seconds)
                update_counter += 1
                if update_counter >= 5:
                    if self.use_plot:
                        self._update_plot()
                    else:
                        self._display_console()
                    update_counter = 0
                
                # Pump matplotlib event loop
                if self.use_plot:
                    plt.pause(0.001)
        
        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("Stopped by user")
        
        except Exception as e:
            print(f"\n❌ Error: {e}", exc_info=True)
        
        finally:
            self.adapter.stop_streaming()
            
            # Final statistics
            print("\n" + "="*80)
            print("Final Statistics:")
            print(f"  Total samples: {self.sample_count}")
            print(f"  Duration: {time.time() - self.start_time:.1f}s")
            print(f"  Overall min: {self.min_val:.2f} μV")
            print(f"  Overall max: {self.max_val:.2f} μV")
            print(f"  Overall range: {self.max_val - self.min_val:.2f} μV")
            print("="*80)
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Realtime Channel Display for NPG Lite')
    parser.add_argument('--channel', type=int, default=2,
                       help='Channel index to display (0=C3, 1=Cz, 2=C4, default: 2)')
    parser.add_argument('--window', type=float, default=4.0,
                       help='Display window duration in seconds (default: 4.0)')
    parser.add_argument('--rate', type=int, default=500,
                       help='Sampling rate in Hz (default: 500)')
    parser.add_argument('--simulate', action='store_true',
                       help='Use simulator instead of real device')
    parser.add_argument('--no-plot', action='store_true',
                       help='Use console mode instead of matplotlib plot')
    
    args = parser.parse_args()
    
    # Create and run display
    display = ChannelDisplay(
        channel=args.channel,
        window_duration=args.window,
        sampling_rate=args.rate,
        simulate=args.simulate,
        use_plot=not args.no_plot
    )
    
    display.run()


if __name__ == "__main__":
    main()
