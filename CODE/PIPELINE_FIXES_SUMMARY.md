# BCI Pipeline Fixes Summary

## 3-Channel, 2-Class Motor Imagery - Signal Acquisition to Model Output

This document summarizes all fixes applied to improve accuracy and model performance for the NPG Lite BCI system.

---

## 1. NPG Preprocessor (`npg_preprocessor.py`)

### Issues Fixed:

#### 1.1 Resampling Without Anti-Aliasing (CRITICAL)
**Problem**: Original code resampled from 500 Hz to 250 Hz without anti-aliasing filter, causing aliasing artifacts.

**Fix**: Added proper anti-aliasing low-pass filter at 100 Hz (80% of target Nyquist) BEFORE downsampling.

```python
# Anti-aliasing cutoff at 80% of target Nyquist (125 Hz * 0.8 = 100 Hz)
nyquist_target = output_rate / 2
antialias_cutoff = nyquist_target * 0.8
self.antialias_sos = signal.butter(8, antialias_cutoff, btype='low', 
                                   fs=input_rate, output='sos')
```

#### 1.2 Inappropriate Spatial Filter (CRITICAL)
**Problem**: Common Average Reference (CAR) was used for 3-channel setup, which is mathematically inappropriate because:
- The "common average" of 3 motor cortex channels is heavily biased
- Removes meaningful motor imagery signal, not just noise
- Reduces C3-C4 differential needed for left/right discrimination

**Fix**: Replaced CAR with Small Laplacian spatial filter:

```python
def apply_small_laplacian(self, data):
    """Small Laplacian for C3, Cz, C4 arrangement."""
    c3, cz, c4 = data[:, 0], data[:, 1], data[:, 2]
    
    laplacian = np.zeros_like(data)
    laplacian[:, 0] = c3 - 0.5 * cz  # C3 referenced to Cz
    laplacian[:, 1] = cz - 0.25 * (c3 + c4)  # Cz referenced to motor average
    laplacian[:, 2] = c4 - 0.5 * cz  # C4 referenced to Cz
    
    return laplacian
```

#### 1.3 Non-Stateful Filters for Real-Time Streaming
**Problem**: Using `filtfilt` (zero-phase filtering) in real-time mode causes edge effects at epoch boundaries and isn't suitable for streaming.

**Fix**: Implemented stateful SOS filters (`RealtimeBandpassFilter`, `RealtimeNotchFilter`) that maintain state between calls:

```python
class RealtimeBandpassFilter:
    def __init__(self, low_freq, high_freq, fs, order=4, n_channels=3):
        self.sos = signal.butter(order, [low_freq, high_freq], 
                                  btype='band', fs=fs, output='sos')
        self.zi = [signal.sosfilt_zi(self.sos) for _ in range(n_channels)]
    
    def filter(self, data, stateful=True):
        # Maintains state for continuous streaming
        for ch in range(data.shape[1]):
            filtered[:, ch], self.zi[ch] = signal.sosfilt(
                self.sos, data[:, ch], zi=self.zi[ch]
            )
```

#### 1.4 Slow Normalization Adaptation
**Problem**: EMA alpha of 0.99 made normalization adapt too slowly to new subjects.

**Fix**: Reduced alpha to 0.95 for faster cross-subject adaptation.

---

## 2. NPG Inference Engine (`npg_inference.py`)

### Issues Fixed:

#### 2.1 Uncalibrated Softmax Confidence
**Problem**: Raw softmax outputs are NOT calibrated probabilities. A 0.7 softmax output doesn't mean 70% accuracy.

**Fix**: Implemented temperature scaling for calibrated confidence:

```python
class CalibratedConfidence:
    def __init__(self, temperature=1.5):
        self.temperature = temperature
    
    def calibrate(self, logits):
        scaled = logits / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        return exp_scaled / exp_scaled.sum()
    
    def compute_uncertainty(self, probabilities):
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(probabilities))
        return entropy / max_entropy  # 0 = certain, 1 = uncertain
```

#### 2.2 Simple Majority Voting in Smoothing
**Problem**: Original smoothing used simple majority vote, ignoring confidence levels.

**Fix**: Implemented confidence-weighted voting:

```python
def predict_smoothed(self, preprocessed_data):
    # Weight predictions by confidence squared (emphasize high confidence)
    for idx, conf in self.prediction_buffer:
        weight = conf ** 2
        class_weights[idx] += weight
```

#### 2.3 Missing Uncertainty Tracking
**Problem**: No way to know how certain the model is about predictions.

**Fix**: Added entropy-based uncertainty estimation and tracking.

---

## 3. NPG Real-Time BCI (`npg_realtime_bci.py`)

### Issues Fixed:

#### 3.1 Inconsistent Window Buffer Size
**Problem**: Window buffer was set to 1024 samples assuming 256 Hz, but actual input is 500 Hz.

**Fix**: Created global constants and corrected buffer size:

```python
INPUT_SAMPLING_RATE = 500   # NPG Lite via Chords-Python
OUTPUT_SAMPLING_RATE = 250  # Model trained at this rate
EPOCH_DURATION = 4.0        # 4 seconds per classification

INPUT_EPOCH_SAMPLES = int(EPOCH_DURATION * INPUT_SAMPLING_RATE)  # 2000
```

#### 3.2 Missing Filter State Reset
**Problem**: Filter states weren't reset between sessions, causing initial transients.

**Fix**: Added `reset_filters()` call after warmup completes.

---

## 4. NPG Lite Adapter (`npg_lite_adapter.py`)

### Issues Fixed:

#### 4.1 Unrealistic Signal Quality Thresholds
**Problem**: Quality thresholds were set for raw ADC values, not microvolts.

**Fix**: Updated thresholds for typical EEG in µV:

```python
def check_signal_quality(self):
    # Good EEG: std 10-100 µV, max abs < 200 µV
    if std < 0.5:
        score = 0.1  # Flat signal - disconnected
    elif 10 <= std <= 100 and max_abs < 300:
        score = 1.0  # Good signal
```

#### 4.2 Simulator Signal Amplitudes
**Problem**: Simulated signals had unrealistic amplitudes.

**Fix**: Adjusted to typical EEG amplitudes in microvolts:

```python
self.alpha_amplitude = 20.0   # Alpha (8-13 Hz): ~20 µV
self.mu_amplitude = 15.0      # Mu (8-12 Hz): ~15 µV
self.beta_amplitude = 10.0    # Beta (13-30 Hz): ~10 µV
self.noise_amplitude = 5.0    # Background noise: ~5 µV
```

---

## Pipeline Flow (After Fixes)

```
NPG Lite (500 Hz, 3ch: C3, Cz, C4)
    │
    ▼
Anti-aliasing LPF @ 100 Hz (NEW)
    │
    ▼
Resample 500 → 250 Hz (1:2)
    │
    ▼
Notch Filter @ 50 Hz (stateful)
    │
    ▼
Small Laplacian Spatial Filter (FIXED: was CAR)
    │
    ▼
Bandpass 8-30 Hz (stateful)
    │
    ▼
Z-score Normalize (faster α=0.95)
    │
    ▼
Reshape: (1000, 3) → (1, 3, 1000, 1)
    │
    ▼
EEGNet Model
    │
    ▼
Temperature-Scaled Confidence (NEW)
    │
    ▼
Confidence-Weighted Smoothing (FIXED)
    │
    ▼
Leaky Accumulator
    │
    ▼
Command Output: LEFT_HAND / RIGHT_HAND / UNCERTAIN
```

---

## Configuration Summary

| Parameter | Value | Reason |
|-----------|-------|--------|
| Input rate | 500 Hz | NPG Lite via Chords-Python |
| Output rate | 250 Hz | Model trained at this rate |
| Epoch duration | 4.0 s | Standard for motor imagery |
| Input samples | 2000 | 4s × 500 Hz |
| Output samples | 1000 | 4s × 250 Hz |
| Anti-alias cutoff | 100 Hz | 80% of target Nyquist |
| Bandpass | 8-30 Hz | Mu (8-13) + Beta (13-30) |
| Spatial filter | Small Laplacian | Appropriate for 3 channels |
| Temperature | 1.5 | Calibrated confidence |
| Norm alpha | 0.95 | Faster adaptation |
| Confidence threshold | 0.65 | Balance sensitivity/specificity |
| Accumulator threshold | 2.0 | Requires sustained evidence |
| Accumulator decay | 0.15 | Prevents rapid switching |

---

## Expected Impact on Accuracy

1. **Anti-aliasing**: Prevents aliasing artifacts that corrupt mu/beta band power
2. **Small Laplacian**: Preserves C3-C4 differential crucial for left/right discrimination
3. **Stateful filtering**: Eliminates edge effects in continuous streaming
4. **Calibrated confidence**: Provides meaningful certainty estimates
5. **Confidence-weighted smoothing**: Emphasizes high-quality predictions
6. **Faster normalization**: Better cross-subject generalization

These fixes address fundamental signal processing issues that were likely degrading classification accuracy by corrupting the frequency-domain features (mu/beta ERD/ERS) that the model relies on.
