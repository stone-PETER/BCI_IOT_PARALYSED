# BCI System — Complete Guide

This document consolidates all detailed documentation for the BCI Motor Imagery Classification System. It covers hardware setup, signal processing, calibration, model configuration, and troubleshooting.

## Table of Contents

1. [NPG Lite Hardware Setup](#1-npg-lite-hardware-setup)
2. [Installation](#2-installation)
3. [Running the BCI System](#3-running-the-bci-system)
4. [Signal Processing Pipeline](#4-signal-processing-pipeline)
5. [Baseline Calibration](#5-baseline-calibration)
6. [Virtual Bipolar Filtering & Microvolt Scaling](#6-virtual-bipolar-filtering--microvolt-scaling)
7. [Leaky Accumulator](#7-leaky-accumulator)
8. [Pipeline Fixes Summary (v2.x)](#8-pipeline-fixes-summary-v2x)
9. [Model Configuration System](#9-model-configuration-system)
10. [Command Reference](#10-command-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Version History](#12-version-history)
13. [References](#13-references)

---

## 1. NPG Lite Hardware Setup

### Hardware Overview

- **Device**: NPG Lite Beast Pack by Upside Down Labs
- **Channels**: 3 (C3, Cz, C4 for motor imagery)
- **Sampling Rate**: 500 Hz
- **Resolution**: 12-bit ADC
- **Connectivity**: USB, WiFi, or Bluetooth
- **Official Library**: Chords-Python

### Why NPG Lite is Perfect for This Project

- ✅ **3 channels** exactly match our trained model (BCI Competition IV 2b)
- ✅ **C3, Cz, C4** placement ideal for left/right hand motor imagery
- ✅ **500 Hz** sampling easily downsamples to 250 Hz (1:2 ratio)
- ✅ **Wireless** options for comfortable, unrestricted use
- ✅ **Open-source** ecosystem with Chords-Python library

### Electrode Placement

**Standard 10-20 System for Motor Imagery**:

```
         Cz (center, reference)
          |
    C3 -------+------- C4
 (left)               (right)

Ground: Right mastoid (behind right ear)
```

| Electrode | Position | Function |
|---|---|---|
| C3 | 2 cm behind, 7 cm left of vertex | Left motor cortex |
| Cz | Top center of head | Central reference |
| C4 | 2 cm behind, 7 cm right of vertex | Right motor cortex |
| Ground | Right mastoid (bony bump behind right ear) | Common reference |

### Skin Preparation

1. Clean electrode sites with alcohol wipes
2. Gently abrade skin with prep gel (if provided)
3. Apply conductive gel to electrodes
4. Ensure impedance < 10 kΩ (check with NPG Lite app)

### Connecting NPG Lite

**USB** (recommended for testing):
```bash
python -m chordspy.connection --protocol usb
```

**WiFi** (wireless operation):
```bash
python -m chordspy.connection --protocol wifi --ip 192.168.1.XXX
```

**Bluetooth**:
```bash
python -m chordspy.connection --protocol ble
```

### Verify Signal Quality

```bash
python electrode_placement_verifier.py

# Good output:
# ✓ C3: 8.2 µV RMS (GOOD)
# ✓ Cz: 7.9 µV RMS (GOOD)
# ✓ C4: 8.5 µV RMS (GOOD)
```

---

## 2. Installation

```bash
cd CODE
pip install -r requirements.txt

# Verify installation
python -c "import chordspy; print('✓ Chords-Python installed')"
python -c "import pylsl; print('✓ LSL installed')"
python -c "import tensorflow; print('✓ TensorFlow installed')"
```

**Key packages**:

- `chordspy` — NPG Lite communication library (official Upside Down Labs)
- `pylsl` — Lab Streaming Layer for data streaming
- `tensorflow` — Neural network inference
- `scipy` — Signal processing

---

## 3. Running the BCI System

### Two-Terminal Workflow (with Hardware)

**Terminal 1** — Connect NPG Lite:
```bash
cd CODE
python -m chordspy.connection --protocol usb
# Expected output:
# "Connected to NPG Lite"
# "Streaming EEG data to LSL (500 Hz, 3 channels)"
# "Stream name: BioAmpDataStream"
```

**Terminal 2** — Run BCI:
```bash
cd CODE
python npg_realtime_bci.py
# Expected output:
# "[INFO] Found LSL stream: BioAmpDataStream"
# "[INFO] Connected: 3 channels @ 500 Hz"
# "[INFO] Model loaded: 73.64% accuracy"
# "[INFO] Ready for motor imagery!"
```

### Simulator Mode (no hardware)

```bash
python npg_realtime_bci.py --simulate
```

### Command-Line Options

```bash
python npg_realtime_bci.py [options]

  --simulate              Use synthetic EEG data (no hardware needed)
  --direct --port COM6    Direct serial connection to NPG Lite
  --model PATH            Custom model path
  --threshold 0.6         Classification confidence threshold
  --verbose               Enable debug logging
  --calibrate             Run 60-second baseline calibration
  --calibrate-duration 90 Custom calibration duration (seconds)
  --no-bias-correction    Disable baseline bias correction
  --no-accumulator        Disable leaky accumulator
  --acc-threshold 2.0     Accumulator trigger threshold
  --acc-decay 0.15        Accumulator decay rate per update
  --neutral-low 0.45      Neutral zone lower bound
  --neutral-high 0.55     Neutral zone upper bound
  --smoothing 8           Prediction smoothing window
  --confidence 0.65       Minimum confidence for classification
```

### Expected System Output

```
=== NPG Lite Real-Time BCI System ===

[INFO] Searching for NPG Lite LSL stream (type: EXG)...
[INFO] Found stream: BioAmpDataStream (3 channels @ 500.0 Hz)
[INFO] Connected to NPG Lite via Chords-Python
[INFO] Model loaded: best_eegnet_2class_bci2b.keras
[INFO] Model accuracy: 73.64%
[INFO] Preprocessing: 500 Hz → 250 Hz, 8-30 Hz bandpass
[INFO] Collecting baseline data (10 seconds)...

[INFO] Ready for motor imagery!

--- Classification Results ---
[2024-01-15 10:23:15] LEFT HAND  (confidence: 0.78)
[2024-01-15 10:23:23] RIGHT HAND (confidence: 0.72)
```

---

## 4. Signal Processing Pipeline

### Complete Pipeline (v2.1)

```
NPG Lite (500 Hz, 3ch: C3, Cz, C4)
    │
    ▼
Anti-aliasing LPF @ 100 Hz              ← prevents aliasing during downsample
    │
    ▼
Resample 500 → 250 Hz (1:2 ratio)
    │
    ▼
Notch Filter @ 50 Hz (Q=30)             ← removes powerline interference
    │
    ▼
Virtual Bipolar Montage                 ← matches training data format
  C3_out = C3 - Cz
  Cz_out = Cz - 0.25*(C3 + C4)
  C4_out = C4 - Cz
    │
    ▼
Bandpass Filter 8-30 Hz (4th-order Butterworth)
  mu rhythm: 8-12 Hz, beta rhythm: 13-30 Hz
    │
    ▼
Z-score Normalization (EMA, α=0.95)
    │
    ▼
Reshape: (1000, 3) → (1, 3, 1000, 1)
    │
    ▼
EEGNet Model
    │
    ▼
Temperature-Scaled Confidence (T=1.5)
    │
    ▼
Baseline Bias Correction
    │
    ▼
Confidence-Weighted Smoothing (window=8)
    │
    ▼
Leaky Accumulator
    │
    ▼
Command: LEFT_HAND / RIGHT_HAND / UNCERTAIN
```

### Configuration Summary

| Parameter | Value | Reason |
|---|---|---|
| Input rate | 500 Hz | NPG Lite via Chords-Python |
| Output rate | 250 Hz | Model trained at this rate |
| Epoch duration | 4.0 s | Standard for motor imagery |
| Input samples | 2000 | 4s × 500 Hz |
| Output samples | 1000 | 4s × 250 Hz |
| Anti-alias cutoff | 100 Hz | 80% of target Nyquist |
| Bandpass | 8-30 Hz | Mu + Beta rhythms |
| Spatial filter | Small Laplacian | Appropriate for 3 channels |
| Temperature | 1.5 | Calibrated confidence |
| Norm alpha | 0.95 | Faster cross-subject adaptation |
| Confidence threshold | 0.65 | Balance sensitivity/specificity |
| Accumulator threshold | 2.0 | Requires sustained evidence |
| Accumulator decay | 0.15 | Prevents rapid switching |

### EEGNet Architecture

```
Input: (3, 1000, 1)  — 3 channels × 1000 timepoints

Block 1: Temporal Convolution
  Conv2D: 8 filters, kernel (1, 64) — captures frequency features

Block 2: Depthwise Spatial Convolution
  DepthwiseConv2D: (3, 1) — learns spatial filters for C3, Cz, C4

Block 3: Separable Convolution
  SeparableConv2D: 16 filters, (1, 16) — higher-level temporal patterns

Block 4: Classification
  Flatten → Dense(2) → Softmax
  Output: [p_right_hand, p_left_hand]

Total Parameters: ~2,226
Trained on: BCI Competition IV 2b (9 subjects, 3 channels)
Accuracy: 73.64% on held-out test set
```

### Performance Characteristics

| Metric | Value |
|---|---|
| Data acquisition | 4000 ms (4-second epoch) |
| Preprocessing | ~50 ms |
| Model inference | ~10 ms |
| **Total latency** | **~4060 ms** |
| Training accuracy | 85% |
| Validation accuracy | 76% |
| Test accuracy | 73.64% |
| CPU usage | ~15% (single core) |
| RAM usage | ~500 MB |

---

## 5. Baseline Calibration

### Problem

The BCI model uses a 2-class design (LEFT_HAND vs RIGHT_HAND) which forces a choice even when the user is at rest. This causes:

- Predictions to hover around 50-60% confidence during rest
- Natural bias toward one hand (e.g., 58% Left / 42% Right)
- Accumulator never triggers commands properly

### Solution

Record a rest-state baseline and subtract the bias from all future predictions.

**Bias Correction Formula**:
```python
corrected = raw_probabilities - baseline_bias
corrected = np.maximum(corrected, 0.0)  # Clip negative values
corrected = corrected / corrected.sum()  # Renormalize to sum=1.0
```

### Running Calibration

**Step 1 — Calibrate (first time)**:
```bash
python npg_realtime_bci.py --simulate --calibrate
```

During calibration (60 seconds):
- Sit completely still
- Look at a fixed point
- Do NOT imagine any movement
- Relax and breathe normally

**Step 2 — Normal use (bias auto-loads)**:
```bash
python npg_realtime_bci.py --simulate
```

**Optional — Custom duration**:
```bash
python npg_realtime_bci.py --simulate --calibrate --calibrate-duration 90
```

**Optional — Disable bias correction**:
```bash
python npg_realtime_bci.py --simulate --no-bias-correction
```

### Bias File Format

`CODE/baseline_bias.json`:
```json
{
  "bias": [0.08, -0.08],
  "timestamp": "2026-02-03 14:20:00",
  "class_names": ["LEFT_HAND", "RIGHT_HAND"]
}
```

### When to Recalibrate

- ✅ Always before first use
- ✅ After moving or adjusting the headset
- ✅ After days/weeks of not using the system
- ✅ If predictions consistently favor one side
- ✅ After changing electrode gel or impedances
- ❌ Not needed between short sessions (same day)

### Before vs After Calibration

| Metric | Before | After |
|---|---|---|
| Rest prediction | 58% Left / 42% Right | 50% Left / 50% Right |
| Confidence during imagery | 52-65% | 65-85% |
| Command triggering | Rare / inconsistent | Stable and intentional |

---

## 6. Virtual Bipolar Filtering & Microvolt Scaling

### Problem: Signal Format Mismatch

The BCI Competition IV 2b dataset (used for training) used **bipolar recordings**:
- C3 referenced to Cz (C3 - Cz)
- C4 referenced to Cz (C4 - Cz)

The NPG Lite provides **monopolar recordings** (single-ended). This mismatch causes poor model performance because the signal "shape" is different from what EEGNet expects.

### Solution: Virtual Bipolar Montage

Simulate bipolar recording by computing differential signals:

```python
C3_filtered = C3 - Cz
Cz_filtered = Cz - 0.25 * (C3 + C4)   # Small Laplacian for Cz
C4_filtered = C4 - Cz
```

**Why this works**:
1. **Spatial Filtering**: Differential recordings enhance local activity and suppress common-mode noise
2. **Topographic Specificity**: C3-Cz emphasizes left motor cortex; C4-Cz emphasizes right motor cortex
3. **Training Data Match**: The model learned patterns in differential signals, not raw monopolar signals

Virtual bipolar is **enabled by default** in `NPGPreprocessor`:
```python
preprocessor = NPGPreprocessor(
    use_bipolar=True,      # Default: True
    scaling_factor=1.0     # Adjust based on your hardware
)
```

### Microvolt Scaling

The training data (BCI Competition IV 2b) had amplitudes of ±50–100 µV. NPG Lite may output in different units.

#### Determining Your Scaling Factor

**Method 1 — Use the calibration tool**:
```bash
python calibrate_scaling.py --direct --port COM6
```

**Method 2 — Check NPG Lite documentation**:
- If output is in Volts: `scaling_factor = 1e6`
- If in millivolts: `scaling_factor = 1000`
- If already in microvolts: `scaling_factor = 1.0`

**Method 3 — Empirical testing**:
```python
# Record 10 seconds of rest data
rest_data = adapter.get_latest_data(2500)  # 10s at 250 Hz
c3_rms = np.sqrt(np.mean(rest_data[:, 0]**2))
print(f"C3 RMS: {c3_rms}")
# Expected: 10-50 µV at rest
```

**Method 4 — Visual inspection**:
- Too low confidence (30-40%): Signal amplitude too small → increase `scaling_factor`
- Saturated predictions: Signal too large → decrease `scaling_factor`
- Normal (50-85%): Scaling is correct

### Expected Performance Improvement

| Metric | Without Bipolar | With Bipolar + Correct Scaling |
|---|---|---|
| Baseline accuracy | ~60-65% | ~73-75% |
| Confidence during imagery | 52-58% | 65-85% |
| Left/right discrimination | Poor | Clear |
| False positive rate | High | Low |

---

## 7. Leaky Accumulator

### Overview

The Leaky Accumulator stabilizes command triggering by accumulating prediction confidence over time with a configurable decay (leak). Commands are only emitted when a bucket (per class) exceeds a trigger threshold.

### How It Works

1. **Neutral zone**: Predictions with confidence between 0.45–0.55 are treated as "uncertain" and do not add to buckets
2. **Accumulation**: Predictions outside the neutral zone add a scaled contribution to the corresponding class bucket
3. **Decay (leak)**: All buckets are multiplied by `(1 - decay_rate)` on each update (default: 0.15)
4. **Triggering**: When a bucket ≥ `trigger_threshold` (default: 2.0) and cooldown has passed, a command is emitted

### Default Parameters

| Parameter | Default | Effect |
|---|---|---|
| `confidence_threshold` | 0.65 | Minimum confidence for smoothed mode |
| `smoothing_window` | 8 | Prediction history size |
| `accumulator_threshold` | 2.0 | Bucket fill level to trigger command |
| `accumulator_decay` | 0.15 | Leak rate per update |
| `neutral_zone` | (0.45, 0.55) | Uncertain predictions ignored |

### Usage Examples

```bash
# Normal operation (accumulator enabled by default)
python npg_realtime_bci.py --simulate

# Disable accumulator (fallback to smoothed thresholding)
python npg_realtime_bci.py --simulate --no-accumulator --confidence 0.7

# More sensitive (fewer missed triggers)
python npg_realtime_bci.py --simulate --acc-threshold 1.5 --acc-decay 0.1

# More conservative (fewer false triggers)
python npg_realtime_bci.py --simulate --acc-threshold 3.0 --acc-decay 0.25

# Wider neutral zone (ignore more uncertain predictions)
python npg_realtime_bci.py --simulate --neutral-low 0.40 --neutral-high 0.60
```

### Before vs After Accumulator

| Scenario | Without Accumulator | With Accumulator |
|---|---|---|
| Rest state | Many false triggers | No triggers |
| Motor imagery | Triggers on every epoch | Triggers on sustained imagery |
| Reliability | Poor | Excellent |

---

## 8. Pipeline Fixes Summary (v2.x)

### v2.0 Fixes

#### 1. Added 50 Hz Notch Filter

**Problem**: Powerline interference at 50 Hz corrupts the beta band (15-30 Hz).

**Fix**: IIR notch filter at 50 Hz (Q=30), applied after resampling and before bandpass filtering.

#### 2. Replaced CAR with Small Laplacian

**Problem**: Common Average Reference (CAR) is mathematically inappropriate for 3-channel setups because the average of 3 motor cortex channels is heavily biased and removes meaningful motor imagery signal.

**Fix**: Small Laplacian spatial filter:
```python
laplacian[:, 0] = c3 - 0.5 * cz           # C3 referenced to Cz
laplacian[:, 1] = cz - 0.25 * (c3 + c4)   # Cz referenced to motor average
laplacian[:, 2] = c4 - 0.5 * cz           # C4 referenced to Cz
```

#### 3. Stateful Filters for Real-Time Streaming

**Problem**: Using `filtfilt` (zero-phase, offline filtering) in real-time mode causes edge effects at epoch boundaries.

**Fix**: Implemented stateful SOS filters (`RealtimeBandpassFilter`, `RealtimeNotchFilter`) that maintain state between calls:
```python
class RealtimeBandpassFilter:
    def filter(self, data, stateful=True):
        for ch in range(data.shape[1]):
            filtered[:, ch], self.zi[ch] = signal.sosfilt(
                self.sos, data[:, ch], zi=self.zi[ch]
            )
```

#### 4. Fixed Buffer Size Constants

**Problem**: Window buffer was set to 1024 samples assuming 256 Hz, but actual input is 500 Hz.

**Fix**:
```python
INPUT_SAMPLING_RATE = 500
OUTPUT_SAMPLING_RATE = 250
EPOCH_DURATION = 4.0
INPUT_EPOCH_SAMPLES = int(EPOCH_DURATION * INPUT_SAMPLING_RATE)  # 2000
```

### v2.1 Fixes

#### 5. Anti-Aliasing Before Downsampling

**Problem**: Resampling from 500 Hz to 250 Hz without anti-aliasing caused aliasing artifacts in the mu/beta bands.

**Fix**: Added low-pass filter at 100 Hz (80% of target Nyquist) before downsampling:
```python
nyquist_target = output_rate / 2
antialias_cutoff = nyquist_target * 0.8  # 100 Hz
self.antialias_sos = signal.butter(8, antialias_cutoff, btype='low',
                                   fs=input_rate, output='sos')
```

#### 6. Calibrated Confidence via Temperature Scaling

**Problem**: Raw softmax outputs are not calibrated probabilities (0.7 softmax ≠ 70% accuracy).

**Fix**: Temperature scaling for calibrated confidence:
```python
class CalibratedConfidence:
    def calibrate(self, logits):
        scaled = logits / self.temperature  # temperature=1.5
        exp_scaled = np.exp(scaled - np.max(scaled))
        return exp_scaled / exp_scaled.sum()
```

#### 7. Confidence-Weighted Smoothing

**Problem**: Original smoothing used simple majority vote, ignoring confidence levels.

**Fix**: Confidence-weighted voting (emphasizes high-confidence predictions):
```python
for idx, conf in self.prediction_buffer:
    weight = conf ** 2
    class_weights[idx] += weight
```

#### 8. Faster Normalization Adaptation

**Problem**: EMA alpha of 0.99 made normalization adapt too slowly to new subjects.

**Fix**: Reduced alpha to 0.95 for faster cross-subject adaptation.

#### 9. Realistic Signal Quality Thresholds

**Problem**: Quality thresholds were set for raw ADC values, not microvolts.

**Fix**: Updated thresholds for typical EEG in µV:
```python
if std < 0.5:
    score = 0.1   # Flat signal — disconnected
elif 10 <= std <= 100 and max_abs < 300:
    score = 1.0   # Good signal
```

---

## 9. Model Configuration System

### Overview

The model factory allows switching between neural network architectures by editing a config file — no code changes needed.

```yaml
# config_2b.yaml
model:
  architecture: "eegnet"   # ← Change this to switch models
  chans: 3
  samples: 1000
  nb_classes: 2
```

```bash
python train_model_2b.py config_2b.yaml
```

### Available Models

| Model ID | Class | Parameters | Best For |
|---|---|---|---|
| `eegnet` | EEGNet | ~2,226 | Default, compact, proven |
| `simplecnn` | SimpleCNN | ~779,330 | Baseline comparison, prototyping |

### Adding a New Model (3 Steps)

**Step 1 — Create model file** (`my_model.py`):
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D
import yaml

class MyModel:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_config = self.config['model']
        self.model = None

    def build_model(self):
        input_layer = Input(shape=(
            self.model_config['chans'],
            self.model_config['samples'],
            1
        ))
        x = Conv2D(32, (1, 64), activation='relu')(input_layer)
        output = Dense(self.model_config['nb_classes'], activation='softmax')(x)
        self.model = Model(inputs=input_layer, outputs=output)
        return self.model

    def compile_model(self, optimizer='adam', learning_rate=0.001):
        from tensorflow.keras.optimizers import Adam
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
```

**Step 2 — Register in** `model_factory.py`:
```python
from my_model import MyModel

AVAILABLE_MODELS = {
    'eegnet': EEGNet,
    'simplecnn': SimpleCNN,
    'mymodel': MyModel,   # ← Add this line
}
```

**Step 3 — Use it**:
```yaml
model:
  architecture: "mymodel"
```

### Config File Templates

**EEGNet**:
```yaml
model:
  architecture: "eegnet"
  chans: 3
  samples: 1000
  nb_classes: 2
  kernLength: 64
  F1: 8
  D: 2
  F2: 16
  dropoutRate: 0.5
```

**SimpleCNN**:
```yaml
model:
  architecture: "simplecnn"
  chans: 3
  samples: 1000
  nb_classes: 2
  filters1: 16
  filters2: 32
  dropoutRate: 0.5
  dense_units: 128
```

### Running Multiple Models for Comparison

```bash
# Terminal 1: Train EEGNet
python train_model_2b.py config_2b.yaml

# Terminal 2: Train SimpleCNN
python train_model_2b.py config_simplecnn.yaml

# Check results
ls logs/evaluation_results_*.json
cat models/best_accuracy.json
```

### Testing the Model Factory

```bash
python test_model_factory.py
# Expected output:
# ✓ PASS: List Models
# ✓ PASS: EEGNet Creation
# ✓ PASS: SimpleCNN Creation
# ✓ PASS: Model Inference
# ✓ PASS: Model Switching
# 🎉 ALL TESTS PASSED!
```

### Using the Model Factory Directly

```python
from model_factory import ModelFactory

factory = ModelFactory('config_2b.yaml')
model_instance = factory.create_model(compile_model=True)
keras_model = model_instance.model
keras_model.summary()
```

---

## 10. Command Reference

### First-Time Setup

```bash
# Step 0: Determine scaling factor (hardware users only)
python calibrate_scaling.py --direct --port COM6

# Step 1: Run baseline calibration
python npg_realtime_bci.py --simulate --calibrate

# Step 2: Run BCI system
python npg_realtime_bci.py --simulate
```

### Daily Use

```bash
# Normal operation (loads saved calibration automatically)
python npg_realtime_bci.py --simulate

# Recalibrate if headset moved
python npg_realtime_bci.py --simulate --calibrate

# Hardware mode
python npg_realtime_bci.py --direct --port COM6
```

### Tuning Accumulator Sensitivity

```bash
# More sensitive (lower threshold, less decay)
python npg_realtime_bci.py --simulate --acc-threshold 1.5 --acc-decay 0.1

# More conservative (higher threshold, more decay)
python npg_realtime_bci.py --simulate --acc-threshold 3.0 --acc-decay 0.25

# Wider neutral zone
python npg_realtime_bci.py --simulate --neutral-low 0.40 --neutral-high 0.60
```

### Legacy Mode (Disable Enhancements)

```bash
python npg_realtime_bci.py --simulate --no-accumulator --no-bias-correction --confidence 0.3 --smoothing 3
```

### Training Models

```bash
# Train EEGNet on BCI Competition IV 2b (default)
python train_model_2b.py config_2b.yaml

# Train with SimpleCNN
python train_model_2b.py config_simplecnn.yaml

# Run hyperparameter sweep
python hyperparameter_sweep.py
```

---

## 11. Troubleshooting

### Hardware / Connection Issues

**"No LSL stream found"**
```
[ERROR] No NPG Lite LSL stream found (type: EXG)
```
→ Start Chords-Python in Terminal 1:
```bash
python -m chordspy.connection --protocol usb
```
→ Verify LSL stream is broadcasting:
```bash
python -c "from pylsl import resolve_streams; print(resolve_streams())"
```
→ Check USB connection: Windows Device Manager → Ports (COM & LPT), or Linux: `ls /dev/ttyUSB*`

**"Import error: chordspy"**
→ `pip install chordspy pylsl`

**"Model load error"**
```
[ERROR] Failed to load model: best_eegnet_2class_bci2b.keras
```
→ Must run from CODE directory: `cd CODE`
→ Retrain if missing: `python train_model_2b.py`

### Signal Quality Issues

**"Poor signal quality"**
```
[WARNING] C3: 45.2 µV RMS (HIGH NOISE)
```
1. Apply more conductive gel; press electrodes firmly
2. Move away from power cables; turn off nearby electronics
3. Clean skin with alcohol; ensure ground is on bony mastoid

**"Connection drops"**
- USB: Use high-quality USB cable; try different port; avoid USB hubs
- WiFi: Stay within 5 meters; use 2.4 GHz band
- Bluetooth: Stay within 10 meters; USB is more reliable

### Classification Issues

**"No commands triggering"**
1. Not calibrated → Run `--calibrate` first
2. Accumulator threshold too high → Try `--acc-threshold 1.5`
3. Decay too fast → Try `--acc-decay 0.1`
4. Neutral zone too wide → Try `--neutral-low 0.47 --neutral-high 0.53`

**"Too many false triggers"**
1. Accumulator threshold too low → Try `--acc-threshold 3.0`
2. Decay too slow → Try `--acc-decay 0.2`
3. Need recalibration → Run `--calibrate` again

**"Predictions stuck around 50%"**
1. Scaling factor wrong → Run `python calibrate_scaling.py`
2. Bipolar disabled → Verify `use_bipolar=True` in preprocessor
3. Baseline not calibrated → Run `--calibrate`
4. Poor electrode contact → Check impedances

**"Random classifications / no correlation with imagery"**
1. Use kinesthetic (feeling) not visual imagery
2. Imagine squeezing, not just thinking about the hand
3. Maintain imagery for full 4 seconds
4. Stay relaxed during rest periods between trials
5. Adjust threshold: `python npg_realtime_bci.py --threshold 0.7`

### Motor Imagery Best Practices

- Use **kinesthetic** (feeling/sensation) imagery, not visual imagery
- Imagine the **sensation** of squeezing, not the movement itself
- Maintain consistent imagery intensity for the full 4 seconds
- Stay completely relaxed between trials
- Rest 2-3 seconds between imagery attempts
- Practice regularly — accuracy improves with experience

---

## 12. Version History

| Version | Date | Key Features |
|---|---|---|
| 1.0 | Pre-2026 | Basic EEGNet model; simple threshold (0.3); bandpass + CAR + z-score |
| 2.0 | 2026-02-03 | Leaky Accumulator; 50 Hz notch filter; Baseline Calibration; stateful filters; fixed buffer size |
| 2.1 | 2026-02-03 | Virtual Bipolar montage; Microvolt scaling; Anti-aliasing LPF; calibrated confidence; confidence-weighted smoothing; faster normalization |

### v1.0 → v2.1 Configuration Changes

```python
# v1.0
confidence_threshold = 0.3
smoothing_window = 3
# No accumulator, no notch, CAR referencing, no calibration

# v2.1
confidence_threshold = 0.65    # +117% increase
smoothing_window = 8           # +167% increase
accumulator_threshold = 2.0    # New
accumulator_decay = 0.15       # New
neutral_zone = (0.45, 0.55)    # New
notch_freq = 50.0              # New
use_bipolar = True             # New (critical!)
scaling_factor = 1.0           # New (hardware-dependent)
apply_baseline_correction = True  # New
```

### Performance Improvement Summary

| Metric | v1.0 | v2.1 | Change |
|---|---|---|---|
| Rest confidence | 50-60% | ~50% | Calibrated |
| Imagery confidence | 52-65% | 65-85% | +20-25% |
| False positive rate | High | Low | Significant |
| Command stability | Poor | Excellent | Major |
| Signal format match | ❌ Monopolar | ✅ Bipolar | Critical |

---

## 13. References

### Academic

1. **EEGNet**: Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
2. **Motor Imagery**: Pfurtscheller, G., & Neuper, C. (2001). Motor imagery and direct brain-computer communication. *Proceedings of the IEEE*, 89(7), 1123-1134.
3. **BCI Competition IV**: Tangermann, M., et al. (2012). Review of the BCI competition IV. *Frontiers in Neuroscience*, 6, 55.
4. **Laplacian Filtering**: McFarland, D. J., et al. (1997). Spatial filter selection for EEG-based communication. *Electroencephalography and Clinical Neurophysiology*, 103(3), 386-394.
5. **BCI Competition IV 2b Dataset**: Leeb, R., Brunner, C., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). http://www.bbci.de/competition/iv/

### Software

- **Chords-Python**: https://github.com/upsidedownlabs/Chords-Python
- **NPG Lite Product Page**: https://upsidedownlabs.tech/
- **pylsl (Lab Streaming Layer)**: https://github.com/labstreaminglayer/liblsl-Python
- **TensorFlow/Keras**: https://www.tensorflow.org/

### Community

- Upside Down Labs Discord (NPG Lite hardware support)
- BCI subreddit: r/BCI
- NeuroTech community forums

---

**Document Version**: 2.1 | **Last Updated**: 2026-02-03

This document consolidates content from:
`NPG_LITE_QUICKSTART.md`, `NPG_LITE_USER_GUIDE.md`, `NPG_LITE_CORRECTION_SUMMARY.md`,
`BASELINE_CALIBRATION.md`, `VIRTUAL_BIPOLAR_GUIDE.md`, `LEAKY_ACCUMULATOR_SUMMARY.md`,
`PIPELINE_FIXES_SUMMARY.md`, `IMPLEMENTATION_SUMMARY_v2.1.md`,
`MODEL_CONFIGURATION_GUIDE.md`, `MODEL_SYSTEM_SUMMARY.md`, `MODEL_QUICK_REFERENCE.md`,
`QUICK_REFERENCE.md`
