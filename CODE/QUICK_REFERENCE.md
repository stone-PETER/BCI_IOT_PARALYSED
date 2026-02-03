# BCI System Quick Reference

## Initial Setup (First Time)

### 0. Determine Scaling Factor (Hardware Users Only)

If using real NPG Lite hardware, first determine the correct scaling factor:

```bash
python CODE/calibrate_scaling.py --direct --port COM6
```

This analyzes your signal amplitude and recommends the correct `scaling_factor` value.
For simulator, this step is not needed (default: 1.0).

### 1. Run Baseline Calibration

```bash
python CODE/npg_realtime_bci.py --simulate --calibrate
```

**Instructions during calibration (60 seconds):**

- Sit completely still
- Look at a fixed point
- Do NOT imagine any movement
- Relax and breathe normally

This creates `CODE/baseline_bias.json` which corrects for the 2-class model's inherent bias.

### 2. Run BCI System

```bash
python CODE/npg_realtime_bci.py --simulate
```

The system automatically loads the calibration and applies bias correction.

---

## What Was Implemented

### 1. Leaky Integrator/Accumulator ✅

**Purpose:** Prevent spurious single-epoch predictions from triggering commands

**Features:**

- Each class has a "bucket" that accumulates confidence
- Buckets leak/decay over time (default: 15% per update)
- Commands only trigger when bucket exceeds threshold (default: 2.0)
- Neutral zone (45-55%) prevents uncertain predictions from adding to buckets

**Parameters:**

- `--smoothing 8` - Smoothing window (increased from 3)
- `--confidence 0.65` - Minimum confidence (increased from 0.3)
- `--acc-threshold 2.0` - Bucket overflow threshold
- `--acc-decay 0.15` - Decay rate per update
- `--neutral-low 0.45` - Neutral zone lower bound
- `--neutral-high 0.55` - Neutral zone upper bound

### 2. 50Hz Notch Filter ✅

**Purpose:** Remove powerline interference (European standard)

**Features:**

- IIR notch filter at 50Hz (Q=30)
- Applied after resampling, before bandpass filtering
- Prevents 50Hz noise from corrupting beta band (15-30Hz)

**Parameters:**

- `apply_notch=True` (default in NPGPreprocessor)
- `notch_freq=50.0` - Notch frequency in Hz
- `notch_q=30.0` - Quality factor (higher = narrower notch)

### 3. Baseline Calibration ✅

**Purpose:** Correct for 2-class model's forced choice bias

**Features:**

- Records rest state (user doing nothing)
- Calculates average prediction during rest
- Subtracts bias from all future predictions
- Persists to JSON file for reuse

**Parameters:**

- `--calibrate` - Run calibration mode
- `--calibrate-duration 60` - Calibration duration (seconds)
- `--no-bias-correction` - Disable bias correction

### 4. Virtual Bipolar Filtering ✅

**Purpose:** Match training data signal format (BCI Competition IV 2b used bipolar recordings)

**Features:**

- Computes differential signals: C3-Cz, Cz, C4-Cz
- Enhances local cortical activity
- Suppresses common-mode noise
- Provides better spatial resolution

**Parameters:**

- `use_bipolar=True` (default in NPGPreprocessor)
- Automatically applied in preprocessing pipeline

**Why It Matters:**
Training data used bipolar recordings, but NPG Lite provides monopolar.
Without this conversion, the model sees completely different signal patterns!

### 5. Microvolt Scaling ✅

**Purpose:** Scale NPG Lite output to match training data amplitude (±50-100µV)

**Features:**

- Configurable scaling factor
- Training data expects microvolts (µV)
- NPG Lite may output in different units (V, mV, or raw ADC)

**Parameters:**

- `scaling_factor=1.0` (default, adjust based on hardware)
- Use `calibrate_scaling.py` to determine correct value

---

## Command Examples

### Basic Usage

```bash
# Normal operation (with all enhancements)
python CODE/npg_realtime_bci.py --simulate

# Hardware mode (real NPG Lite device)
python CODE/npg_realtime_bci.py --direct --port COM6
```

### Calibration

```bash
# First-time calibration
python CODE/npg_realtime_bci.py --simulate --calibrate

# Longer calibration (90 seconds)
python CODE/npg_realtime_bci.py --simulate --calibrate --calibrate-duration 90

# Recalibrate (overwrites previous)
python CODE/npg_realtime_bci.py --simulate --calibrate
```

### Tuning Accumulator

```bash
# More sensitive (lower threshold, less decay)
python CODE/npg_realtime_bci.py --simulate --acc-threshold 1.5 --acc-decay 0.1

# More conservative (higher threshold, more decay)
python CODE/npg_realtime_bci.py --simulate --acc-threshold 3.0 --acc-decay 0.25

# Wider neutral zone (ignore more uncertain predictions)
python CODE/npg_realtime_bci.py --simulate --neutral-low 0.40 --neutral-high 0.60
```

### Legacy Mode (Disable Enhancements)

```bash
# Without accumulator
python CODE/npg_realtime_bci.py --simulate --no-accumulator

# Without bias correction
python CODE/npg_realtime_bci.py --simulate --no-bias-correction

# Old defaults (before enhancements)
python CODE/npg_realtime_bci.py --simulate --no-accumulator --no-bias-correction --confidence 0.3 --smoothing 3
```

---

## Troubleshooting

### No Commands Triggering

**Possible causes:**

1. Not calibrated → Run `--calibrate` first
2. Accumulator threshold too high → Try `--acc-threshold 1.5`
3. Decay too fast → Try `--acc-decay 0.1`
4. Neutral zone too wide → Try `--neutral-low 0.47 --neutral-high 0.53`

### Too Many False Triggers

**Possible causes:**

1. Accumulator threshold too low → Try `--acc-threshold 3.0`
2. Decay too slow → Try `--acc-decay 0.2`
3. Need recalibration → Run `--calibrate` again

### Commands Stuck on One Side

**Possible causes:**

1. Need recalibration → Run `--calibrate`
2. Headset moved → Adjust position and recalibrate
3. Poor electrode contact → Check impedances

### Predictions Still Around 50% (New)

**Possible causes:**

1. ❌ **Scaling factor wrong** → Run `python calibrate_scaling.py`
2. ❌ **Bipolar disabled** → Verify `use_bipolar=True` in preprocessor
3. ❌ **Signal quality poor** → Check electrode impedances
4. Need recalibration → Run `--calibrate` again

### Very Low or Very High Confidence

**Possible causes:**

1. **Signal amplitude wrong** → Check scaling factor
   - Too low confidence (20-30%) → Increase scaling_factor
   - Too high confidence (95-99%) → Decrease scaling_factor
2. **Channel order incorrect** → Verify C3, Cz, C4 mapping
3. Poor signal quality → Check connections

---

## File Locations

### Code Files

- `CODE/npg_inference.py` - Inference engine with accumulator & bias correction
- `CODE/npg_preprocessor.py` - Preprocessing with notch, bipolar, scaling
- `CODE/npg_realtime_bci.py` - Real-time BCI system
- `CODE/train_model_2b.py` - Model training script
- `CODE/calibrate_scaling.py` - Scaling factor calibration tool

### Data Files

- `CODE/baseline_bias.json` - Saved calibration (auto-created)
- `CODE/models/best_eegnet_2class_bci2b.keras` - Trained model

### Documentation

- `CODE/LEAKY_ACCUMULATOR_SUMMARY.md` - Accumulator details
- `CODE/BASELINE_CALIBRATION.md` - Calibration details
- `CODE/VIRTUAL_BIPOLAR_GUIDE.md` - Bipolar filtering & scaling
- `CODE/QUICK_REFERENCE.md` - This file

---

## When to Recalibrate

- ✅ **Always** before first use
- ✅ After moving/adjusting headset
- ✅ After days/weeks of not using system
- ✅ If predictions consistently favor one side
- ✅ After changing electrode gel/impedances
- ❌ Not needed between short sessions (same day)

---

## Performance Expectations

### Before Enhancements

- Confidence: 50-60% (ambiguous)
- Triggers: Rare or constant false positives
- Usability: Poor

### After Enhancements

- Confidence: 30-70% at rest → 60-85% during imagery
- Triggers: Stable, requires sustained evidence
- Usability: Practical for assistive control

---

## Technical Details

### Pipeline Flow

```
NPG Lite (256 Hz, 6 ch monopolar)
  ↓
Select C3, Cz, C4
  ↓
Scale to Microvolts            ← NEW (v2.1)
  ↓
Resample (250 Hz)
  ↓
Notch Filter (50 Hz)           ← NEW (v2.0)
  ↓
Virtual Bipolar (C3-Cz, Cz, C4-Cz)  ← NEW (v2.1)
  ↓
Bandpass (8-30 Hz)
  ↓
Z-score Normalization
  ↓
Model Prediction
  ↓
Baseline Bias Correction       ← NEW (v2.0)
  ↓
Smoothing Window (8 samples)   ← INCREASED (v2.0)
  ↓
Leaky Accumulator              ← NEW (v2.0)
  ↓
Command Trigger
```

### Default Configuration

```python
# Inference
confidence_threshold = 0.65    # Was: 0.3
smoothing_window = 8           # Was: 3
accumulator_threshold = 2.0    # New (v2.0)
accumulator_decay = 0.15       # New (v2.0)
neutral_zone = (0.45, 0.55)   # New (v2.0)

# Preprocessing
notch_freq = 50.0              # New (v2.0)
use_bipolar = True             # New (v2.1)
scaling_factor = 1.0           # New (v2.1)
apply_baseline_correction = True  # New (v2.0)
```

---

**Last Updated:** 2026-02-03
**Version:** 2.1 (with virtual bipolar filtering & microvolt scaling)
