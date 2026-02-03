# Baseline Calibration — 2-Class Model Bias Correction

## Problem

The BCI model uses a 2-class design (LEFT_HAND vs RIGHT_HAND) which forces a choice even when the user is at rest. This causes:

- Predictions to hover around 50-60% confidence during rest
- Predictions fall in the neutral zone (45-55%)
- Accumulator never triggers commands
- Natural bias toward one hand even at rest (e.g., 58% Left / 42% Right)

## Solution: Baseline Calibration

Record a rest state baseline and subtract the bias from all future predictions.

### How It Works

1. **Calibration Phase**: User sits still for 60 seconds (no imagery)
2. **Bias Calculation**: System averages all predictions during rest
   - Example: Model outputs 58% Left / 42% Right on average
   - Bias = [+0.08, -0.08] (deviation from ideal 50/50)
3. **Bias Correction**: Subtract bias from all future predictions
   - Raw prediction: [0.70 Left, 0.30 Right]
   - After correction: [0.62 Left, 0.38 Right]
   - This shifts predictions to be centered around 50/50 at rest

### Files Changed

- **CODE/npg_inference.py**
  - Added `baseline_bias` tracking
  - Added `start_calibration()`, `add_calibration_sample()`, `finalize_calibration()`
  - Added `_apply_bias_correction()` method
  - Applied bias correction in `predict()` method
  - Added bias persistence: `get_baseline_bias()`, `set_baseline_bias()`
- **CODE/npg_realtime_bci.py**
  - Added `run_calibration()` method for 60s baseline recording
  - Added `_save_baseline_bias()` and `_load_baseline_bias()` for JSON persistence
  - Added CLI flags: `--calibrate`, `--calibrate-duration`, `--no-bias-correction`
  - Auto-loads saved bias on startup

## Usage

### Step 1: Run Calibration (First Time)

```bash
python CODE/npg_realtime_bci.py --simulate --calibrate
```

Instructions during calibration:

1. Sit completely still
2. Look at a fixed point
3. Do NOT imagine any movement
4. Wait 60 seconds

The system will:

- Collect ~60-150 prediction samples
- Calculate average rest prediction
- Save bias to `CODE/baseline_bias.json`

### Step 2: Run BCI System (Normal Use)

```bash
python CODE/npg_realtime_bci.py --simulate
```

The system automatically:

- Loads `baseline_bias.json` if it exists
- Applies bias correction to all predictions
- Shows "Calibrated: [timestamp]" in logs

### Optional: Custom Calibration Duration

```bash
python CODE/npg_realtime_bci.py --simulate --calibrate --calibrate-duration 90
```

### Optional: Disable Bias Correction

```bash
python CODE/npg_realtime_bci.py --simulate --no-bias-correction
```

### Optional: Recalibrate

Just run calibration again - it will overwrite the previous baseline:

```bash
python CODE/npg_realtime_bci.py --simulate --calibrate
```

## Expected Results

### Before Calibration

```
Raw prediction during rest: 58% Left / 42% Right
Confidence in neutral zone (45-55%): ❌ No triggers
```

### After Calibration

```
Bias correction: [-0.08, +0.08]
Corrected prediction during rest: 50% Left / 50% Right ✅
Corrected prediction during left imagery: 75% Left / 25% Right ✅
Confidence now meaningful and triggers commands properly
```

## Bias File Format

`CODE/baseline_bias.json`:

```json
{
  "bias": [0.08, -0.08],
  "timestamp": "2026-02-03 14:20:00",
  "class_names": ["LEFT_HAND", "RIGHT_HAND"]
}
```

## Technical Details

### Bias Correction Formula

```python
corrected = raw_probabilities - baseline_bias
corrected = np.maximum(corrected, 0.0)  # Clip negative
corrected = corrected / corrected.sum()  # Renormalize to sum=1.0
```

### When to Recalibrate

- **Always** before first use
- After moving the headset
- After days/weeks of not using the system
- If predictions seem consistently biased to one side
- After changing electrode impedances

### Statistics

Check calibration status:

```python
stats = engine.get_statistics()
if stats['baseline_bias']['calibrated']:
    print(f"Bias: {stats['baseline_bias']['left_bias']}, {stats['baseline_bias']['right_bias']}")
```

## Integration with Leaky Accumulator

Baseline calibration works seamlessly with the Leaky Accumulator:

1. Raw prediction from model
2. **Bias correction applied** ← New step
3. Smoothing window applied
4. Accumulator updated
5. Command triggered when threshold reached

This combination provides stable, reliable command triggering.
