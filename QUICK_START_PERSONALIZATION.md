# Quick Start: Personal BCI Calibration

Fast-track guide to personalizing your BCI system in 4 steps.

## Prerequisites

- ✅ NPG Lite device (or use simulator)
- ✅ Python environment with dependencies installed
- ✅ Base model trained (`models/best/eegnet_2class_bci2b.keras`)

## 1️⃣ Record Calibration Data (25 min)

```bash
python CODE/record_personal_calibration.py --user-id=YOUR_NAME
```

**Tips:**

- Sit comfortably in quiet environment
- Focus on the kinesthetic feeling of hand movement
- Don't actually move your hands - only imagine
- Take breaks if you get tired

**Output:** `calibration_data/YOUR_NAME_calibration_*.npz`

## 2️⃣ Fine-Tune Model (5-10 min)

```bash
python CODE/fine_tune_personal_model.py \
  --calibration-file=calibration_data/YOUR_NAME_*.npz \
  --strategy=last-block
```

**What to expect:**

- Training progress bars for 20-50 epochs
- Best validation accuracy displayed
- Model automatically registered

**Output:**

- `models/personalized/YOUR_NAME_finetuned.keras`
- `models/personalized/YOUR_NAME_neutral_threshold.json`

## 3️⃣ Validate System (3 min)

```bash
python CODE/validate_3state_system.py --user-id=YOUR_NAME
```

**What to expect:**

- 30 trials total (10 LEFT, 10 RIGHT, 10 NEUTRAL)
- Real-time feedback during trials
- Confusion matrix and accuracy report

**Target:** >80% accuracy per state

**If validation fails:**

```bash
# Adjust threshold and retry
python CODE/validate_3state_system.py --user-id=YOUR_NAME --threshold=0.70
```

## 4️⃣ Use Real-Time (Continuous)

```bash
python CODE/npg_realtime_bci.py --user-id=YOUR_NAME
```

**Commands:**

- 👈 **LEFT_HAND** → device1/on
- 👉 **RIGHT_HAND** → device2/on
- ⚪ **NEUTRAL** → all/off

**Press Ctrl+C to stop**

---

## With Simulator (Testing)

If you don't have NPG Lite hardware, test with simulator:

```bash
# Step 1: Record (simulated data)
python CODE/record_personal_calibration.py --user-id=test_user --simulate

# Step 2: Fine-tune (same process)
python CODE/fine_tune_personal_model.py \
  --calibration-file=calibration_data/test_user_*.npz \
  --strategy=last-block

# Step 3: Validate (simulated)
python CODE/validate_3state_system.py --user-id=test_user --simulate

# Step 4: Real-time (simulated)
python CODE/npg_realtime_bci.py --user-id=test_user --simulate
```

---

## Typical Timeline

| Step                  | Duration   | Can Skip?      |
| --------------------- | ---------- | -------------- |
| 1. Record Calibration | 25 min     | ❌ Required    |
| 2. Fine-Tune Model    | 5-10 min   | ❌ Required    |
| 3. Validate System    | 3 min      | ⚠️ Recommended |
| 4. Real-Time Use      | Continuous | -              |

**Total setup time:** ~35 minutes

---

## Common Issues

### ❌ "Model not found"

**Solution:** Train base model first:

```bash
cd CODE
python train_model_2b.py
```

### ❌ "Failed to connect to device"

**Solution:**

- Check USB connection
- Try simulator: add `--simulate` flag
- Or start Chords-Python: `python -m chordspy.connection --protocol usb`

### ❌ "Validation accuracy too low"

**Solutions:**

1. Re-record calibration with better focus
2. Use more trials: `--left=150 --right=150 --rest=75`
3. Try different strategy: `--strategy=full`
4. Adjust threshold: `--threshold=0.70`

---

## What's Next?

- 📖 Read [PERSONALIZATION_GUIDE.md](PERSONALIZATION_GUIDE.md) for detailed docs
- 🔧 Adjust parameters for better performance
- 📊 Monitor accuracy and re-calibrate weekly
- 🚀 Integrate with IoT devices

---

**Need help?** Check troubleshooting section in [PERSONALIZATION_GUIDE.md](PERSONALIZATION_GUIDE.md#-troubleshooting)
