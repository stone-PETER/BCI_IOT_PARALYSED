# Personal EEG Calibration & Fine-Tuning System

Complete implementation for personalizing BCI models with user-specific data and 3-state classification (LEFT/RIGHT/NEUTRAL).

## 📋 Overview

This system enables:

- **Personal calibration data collection** with guided motor imagery trials
- **Transfer learning** with layer freezing strategies (head-only, last-block, full)
- **Neutral threshold calibration** from REST trials for 3-state classification
- **Model registry** for managing base and personalized models
- **3-state validation** to ensure LEFT/RIGHT/NEUTRAL accuracy before deployment
- **Real-time inference** with personalized models

## 🎯 Complete Workflow

### Step 1: Collect Calibration Data

Record personal motor imagery trials with guided visual cues:

```bash
# With real NPG Lite device
python CODE/record_personal_calibration.py --user-id=john_doe

# With simulator for testing
python CODE/record_personal_calibration.py --user-id=test_user --simulate

# Custom trial counts (default: 100 LEFT, 100 RIGHT, 50 REST)
python CODE/record_personal_calibration.py --user-id=john_doe --left=50 --right=50 --rest=30
```

**What happens:**

- Displays randomized cues: "IMAGINE LEFT HAND", "IMAGINE RIGHT HAND", "STAY NEUTRAL"
- Records 4-second EEG epochs with 2-second rest between trials
- Applies preprocessing: bandpass 8-30Hz, resample 500→250Hz, z-score normalization
- Saves to `calibration_data/{user_id}_calibration_{timestamp}.npz`

**Duration:** ~25 minutes for default trial counts (250 trials × 6s/trial)

### Step 2: Fine-Tune Model

Fine-tune pretrained EEGNet on personal data with layer freezing:

```bash
# Recommended: Last-block strategy (freeze Block 1, train Block 2 + head)
python CODE/fine_tune_personal_model.py --calibration-file=calibration_data/john_doe_*.npz --strategy=last-block

# Head-only (minimal data, faster)
python CODE/fine_tune_personal_model.py --calibration-file=calibration_data/john_doe_*.npz --strategy=head-only

# Full fine-tuning (requires lots of data)
python CODE/fine_tune_personal_model.py --calibration-file=calibration_data/john_doe_*.npz --strategy=full

# Mix with benchmark data (for small personal datasets)
python CODE/fine_tune_personal_model.py --calibration-file=calibration_data/john_doe_*.npz --mix-benchmark --personal-weight=0.7
```

**What happens:**

1. Loads pretrained `models/best/eegnet_2class_bci2b.keras`
2. Freezes layers according to strategy:
   - **head-only**: Freeze all except Dense layer (30+ trials needed)
   - **last-block**: Freeze Block 1, train Block 2 + head (100+ trials, recommended)
   - **full**: Train entire model (500+ trials needed)
3. Trains with learning rate 1e-4 (10x lower than base training)
4. Uses EarlyStopping (patience=10) and saves best checkpoint
5. Calibrates neutral threshold from REST trials (75th percentile)
6. Saves to `models/personalized/{user_id}_finetuned.keras`
7. Registers in `model_registry.json`

**Output files:**

- `models/personalized/{user_id}_finetuned.keras` - Fine-tuned model
- `models/personalized/{user_id}_neutral_threshold.json` - Neutral threshold
- `models/personalized/{user_id}_metadata.json` - Training metadata

### Step 3: Validate 3-State System

Test LEFT/RIGHT/NEUTRAL classification with 30 trials (10 per state):

```bash
# With real device
python CODE/validate_3state_system.py --user-id=john_doe

# With simulator
python CODE/validate_3state_system.py --user-id=john_doe --simulate

# Adjust threshold if needed
python CODE/validate_3state_system.py --user-id=john_doe --threshold=0.70
```

**What happens:**

- Loads personalized model and neutral threshold from registry
- Runs 10 trials for each state (LEFT, RIGHT, NEUTRAL)
- Calculates per-state accuracy (target: >80%)
- Computes confusion matrix and neutral false-trigger rate
- Suggests threshold adjustments if needed
- Saves results to `models/personalized/{user_id}_validation.json`

**Interpretation:**

- ✅ **PASS**: All states >80% accuracy, neutral false-triggers <20%
- ❌ **FAIL**: Need to adjust threshold or re-record calibration data

### Step 4: Use Personalized Model in Real-Time

Run BCI system with personalized model:

```bash
# With real NPG Lite device
python CODE/npg_realtime_bci.py --user-id=john_doe

# With simulator
python CODE/npg_realtime_bci.py --user-id=john_doe --simulate
```

**What happens:**

- Loads personalized model from registry
- Enables 3-state classification (LEFT/RIGHT/NEUTRAL)
- Uses calibrated neutral threshold
- Displays real-time predictions with confidence

**Commands:**

- **LEFT_HAND** (👈): Imagine clenching LEFT fist → device1/on
- **RIGHT_HAND** (👉): Imagine clenching RIGHT fist → device2/on
- **NEUTRAL** (⚪): No motor imagery → all/off

## 📁 File Structure

### New Files Created

```
CODE/
├── record_personal_calibration.py    # Step 1: Calibration data recorder
├── fine_tune_personal_model.py       # Step 2: Fine-tuning script
├── validate_3state_system.py         # Step 3: Validation tool
├── model_registry.json               # Model registry (base + personalized)
└── models/
    └── personalized/
        ├── {user_id}_finetuned.keras          # Fine-tuned model
        ├── {user_id}_neutral_threshold.json   # Neutral threshold
        ├── {user_id}_metadata.json           # Training metadata
        └── {user_id}_validation.json         # Validation results

calibration_data/
└── {user_id}_calibration_{timestamp}.npz  # Personal calibration data
```

### Modified Files

1. **bci4_2b_loader_v2.py**
   - Added `load_personal_calibration(npz_path)` - Load personal .npz files
   - Added `load_mixed_data()` - Mix personal + benchmark data

2. **npg_inference.py**
   - Added `neutral_threshold` parameter to `NPGInferenceEngine.__init__`
   - Added 3-state classification logic in `predict()` and `predict_with_uncertainty()`
   - Added `load_neutral_threshold()`, `set_neutral_threshold()`, `clear_neutral_threshold()`
   - Updated `LeakyAccumulator` to handle NEUTRAL state (class_idx=2)

3. **model_factory.py**
   - Added `load_from_registry(user_id)` - Load model info from registry
   - Added `register_personalized_model()` - Register personalized model
   - Added `list_personalized_models()` - List all user models
   - Added `get_model_info(user_id)` - Get model details

4. **npg_realtime_bci.py**
   - Added `--user-id` argument to load personalized models
   - Added `neutral_threshold` parameter throughout
   - Updated command tracking to include NEUTRAL state
   - Updated logging to display ⚪ icon for NEUTRAL

## 🔧 Fine-Tuning Strategies

### Head-Only (Fastest, Minimal Data)

- **Trainable:** Dense classification layer only
- **Frozen:** All convolutional layers (Block 1 + Block 2)
- **Data needed:** 30+ trials (minimum)
- **Use case:** Very limited calibration data, quick personalization

### Last-Block (Recommended, Balanced)

- **Trainable:** Block 2 (SeparableConv + pooling + dropout) + Dense head
- **Frozen:** Block 1 (temporal + spatial filters)
- **Data needed:** 100+ trials
- **Use case:** Standard personalization with decent data

### Full (Best Quality, Most Data)

- **Trainable:** Entire model (all layers)
- **Frozen:** None
- **Data needed:** 500+ trials
- **Use case:** Extensive calibration data available

## 🎚️ Neutral Threshold Calibration

### How It Works

1. Pass 50 REST epochs through fine-tuned model
2. For each: `max_confidence = max(prob_left, prob_right)`
3. Compute **75th percentile** as threshold
4. At inference: if `max(confidences) < threshold` → NEUTRAL

### Threshold Adjustment

If validation shows:

- **High neutral false-triggers (>20%)**: Increase threshold by 0.05
- **Missed LEFT/RIGHT commands**: Decrease threshold by 0.05

Manual adjustment:

```bash
python CODE/validate_3state_system.py --user-id=john_doe --threshold=0.70
```

## 📊 Model Registry

**Location:** `CODE/model_registry.json`

**Structure:**

```json
{
  "base_model": {
    "path": "models/best/eegnet_2class_bci2b.keras",
    "type": "pretrained",
    "accuracy": 0.7364,
    "trained_on": "BCI_IV_2b_9subjects"
  },
  "personalized_models": {
    "john_doe": {
      "path": "models/personalized/john_doe_finetuned.keras",
      "threshold_path": "models/personalized/john_doe_neutral_threshold.json",
      "type": "personalized",
      "neutral_threshold": 0.68,
      "timestamp": "2026-03-08T10:30:00",
      "val_accuracy": 0.82,
      "fine_tuning_strategy": "last-block"
    }
  }
}
```

**Usage:**

```python
from model_factory import ModelFactory

# Load personalized model info
model_info = ModelFactory.load_from_registry("john_doe")
print(model_info['path'])          # Model path
print(model_info['neutral_threshold'])  # Neutral threshold

# List all personalized models
users = ModelFactory.list_personalized_models()
print(users)  # ['john_doe', 'jane_smith', ...]
```

## ⚙️ Technical Details

### EEGNet Layer Structure

```
Block 1 (Layers 0-6):
  [0] Input (3, 1000, 1)
  [1] Conv2D (temporal filters)
  [2] BatchNormalization
  [3] DepthwiseConv2D (spatial filters)
  [4] BatchNormalization
  [5] AveragePooling2D
  [6] Dropout

Block 2 (Layers 7-12):
  [7] SeparableConv2D
  [8] BatchNormalization
  [9] Activation (ELU)
  [10] AveragePooling2D
  [11] Dropout

Head (Layers 13-15):
  [12] Flatten
  [13] Dense (classification)
  [14] Softmax
```

### Data Formats

**Calibration NPZ:**

```python
{
    'epochs_left': (n_left, 3, 1000),    # LEFT trials
    'epochs_right': (n_right, 3, 1000),  # RIGHT trials
    'epochs_rest': (n_rest, 3, 1000),    # REST trials
    'sampling_rate': 250,                 # Hz
    'channel_names': ['C3', 'Cz', 'C4'],
    'user_id': 'john_doe',
    'timestamp': '20260308_103000'
}
```

**Neutral Threshold JSON:**

```json
{
  "threshold": 0.68,
  "mean": 0.61,
  "median": 0.62,
  "std": 0.08,
  "p75": 0.68,
  "p90": 0.74,
  "recommended_min": 0.63,
  "recommended_max": 0.73,
  "n_rest_trials": 50
}
```

## 🚨 Troubleshooting

### Low Validation Accuracy (<80%)

**Possible causes:**

1. Inconsistent motor imagery during calibration
2. Too much noise/artifacts in data
3. Insufficient calibration data
4. Wrong neutral threshold

**Solutions:**

- Re-record calibration data with better focus
- Use more trials (150-200 per class)
- Adjust neutral threshold manually
- Try different fine-tuning strategy

### High Neutral False-Trigger Rate (>20%)

**Cause:** Neutral threshold too low

**Solution:**

```bash
# Increase threshold
python CODE/validate_3state_system.py --user-id=john_doe --threshold=0.70

# If successful, update manually
python -c "
import json
threshold_file = 'models/personalized/john_doe_neutral_threshold.json'
with open(threshold_file, 'r+') as f:
    data = json.load(f)
    data['threshold'] = 0.70
    f.seek(0)
    json.dump(data, f, indent=2)
"
```

### Model Not Found Error

**Cause:** Trying to use personalized model before fine-tuning

**Solution:**

```bash
# Complete fine-tuning first
python CODE/fine_tune_personal_model.py --calibration-file=calibration_data/john_doe_*.npz
```

## 📈 Performance Expectations

### Base Model (BCI Competition IV 2b)

- **Accuracy:** 73.64% (average across 9 subjects)
- **Range:** 60-85% per subject (high variability)

### Personalized Model (After Fine-Tuning)

- **Expected improvement:** +5-15% absolute accuracy
- **Target:** >80% on personal validation data
- **3-state accuracy:** >80% per state (LEFT/RIGHT/NEUTRAL)

### Factors Affecting Performance

1. **Quality of motor imagery:** Consistent, vivid imagination
2. **Electrode placement:** Proper C3, Cz, C4 positioning
3. **Signal quality:** Low impedance
   , minimal artifacts
4. **Training data:** More trials = better performance
5. **Individual differences:** Some users naturally perform better

## 🎓 Best Practices

### Calibration Session

1. **Environment:** Quiet, distraction-free
2. **Focus:** Concentrate on kinesthetic feeling, not visual imagery
3. **Consistency:** Use same mental strategy for all trials
4. **Breaks:** Take 2-3 minute breaks every 50 trials
5. **Timing:** Record when alert and focused (avoid fatigue)

### Fine-Tuning

1. **Start with last-block strategy** (good balance)
2. **Use mixed data if <100 trials** per class
3. **Monitor validation accuracy** (watch for overfitting)
4. **Keep base model intact** (never overwrite)

### Real-Time Use

1. **Validate before deployment** (ensure >80% accuracy)
2. **Use Leaky Accumulator** (reduces false triggers)
3. **Monitor uncertainty** (high uncertainty = poor signal)
4. **Re-calibrate periodically** (every 1-2 weeks)

## 📚 References

- **EEGNet:** Lawhern et al. (2018), "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces"
- **BCI Competition IV Dataset 2b:** Leeb et al. (2008), Binary motor imagery classification
- **Transfer Learning:** Yosinski et al. (2014), "How transferable are features in deep neural networks?"

## 🔗 Related Files

- [MODEL_CONFIGURATION_GUIDE.md](CODE/MODEL_CONFIGURATION_GUIDE.md) - Model architecture details
- [NPG_LITE_USER_GUIDE.md](CODE/NPG_LITE_USER_GUIDE.md) - NPG Lite hardware guide
- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) - System roadmap

---

**Created:** March 8, 2026  
**Version:** 1.0.0  
**Status:** ✅ Production Ready
