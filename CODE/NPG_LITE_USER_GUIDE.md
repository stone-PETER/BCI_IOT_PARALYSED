# NPG Lite BCI Integration Guide

Complete guide for using NPG Lite Beast Pack (by Upside Down Labs) with our motor imagery BCI system.

## Table of Contents

1. [Hardware Overview](#hardware-overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Hardware Setup](#hardware-setup)
5. [Running the BCI System](#running-the-bci-system)
6. [Testing & Validation](#testing--validation)
7. [Troubleshooting](#troubleshooting)
8. [Technical Details](#technical-details)

---

## Hardware Overview

### NPG Lite Beast Pack

- **Manufacturer**: Upside Down Labs
- **Channels**: 3 (C3, Cz, C4 for motor imagery)
- **Sampling Rate**: 500 Hz
- **Resolution**: 12-bit ADC
- **Connectivity**: USB, WiFi, or Bluetooth
- **Official Library**: Chords-Python

### Why NPG Lite is Perfect for This Project

✅ **3 channels** exactly match our trained model (BCI Competition IV 2b)  
✅ **C3, Cz, C4** placement ideal for left/right hand motor imagery  
✅ **500 Hz** sampling easily downsamples to 250 Hz (1:2 ratio)  
✅ **Wireless** options for comfortable, unrestricted use  
✅ **Open-source** ecosystem with Chords-Python library

---

## Quick Start

**30-Second Test** (no hardware required):

```bash
# Activate Python environment
cd CODE

# Test with simulator
python npg_realtime_bci.py --simulate
```

**With NPG Lite Hardware**:

```bash
# Terminal 1: Start Chords-Python connection
python -m chordspy.connection --protocol usb

# Terminal 2: Start BCI system
python npg_realtime_bci.py
```

---

## Installation

### 1. Install Python Dependencies

```bash
cd CODE
pip install -r requirements.txt
```

Key packages installed:

- `chordspy` - NPG Lite communication library
- `pylsl` - Lab Streaming Layer for data streaming
- `tensorflow` - Neural network inference
- `scipy` - Signal processing

### 2. Verify Installation

```bash
# Check Chords-Python
python -c "import chordspy; print('Chords-Python OK')"

# Check LSL
python -c "import pylsl; print('LSL OK')"

# Test simulator
python npg_realtime_bci.py --simulate
```

---

## Hardware Setup

### 1. Electrode Placement

**Standard 10-20 System for Motor Imagery**:

```
         Cz (center, reference)
          |
    C3 -------+------- C4
 (left)               (right)

Ground: Right mastoid (behind right ear)
```

**Electrode positions**:

- **C3** (left hemisphere): 2cm behind, 7cm left of vertex
- **Cz** (central): Top center of head
- **C4** (right hemisphere): 2cm behind, 7cm right of vertex
- **Ground**: Right mastoid (bony protrusion behind right ear)

### 2. Skin Preparation

1. Clean electrode sites with alcohol wipes
2. Gently abrade skin with prep gel (if provided)
3. Apply conductive gel to electrodes
4. Ensure impedance < 10 kΩ (check with NPG Lite app)

### 3. Connect NPG Lite

**USB Connection** (recommended for testing):

```bash
# Connect NPG Lite via USB cable
# Device should appear as COM port (Windows) or /dev/ttyUSB* (Linux)

# Start Chords-Python
python -m chordspy.connection --protocol usb
```

**WiFi Connection** (for wireless operation):

```bash
# Configure NPG Lite WiFi using official app
# Note the IP address assigned to NPG Lite

# Start Chords-Python with WiFi
python -m chordspy.connection --protocol wifi --ip 192.168.1.XXX
```

**Bluetooth Connection** (alternative wireless):

```bash
# Pair NPG Lite via system Bluetooth settings
# Note the device address

# Start Chords-Python with BLE
python -m chordspy.connection --protocol ble
```

### 4. Verify Signal Quality

```bash
# Run electrode verification tool
python electrode_placement_verifier.py

# Should show:
# ✓ C3: 8.2 µV RMS (GOOD)
# ✓ Cz: 7.9 µV RMS (GOOD)
# ✓ C4: 8.5 µV RMS (GOOD)
```

---

## Running the BCI System

### Two-Terminal Workflow

**Terminal 1: Chords-Python (Hardware Connection)**

```bash
cd CODE

# For USB connection
python -m chordspy.connection --protocol usb

# You should see:
# "Connected to NPG Lite"
# "Streaming EEG data to LSL (500 Hz, 3 channels)"
# "Stream name: BioAmpDataStream"
```

**Terminal 2: BCI Application**

```bash
cd CODE

# Start real-time BCI
python npg_realtime_bci.py

# You should see:
# "Searching for NPG Lite LSL stream..."
# "Connected to: BioAmpDataStream (3 channels @ 500 Hz)"
# "Model loaded: 73.64% accuracy"
# "Ready for motor imagery classification"
```

### Command-Line Options

```bash
# Basic usage (looks for LSL stream)
python npg_realtime_bci.py

# Simulator mode (no hardware needed)
python npg_realtime_bci.py --simulate

# Custom model path
python npg_realtime_bci.py --model ../models/custom_model.keras

# Adjust classification threshold
python npg_realtime_bci.py --threshold 0.6

# Enable debug logging
python npg_realtime_bci.py --verbose
```

### Expected Output

```
=== NPG Lite Real-Time BCI System ===

[INFO] Searching for NPG Lite LSL stream (type: EXG)...
[INFO] Found stream: BioAmpDataStream (3 channels @ 500.0 Hz)
[INFO] Connected to NPG Lite via Chords-Python
[INFO] Model loaded: best_eegnet_2class_bci2b.keras
[INFO] Model accuracy: 73.64%
[INFO] Preprocessing: 500 Hz → 250 Hz, 8-30 Hz bandpass

[INFO] Collecting baseline data (10 seconds)...
Progress: ████████████████████ 100%

[INFO] Ready for motor imagery!

--- Classification Results ---
[2024-01-15 10:23:15] LEFT HAND  (confidence: 0.78)
[2024-01-15 10:23:19] LEFT HAND  (confidence: 0.81)
[2024-01-15 10:23:23] RIGHT HAND (confidence: 0.72)
[2024-01-15 10:23:27] RIGHT HAND (confidence: 0.85)
```

---

## Testing & Validation

### 1. Simulator Test (No Hardware)

```bash
# Test entire pipeline with synthetic data
python npg_realtime_bci.py --simulate

# Expected: Should run without errors and classify random motor imagery
```

### 2. Signal Quality Check

```bash
# Verify electrode connections
python electrode_placement_verifier.py

# Check for:
# - Low noise (<15 µV RMS)
# - Balanced signals across channels
# - Alpha rhythm visible (8-12 Hz)
```

### 3. Motor Imagery Test Protocol

**Left Hand Imagery** (5 trials):

1. Relax, look at fixation cross on screen
2. When cued, imagine squeezing left hand (4 seconds)
3. System should classify as "LEFT HAND" (>70% accuracy expected)

**Right Hand Imagery** (5 trials):

1. Relax, look at fixation cross
2. When cued, imagine squeezing right hand (4 seconds)
3. System should classify as "RIGHT HAND"

**Tips for Better Performance**:

- Use vivid, kinesthetic imagery (feel the movement, not visualize it)
- Maintain consistent imagery intensity
- Avoid actual muscle movement
- Stay relaxed between trials

### 4. Accuracy Validation

```bash
# Run full accuracy test with labeled trials
python test_npg_accuracy.py --trials 20

# Expected output:
# Accuracy: 72-78% (matches training performance)
# False positive rate: <15%
```

---

## Troubleshooting

### Issue: "No LSL stream found"

**Symptoms**:

```
[ERROR] No NPG Lite LSL stream found (type: EXG)
[ERROR] Make sure Chords-Python is running
```

**Solutions**:

1. Check Terminal 1: Is Chords-Python running?

   ```bash
   python -m chordspy.connection --protocol usb
   ```

2. Verify LSL streams are broadcasting:

   ```bash
   python -c "from pylsl import resolve_streams; print(resolve_streams())"
   ```

3. Check USB connection:

   - Windows: Device Manager → Ports (COM & LPT)
   - Linux: `ls /dev/ttyUSB*`

4. Try different protocol:
   ```bash
   python -m chordspy.connection --protocol wifi --ip YOUR_IP
   ```

---

### Issue: "Poor Signal Quality"

**Symptoms**:

```
[WARNING] C3: 45.2 µV RMS (HIGH NOISE)
[WARNING] Cz: 52.1 µV RMS (HIGH NOISE)
```

**Solutions**:

1. **Check electrode contact**:

   - Apply more conductive gel
   - Press electrodes firmly to scalp
   - Part hair at electrode sites

2. **Reduce electrical noise**:

   - Move away from power cables
   - Turn off nearby electronics
   - Use battery power if possible

3. **Improve skin preparation**:

   - Clean with alcohol
   - Abrade skin gently
   - Wait for gel to warm up (improves conductivity)

4. **Check ground electrode**:
   - Ensure ground is on bony mastoid
   - Ground should have lowest impedance

---

### Issue: "Random Classifications"

**Symptoms**:

```
Classifications alternate randomly, no correlation with imagery
```

**Solutions**:

1. **Verify motor imagery technique**:

   - Use kinesthetic (feeling) not visual imagery
   - Imagine squeezing, not just thinking about hand
   - Maintain imagery for full 4 seconds

2. **Check baseline calibration**:

   - Stay relaxed during 10-second baseline
   - Don't perform imagery during baseline

3. **Adjust classification threshold**:

   ```bash
   # More conservative (fewer false positives)
   python npg_realtime_bci.py --threshold 0.7

   # Less conservative (more detections)
   python npg_realtime_bci.py --threshold 0.5
   ```

4. **Collect more training data** (advanced):
   - Record your own motor imagery sessions
   - Retrain model with personal data

---

### Issue: "Connection Drops"

**Symptoms**:

```
[ERROR] LSL stream timeout
[WARNING] Data buffer empty
```

**Solutions**:

1. **USB Connection**:

   - Use high-quality USB cable (not charging-only cable)
   - Try different USB port
   - Avoid USB hubs

2. **WiFi Connection**:

   - Ensure strong signal (< 5 meters from router)
   - Use 2.4 GHz (better range than 5 GHz)
   - Reduce WiFi congestion (turn off other devices)

3. **Bluetooth Connection**:
   - Stay within 10 meters
   - Avoid interference from other BLE devices
   - USB is more reliable for stationary use

---

### Issue: "Model Load Error"

**Symptoms**:

```
[ERROR] Failed to load model: best_eegnet_2class_bci2b.keras
```

**Solutions**:

1. Check model file exists:

   ```bash
   ls models/best_eegnet_2class_bci2b.keras
   ```

2. Verify correct directory:

   ```bash
   cd CODE  # Must run from CODE directory
   ```

3. Re-download or retrain model:
   ```bash
   python train_model_2b.py  # Retrains from BCI Competition data
   ```

---

## Technical Details

### System Architecture

```
┌─────────────────┐
│  NPG Lite       │  3 channels (C3, Cz, C4)
│  Hardware       │  500 Hz, 12-bit ADC
└────────┬────────┘
         │ USB/WiFi/BLE
┌────────▼────────┐
│  Chords-Python  │  Hardware communication
│  (chordspy)     │  USB/WiFi/BLE protocols
└────────┬────────┘
         │ LSL Stream (EXG type)
┌────────▼────────┐
│  NPG Lite       │  LSL stream receiver
│  Adapter        │  (npg_lite_adapter.py)
└────────┬────────┘
         │ Raw EEG (500 Hz)
┌────────▼────────┐
│  Preprocessor   │  500→250 Hz resample
│                 │  8-30 Hz bandpass
│                 │  CAR, z-score
└────────┬────────┘
         │ Clean EEG (250 Hz, 1000 samples)
┌────────▼────────┐
│  EEGNet Model   │  CNN classification
│  (73.64% acc)   │  Left/Right hand
└────────┬────────┘
         │
┌────────▼────────┐
│  Classification │  "LEFT HAND" or
│  Output         │  "RIGHT HAND"
└─────────────────┘
```

### Data Flow

**Raw Data from NPG Lite**:

- Shape: `(n_samples, 3)` where n = time × 500 Hz
- Channels: `[C3, Cz, C4]`
- Units: Microvolts (µV)
- Typical range: -100 to +100 µV

**After Preprocessing**:

- Shape: `(1, 3, 1000, 1)` - Batch × Channels × Time × Features
- Sampling rate: 250 Hz
- Frequency range: 8-30 Hz (motor imagery relevant)
- Normalized: Mean=0, Std=1 per channel

**Model Output**:

- Shape: `(1, 1)` - Probability of "LEFT HAND"
- Range: 0.0 to 1.0
- Interpretation:
  - < 0.5: RIGHT HAND (1 - probability)
  - > 0.5: LEFT HAND (probability)
  - = 0.5: Uncertain

### Preprocessing Pipeline Details

```python
# 1. Resampling (500 Hz → 250 Hz)
# Uses scipy.signal.resample_poly with 1:2 ratio (very clean!)
resampled = resample_poly(data, up=1, down=2, axis=0)

# 2. Bandpass Filter (8-30 Hz, 4th-order Butterworth)
# Isolates mu (8-12 Hz) and beta (13-30 Hz) rhythms
filtered = signal.filtfilt(b, a, resampled, axis=0)

# 3. Common Average Reference (CAR)
# Removes common-mode noise
car_data = filtered - filtered.mean(axis=1, keepdims=True)

# 4. Z-score Normalization
# Per-channel standardization
normalized = (car_data - mean) / std

# 5. Epoch Extraction
# 4-second windows (1000 samples @ 250 Hz)
epoch = normalized[-1000:, :]  # Last 4 seconds
```

### Model Architecture

**EEGNet (Compact CNN for EEG)**:

```
Input: (3, 1000, 1) - 3 channels × 1000 timepoints

Block 1: Temporal Convolution
- Conv2D: 8 filters, kernel (1, 64), captures frequency features

Block 2: Depthwise Spatial Convolution
- DepthwiseConv2D: (3, 1), learns spatial filters per channel
- Captures C3, Cz, C4 relationships

Block 3: Separable Convolution
- SeparableConv2D: 16 filters, (1, 16)
- Extracts higher-level temporal patterns

Block 4: Classification
- Flatten → Dense(2) → Softmax
- Output: [p_right_hand, p_left_hand]

Total Parameters: ~2,500
Trained on: BCI Competition IV 2b (9 subjects, 3 channels)
Accuracy: 73.64% on held-out test set
```

### Performance Characteristics

**Latency** (end-to-end):

- Data acquisition: 4000 ms (4-second epoch)
- Preprocessing: ~50 ms
- Model inference: ~10 ms
- **Total: ~4060 ms** (acceptable for non-real-time control)

**Accuracy**:

- Training set: 85% (some overfitting)
- Validation set: 76%
- Test set: **73.64%** (realistic performance)
- Expected with your data: 70-78% (individual variability)

**Resource Usage**:

- CPU: ~15% (single core, preprocessing + inference)
- RAM: ~500 MB (model + buffer)
- GPU: Optional, minimal benefit for single-epoch inference

---

## Advanced Topics

### Recording Your Own Training Data

```bash
# Start data recording session
python record_motor_imagery.py --subject YOUR_NAME --trials 40

# Follow on-screen protocol:
# - 20 left hand trials
# - 20 right hand trials
# - Random order with rest periods

# Retrain model with your data
python train_model_2b.py --user-data recordings/YOUR_NAME.npz

# Test personalized model
python npg_realtime_bci.py --model models/YOUR_NAME_model.keras
```

### Integrating with External Devices

```python
# Example: Control robot hand via serial
from npg_realtime_bci import NPGRealtimeBCI
import serial

bci = NPGRealtimeBCI()
robot = serial.Serial('/dev/ttyUSB1', 9600)

while True:
    prediction = bci.get_latest_classification()

    if prediction == "LEFT HAND":
        robot.write(b'LEFT\n')
    elif prediction == "RIGHT HAND":
        robot.write(b'RIGHT\n')
```

### Custom Preprocessing

```python
# Modify preprocessing pipeline
from npg_preprocessor import NPGPreprocessor

# Custom frequency band (e.g., focus on beta rhythm)
preprocessor = NPGPreprocessor(
    filter_low=13.0,  # Beta band only
    filter_high=30.0
)

# Custom resampling (keep original 500 Hz)
preprocessor = NPGPreprocessor(
    input_rate=500,
    output_rate=500  # No resampling
)
```

---

## Support & Resources

### Official Documentation

- **Chords-Python GitHub**: https://github.com/upsidedownlabs/Chords-Python
- **NPG Lite Product Page**: https://upsidedownlabs.tech/
- **BCI Competition IV Dataset**: https://www.bbci.de/competition/iv/

### Troubleshooting Help

- Check `logs/` directory for detailed error logs
- Run with `--verbose` flag for debug output
- Create GitHub issue with log files

### Community

- Upside Down Labs Discord (for NPG Lite hardware)
- BCI subreddit: r/BCI
- NeuroTech community forums

### Citation

If you use this system in research, please cite:

```
BCI Competition IV Dataset 2b
Leeb, R., Brunner, C., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008)
```

---

## FAQ

**Q: Can I use other electrode positions?**  
A: The model is trained specifically for C3, Cz, C4. Other positions would require retraining.

**Q: Why only left/right hand? Can I detect feet or tongue?**  
A: The model is binary (left/right hand only). Multi-class requires retraining with BCI Competition IV 2a dataset (4 classes).

**Q: How can I improve accuracy?**  
A: 1) Practice motor imagery technique, 2) Optimize electrode placement, 3) Collect personal training data.

**Q: Can I use this for real-time control?**  
A: Yes, but note ~4 second latency. Best for discrete commands (e.g., select left/right), not continuous control.

**Q: Do I need the Beast Pack or base NPG Lite?**  
A: Either works! Beast Pack includes more accessories, but base NPG Lite has same 3 channels.

**Q: Can I use dry electrodes?**  
A: Gel electrodes recommended for best signal quality. Dry electrodes may work but expect lower accuracy.

---

## Changelog

**Version 1.0** (Current)

- Initial release
- NPG Lite integration via Chords-Python
- 3-channel motor imagery (left/right hand)
- 73.64% test accuracy
- Simulator mode for testing

**Planned Features**

- [ ] Web interface for real-time visualization
- [ ] Personalized model training wizard
- [ ] Multi-class classification (4-class motor imagery)
- [ ] Continuous control mode (sliding window)
- [ ] Mobile app for NPG Lite configuration

---

## License

This integration code is open-source (MIT License).  
Chords-Python is developed by Upside Down Labs (check their license).  
BCI Competition data has specific usage terms (research/educational).

---

**Ready to start?** Run the simulator test:

```bash
cd CODE
python npg_realtime_bci.py --simulate
```

Good luck with your BCI journey! 🧠⚡
