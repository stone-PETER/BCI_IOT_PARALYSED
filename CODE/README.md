# BCI Motor Imagery Classification System

## Overview

Complete **Brain-Computer Interface (BCI)** system for motor imagery classification using EEG signals. The system processes brain signals to classify motor imagery intentions (left hand vs right hand movement) for real-time control applications using the **NPG Lite** hardware by Upside Down Labs.

**Key Features**:

- ✅ Real-time EEG processing from **NPG Lite** hardware (Upside Down Labs)
- ✅ Pre-trained **EEGNet** model (73.64% accuracy on BCI Competition IV 2b)
- ✅ **3-channel** motor imagery classification (C3, Cz, C4)
- ✅ **Chords-Python + LSL** integration for hardware communication
- ✅ REST API for IoT control integration
- ✅ Simulator mode for testing without hardware
- ✅ **Leaky Accumulator** for stable command triggering (v2.0)
- ✅ **Baseline Calibration** to correct model bias (v2.0)
- ✅ **Virtual Bipolar Filtering** to match training data format (v2.1)
- ✅ **Flexible Model Configuration** — switch architectures via config file

## 📄 Documentation

- **[README.md](README.md)** — This file: overview, quick start, architecture, API reference
- **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)** — Full guide: hardware setup, signal processing, model configuration, troubleshooting

---

## 🚀 Quick Start

### Test Without Hardware (Simulator)

```bash
cd CODE
pip install -r requirements.txt

# Step 1: Run baseline calibration (first time only)
python npg_realtime_bci.py --simulate --calibrate

# Step 2: Run the BCI system
python npg_realtime_bci.py --simulate
```

### Use With NPG Lite Hardware

```bash
# Terminal 1: Start Chords-Python (streams EEG to LSL)
python -m chordspy.connection --protocol usb

# Terminal 2: Run BCI system
python npg_realtime_bci.py
```

### Motor Imagery Instructions

1. **Relax**: Sit comfortably, minimize muscle tension
2. **Imagine left hand**: Feel squeezing/clenching left fist (don't move!)
3. **Imagine right hand**: Feel squeezing/clenching right fist
4. **Duration**: Hold imagery for 4 seconds; rest 2-3 seconds between trials

> **Tip**: Use kinesthetic (feeling) imagery, not visual. Imagine the sensation, not the movement.

---

## 🧠 System Architecture

```
NPG Lite → Chords-Python → LSL Stream → Preprocessor → EEGNet → Classification → IoT Commands
```

### Signal Processing Pipeline (v2.1)

```
NPG Lite (500 Hz, 3ch: C3, Cz, C4)
    ↓
Anti-aliasing LPF @ 100 Hz
    ↓
Resample 500 → 250 Hz (1:2)
    ↓
Notch Filter @ 50 Hz
    ↓
Virtual Bipolar (C3-Cz, Cz, C4-Cz)  ← matches training data format
    ↓
Bandpass 8-30 Hz (mu + beta)
    ↓
Z-score Normalize
    ↓
EEGNet Model
    ↓
Baseline Bias Correction
    ↓
Confidence-Weighted Smoothing (window=8)
    ↓
Leaky Accumulator (threshold=2.0, decay=0.15)
    ↓
Command: LEFT_HAND / RIGHT_HAND / UNCERTAIN
```

### Core Components

1. **NPG Lite Adapter** (`npg_lite_adapter.py`) - Receives 3-channel EEG via LSL from Chords-Python
2. **Preprocessing Module** (`npg_preprocessor.py`) - Signal processing: anti-aliasing, resample, notch, bipolar, bandpass, z-score
3. **EEGNet Model** (`eegnet_model.py`) - Deep learning classification (73.64% accuracy)
4. **Inference Engine** (`npg_inference.py`) - Real-time classification with leaky accumulator and baseline calibration
5. **Real-time BCI** (`npg_realtime_bci.py`) - Complete integration system
6. **API Server** (`api_server.py`) - REST API gateway for IoT control

---

## 📁 Project Structure

```
CODE/
├── # NPG Lite Integration
├── npg_lite_adapter.py              # LSL stream receiver for NPG Lite
├── npg_preprocessor.py              # Signal preprocessing (500→250 Hz, bipolar, notch)
├── npg_inference.py                 # Inference engine + leaky accumulator + calibration
├── npg_realtime_bci.py              # Complete real-time BCI system
├── electrode_placement_verifier.py  # Signal quality checker
├── calibrate_scaling.py             # Scaling factor calibration tool
├── baseline_bias.json               # Saved calibration (auto-created)
├──
├── # Configuration and Docs
├── config.yaml                      # System configuration
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── COMPLETE_GUIDE.md                # Full documentation
├──
├── # Core Training Modules
├── preprocessing.py                 # Original preprocessing (BCI datasets)
├── eegnet_model.py                  # EEGNet model architecture
├── simple_cnn_model.py              # SimpleCNN alternative model
├── model_factory.py                 # Model factory (switch models via config)
├── train_model.py                   # Model training pipeline
├── train_model_2b.py                # BCI Competition IV 2b trainer (3 channels)
├──
├── # BCI Competition Data Integration
├── bci4_2a_loader.py                # BCI IV 2a dataset (4 class)
├── gdf_data_loader.py               # GDF file loader
├── unified_bci_loader.py            # Unified dataset interface
├── dataset_preprocess.py            # Dataset preprocessing
├──
├── # Testing and Demo
├── demo.py                          # Original demo
├── test_api_client.py               # API testing client
├── validate_system.py               # System validation
├── test_4class_motor_imagery.py     # 4-class testing
├── test_model_factory.py            # Model factory test suite
├── test_baseline_calibration.py     # Calibration test
├──
├── # Web Interface
├── api_server.py                    # Flask REST API
├── motor_imagery_streamer.py        # Real-time streaming API
├──
└── # Generated (after training)
    ├── models/                      # Trained models
    │   └── best_eegnet_2class_bci2b.keras  # 73.64% accuracy (3 channels)
    ├── logs/                        # Training logs
    └── BCI/                         # BCI Competition datasets
```

---

## 🔧 Configuration

The system is configured via `config.yaml` (and `config_2b.yaml` for the 3-channel model).

### Switch Between Model Architectures (no code changes needed)

```yaml
# config_2b.yaml
model:
  architecture: "eegnet"  # Change to "simplecnn" to test alternative
  chans: 3
  samples: 1000
  nb_classes: 2
```

```bash
python train_model_2b.py config_2b.yaml
```

---

## 🔌 IoT Integration

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Server information |
| `/classify` | POST | Single epoch classification |
| `/classify_batch` | POST | Batch classification |
| `/status` | GET | System status |
| `/model_info` | GET | Model information |
| `/simulate` | POST | Simulation with stored data |

### Single Classification Example

```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"data": [[...]], "metadata": {"device_id": "bci_001"}}'
```

Response:

```json
{
  "command": "TURN_LEFT",
  "confidence": 0.87,
  "processing_time": 0.045,
  "timestamp": "2025-10-18T10:30:45.123456"
}
```

### MQTT Integration Example

```python
import paho.mqtt.client as mqtt
import requests

def classify_and_publish(eeg_data):
    response = requests.post("http://localhost:5000/classify",
                           json={"data": eeg_data.tolist()})
    if response.status_code == 200:
        result = response.json()
        client.publish("bci/commands", result["command"])
```

---

## 🛠️ Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| "No LSL stream found" | Chords-Python not running | Start Terminal 1: `python -m chordspy.connection --protocol usb` |
| "Import error: chordspy" | Missing dependencies | `pip install chordspy pylsl` |
| "Random classifications" | Poor motor imagery technique | Use kinesthetic (feeling) not visual imagery |
| "No commands triggering" | Not calibrated | Run `python npg_realtime_bci.py --simulate --calibrate` |
| "Predictions stuck at 50%" | Scaling or bipolar issue | Run `python calibrate_scaling.py`; verify `use_bipolar=True` |
| "Model load error" | Missing model file | `python train_model_2b.py` to retrain |

---

## 🧪 Testing

```bash
# Test system without hardware
python npg_realtime_bci.py --simulate

# Test model factory
python test_model_factory.py

# Test API endpoints
python api_server.py &
python test_api_client.py

# Validate electrode quality (hardware)
python electrode_placement_verifier.py
```

---

## 📊 Performance

| Metric | Value |
|---|---|
| Model accuracy (test set) | 73.64% |
| Expected real-world accuracy | 70-78% |
| Classification latency | ~4060 ms (4s epoch) |
| Preprocessing time | ~50 ms |
| Model inference time | ~10 ms |

---

## 📚 References

1. **EEGNet**: Lawhern et al. (2018). EEGNet: a compact convolutional neural network for EEG-based BCIs. *Journal of Neural Engineering*, 15(5).
2. **Motor Imagery**: Pfurtscheller & Neuper (2001). Motor imagery and direct brain-computer communication. *Proceedings of the IEEE*, 89(7).
3. **BCI Competition IV**: Tangermann et al. (2012). Review of the BCI competition IV. *Frontiers in Neuroscience*, 6.
4. **Chords-Python**: https://github.com/upsidedownlabs/Chords-Python
5. **NPG Lite**: https://upsidedownlabs.tech/

---

## 🔮 Future Enhancements

1. **Multi-class Classification** - Support for more motor imagery classes (feet, tongue)
2. **Adaptive Thresholding** - Auto-tune accumulator parameters
3. **Online Learning** - Personalized model adaptation in real-time
4. **Web Dashboard** - Real-time monitoring and visualization
5. **Model Quantization** - Edge deployment on microcontrollers

## 📄 License

This project is part of a BCI IoT system for paralyzed individuals. Please ensure ethical use and appropriate medical/safety considerations when implementing.

---

**Version**: 2.1 | **Last Updated**: 2026-02-03

For detailed hardware setup, signal processing explanations, and model configuration: see **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)**.
