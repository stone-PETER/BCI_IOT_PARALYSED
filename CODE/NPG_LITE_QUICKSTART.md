# NPG Lite BCI - Quick Start

## 🎯 What Was Fixed

The NPG Lite integration was completely **rewritten** to use the correct library:

- ❌ **Old**: Neurosity SDK (wrong company, wrong hardware)
- ✅ **New**: Chords-Python (official Upside Down Labs library)

## ⚡ Quick Test (30 seconds)

```bash
cd CODE
python npg_realtime_bci.py --simulate
```

You should see motor imagery classifications (LEFT/RIGHT HAND) from simulated EEG.

## 📦 Installation

```bash
# Install required packages
cd CODE
pip install -r requirements.txt

# Verify installation
python -c "import chordspy; print('✓ Chords-Python installed')"
python -c "import pylsl; print('✓ LSL installed')"
```

## 🔌 Hardware Usage (NPG Lite)

### Two-Terminal Setup

**Terminal 1** - Connect NPG Lite:

```bash
cd CODE
python -m chordspy.connection --protocol usb
```

_Leave this running - it streams EEG to LSL_

**Terminal 2** - Run BCI:

```bash
cd CODE
python npg_realtime_bci.py
```

### Expected Output

**Terminal 1** (Chords-Python):

```
Connected to NPG Lite
Streaming EEG data to LSL
Stream: BioAmpDataStream (500 Hz, 3 channels)
```

**Terminal 2** (BCI System):

```
=== NPG Lite Real-Time BCI ===
[INFO] Found LSL stream: BioAmpDataStream
[INFO] Connected: 3 channels @ 500 Hz
[INFO] Model loaded: 73.64% accuracy
[INFO] Ready for motor imagery!

--- Classifications ---
[10:23:15] LEFT HAND  (0.78)
[10:23:19] RIGHT HAND (0.85)
```

## 🧠 Motor Imagery Instructions

1. **Relax**: Sit comfortably, minimize muscle tension
2. **Cue**: Wait for system cue (or just start)
3. **Imagine**:
   - **Left hand**: Feel squeezing/clenching left fist (don't move!)
   - **Right hand**: Feel squeezing/clenching right fist
4. **Duration**: Hold imagery for 4 seconds
5. **Rest**: Relax 2-3 seconds between trials

**Tips**:

- Use kinesthetic (feeling) not visual imagery
- Imagine the sensation, not the movement
- Stay consistent in intensity
- Avoid actual muscle movement

## 📊 What Changed

| Aspect       | Old (Wrong)    | New (Correct)    |
| ------------ | -------------- | ---------------- |
| Library      | Neurosity SDK  | Chords-Python    |
| Manufacturer | Neurosity      | Upside Down Labs |
| Channels     | 6 (assumed)    | 3 (C3, Cz, C4)   |
| Sampling     | 256 Hz         | 500 Hz           |
| Connection   | Direct serial  | LSL streaming    |
| Auth         | Email/password | None needed      |

## ✅ Testing Checklist

- [ ] Simulator works: `python npg_realtime_bci.py --simulate`
- [ ] Dependencies installed: `pip install chordspy pylsl`
- [ ] Hardware connected (if available)
- [ ] Chords-Python running in Terminal 1
- [ ] BCI system finds LSL stream
- [ ] Classifications appear during motor imagery

## 📚 Documentation

- **Full guide**: [NPG_LITE_USER_GUIDE.md](NPG_LITE_USER_GUIDE.md)

  - Electrode placement (C3, Cz, C4 positions)
  - Troubleshooting guide
  - Technical details

- **What was fixed**: [NPG_LITE_CORRECTION_SUMMARY.md](NPG_LITE_CORRECTION_SUMMARY.md)
  - Detailed comparison of old vs new
  - Architecture explanation

## 🐛 Common Issues

### "No LSL stream found"

→ Start Terminal 1: `python -m chordspy.connection --protocol usb`

### "Import error: chordspy"

→ Install: `pip install chordspy pylsl`

### "Random classifications"

→ Practice motor imagery technique (kinesthetic, not visual)

### "Poor signal quality"

→ Run: `python electrode_placement_verifier.py`

## 🎓 Technical Details

**NPG Lite Specs**:

- 3 channels: C3 (left), Cz (center), C4 (right)
- 500 Hz sampling rate
- USB/WiFi/Bluetooth connectivity

**Our Model**:

- EEGNet architecture
- Trained on BCI Competition IV 2b
- 73.64% test accuracy
- Perfect match for 3 channels!

**Pipeline**:

```
NPG Lite (500 Hz, 3ch)
  → Chords-Python
    → LSL Stream
      → Preprocessor (500→250 Hz)
        → EEGNet Model
          → LEFT/RIGHT classification
```

## 🚀 Next Steps

1. **Test simulator**: Verify software works
2. **Install packages**: `pip install chordspy pylsl`
3. **Connect hardware**: Follow two-terminal setup
4. **Practice imagery**: Use kinesthetic imagination
5. **Check accuracy**: Should be 70-78% (like training data)

## 📞 Support

- **Chords-Python**: https://github.com/upsidedownlabs/Chords-Python
- **NPG Lite**: https://upsidedownlabs.tech/
- **User Guide**: [NPG_LITE_USER_GUIDE.md](NPG_LITE_USER_GUIDE.md)

---

**Ready?** Start with simulator:

```bash
cd CODE
python npg_realtime_bci.py --simulate
```

Good luck! 🧠⚡
