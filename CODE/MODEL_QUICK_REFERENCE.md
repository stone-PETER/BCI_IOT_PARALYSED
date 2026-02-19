# 🚀 Quick Reference: Model Configuration System

## ⚡ TL;DR - Switch models in 30 seconds

1. **Open any config file** (e.g., `config_2b.yaml`)
2. **Change one line:**
   ```yaml
   model:
     architecture: "eegnet"  # ← Change "eegnet" to "simplecnn"
   ```
3. **Run training:**
   ```bash
   python train_model_2b.py config_2b.yaml
   ```

**Done!** No code changes needed.

---

## 📋 Available Models

| Model | Architecture | Parameters | Best For |
|-------|-------------|------------|----------|
| **eegnet** | EEGNet (compact CNN) | ~2,226 | Limited data, fast training |
| **simplecnn** | Simple baseline CNN | ~779,330 | Baseline comparison, prototyping |

---

## 🔥 Common Use Cases

### Use Case 1: Test Different Models Quickly

```bash
# Test EEGNet
python train_model_2b.py config_2b.yaml

# Change config to SimpleCNN
# (edit config_2b.yaml, change architecture: "simplecnn")
python train_model_2b.py config_2b.yaml
```

### Use Case 2: Run Multiple Models in Parallel

```bash
# Terminal 1: Train EEGNet
python train_model_2b.py config_2b.yaml

# Terminal 2: Train SimpleCNN  
python train_model_2b.py config_simplecnn.yaml

# Terminal 3: Train with optimization settings
python train_model_2b.py config_opt2.yaml
```

Compare results in `logs/` folder!

### Use Case 3: Hyperparameter Tuning with Different Models

Create configs:
- `config_eegnet_opt1.yaml` - architecture: "eegnet"
- `config_eegnet_opt2.yaml` - architecture: "eegnet"
- `config_simplecnn_opt1.yaml` - architecture: "simplecnn"

Run systematic comparison!

---

## 🛠️ Config File Templates

### Template 1: EEGNet Configuration

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

### Template 2: SimpleCNN Configuration

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

---

## ➕ Add Your Own Model (3 Steps)

### Step 1: Create model file `my_model.py`

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
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
        # Your architecture here
        output = Dense(self.model_config['nb_classes'], 
                      activation='softmax')(input_layer)
        self.model = Model(inputs=input_layer, outputs=output)
        return self.model
    
    def compile_model(self, optimizer='adam', learning_rate=0.001):
        # Compile logic here
        pass
```

### Step 2: Register in `model_factory.py`

```python
from my_model import MyModel

AVAILABLE_MODELS = {
    'eegnet': EEGNet,
    'simplecnn': SimpleCNN,
    'mymodel': MyModel,  # ← Add this line
}
```

### Step 3: Use it!

```yaml
model:
  architecture: "mymodel"  # ← That's it!
```

---

## 🧪 Testing

### Test Model Factory

```bash
python test_model_factory.py
```

Should show:
```
✓ PASS: List Models
✓ PASS: EEGNet Creation
✓ PASS: SimpleCNN Creation
✓ PASS: Model Inference
✓ PASS: Model Switching
🎉 ALL TESTS PASSED!
```

### Test Specific Model

```python
from model_factory import create_model_from_config

# Create model
model = create_model_from_config('config_2b.yaml')

# Show summary
model.model.summary()
```

---

## 📊 Compare Models

### Method 1: Check Logs

```bash
# After training multiple models
ls ../logs/evaluation_results_*.json

# View results
cat ../logs/evaluation_results_2b_*.json | findstr accuracy
```

### Method 2: Best Accuracy Tracker

```bash
# Check best model ever
cat models/best_accuracy.json
```

Output:
```json
{
  "best_accuracy": 0.6825,
  "timestamp": "20260213_180950",
  "model_filename": "eegnet_2class_bci2b_20260213_180950.keras"
}
```

---

## 🎯 Tips & Best Practices

✅ **DO:**
- Keep separate config files for different models
- Use descriptive config names: `config_eegnet_opt1.yaml`
- Run tests after adding new models
- Compare models systematically
- Document your custom models

❌ **DON'T:**
- Don't modify `model_factory.py` frequently (register once)
- Don't hardcode model choices in training scripts
- Don't forget to test new models before training

---

## 🐛 Troubleshooting

### Error: "Model type 'xyz' not found"

**Fix:** Register model in `model_factory.py`:
```python
AVAILABLE_MODELS = {
    'xyz': YourModelClass,  # Add this
}
```

### Error: Missing config parameter

**Fix:** Add parameter to config:
```yaml
model:
  architecture: "yourmodel"
  missing_param: value  # Add this
```

### Model not training

**Fix:** Check model has these methods:
- `__init__(config_path)`
- `build_model()`
- `compile_model(optimizer, learning_rate)`

---

## 📁 File Structure

```
CODE/
├── model_factory.py          # ← Model factory (register here)
├── eegnet_model.py           # EEGNet implementation
├── simple_cnn_model.py       # SimpleCNN implementation
├── train_model_2b.py         # Training script (uses factory)
├── config_2b.yaml            # Default config (EEGNet)
├── config_simplecnn.yaml     # SimpleCNN config
├── config_opt*.yaml          # Optimization configs
└── test_model_factory.py     # Test suite
```

---

## 🔗 Full Documentation

See **MODEL_CONFIGURATION_GUIDE.md** for:
- Detailed architecture explanations
- Advanced usage patterns
- More examples
- Model implementation details

---

## ✅ Checklist: Add New Model

- [ ] Create model class file (`my_model.py`)
- [ ] Implement `__init__`, `build_model`, `compile_model`
- [ ] Import in `model_factory.py`
- [ ] Add to `AVAILABLE_MODELS` dictionary
- [ ] Create config file (`config_mymodel.yaml`)
- [ ] Set `architecture: "mymodel"`
- [ ] Run `test_model_factory.py`
- [ ] Train and compare: `python train_model_2b.py config_mymodel.yaml`

---

**Made a change? Test it:**
```bash
python test_model_factory.py
```

**Ready to train:**
```bash
python train_model_2b.py config_2b.yaml
```

🎉 **Happy Model Building!**
