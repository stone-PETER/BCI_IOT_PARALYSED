# BCI Model Configuration Guide

## Easy Model Switching and Testing

This guide explains how to use the flexible model configuration system to easily switch between different neural network architectures for BCI classification.

---

## 🎯 Quick Start

### Switching Models in 3 Steps:

1. **Edit config file** (e.g., `config_2b.yaml`)
2. **Change the architecture parameter**
3. **Run training as usual**

```yaml
model:
  architecture: "eegnet" # Change this to switch models!
  chans: 3
  samples: 1000
  nb_classes: 2
  # ... other parameters
```

**That's it!** No code changes needed.

---

## 📋 Available Models

### Currently Implemented:

1. **eegnet** - EEGNet architecture (default)
   - Compact CNN for EEG data
   - Best for: Limited data, fast training
   - Parameters: ~2,226 for 3-channel binary

2. **Add your own models below** (see "Adding New Models" section)

---

## 🔧 Configuration Files

### Main Config Files:

| File               | Purpose                          | Model Type                |
| ------------------ | -------------------------------- | ------------------------- |
| `config_2b.yaml`   | BCI Competition IV 2b (binary)   | 3-channel, 2-class        |
| `config.yaml`      | BCI Competition III 3a (4-class) | 22-channel, 4-class       |
| `config_opt1.yaml` | Optimization config 1            | High LR + More Filters    |
| `config_opt2.yaml` | Optimization config 2            | Long Kernel + Low Dropout |
| `config_opt3.yaml` | Optimization config 3            | Aggressive + Augmentation |

### Config Structure:

```yaml
model:
  # === CHANGE THIS TO SWITCH MODELS ===
  architecture: "eegnet" # Options: eegnet, deepconvnet, shallowconvnet, etc.

  # Model parameters (architecture-specific)
  chans: 3 # Number of EEG channels
  samples: 1000 # Samples per epoch
  nb_classes: 2 # Number of output classes

  # EEGNet-specific parameters
  kernLength: 64
  F1: 8
  D: 2
  F2: 16
  dropoutRate: 0.5
  dropoutType: "Dropout"
  norm_rate: 0.25

training:
  batch_size: 32
  epochs: 300
  learning_rate: 0.001
  optimizer: "adam"
  # ... more training parameters
```

---

## 🚀 Usage Examples

### Example 1: Train with Default EEGNet

```python
python train_model_2b.py config_2b.yaml
```

### Example 2: Train with Different Model

1. Edit `config_2b.yaml`:

   ```yaml
   model:
     architecture: "deepconvnet" # Changed from "eegnet"
   ```

2. Run training:
   ```python
   python train_model_2b.py config_2b.yaml
   ```

### Example 3: Quick Model Comparison

Create multiple configs for A/B testing:

**config_eegnet.yaml:**

```yaml
model:
  architecture: "eegnet"
  # ... parameters
```

**config_deepconv.yaml:**

```yaml
model:
  architecture: "deepconvnet"
  # ... parameters
```

**Run both:**

```bash
python train_model_2b.py config_eegnet.yaml
python train_model_2b.py config_deepconv.yaml
```

Compare results automatically saved in `logs/` and `models/`!

### Example 4: Using Model Factory Directly

```python
from model_factory import ModelFactory

# Create model from config
factory = ModelFactory('config_2b.yaml')
model_instance = factory.create_model(compile_model=True)

# Access the Keras model
keras_model = model_instance.model
keras_model.summary()

# Train as usual
history = keras_model.fit(X_train, y_train, ...)
```

---

## ➕ Adding New Models

### Step 1: Create Your Model Class

Create a new file `your_model.py`:

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
import yaml

class YourModel:
    """Your custom BCI model."""

    def __init__(self, config_path: str = "config_2b.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.model_config = self.config['model']
        self.nb_classes = self.model_config['nb_classes']
        self.chans = self.model_config['chans']
        self.samples = self.model_config['samples']

        self.model = None

    def build_model(self) -> Model:
        """Build your model architecture."""
        # Input shape: (channels, samples, 1)
        input_layer = Input(shape=(self.chans, self.samples, 1))

        # Your architecture here
        x = Conv2D(32, (1, 64), activation='relu')(input_layer)
        x = Flatten()(x)
        output = Dense(self.nb_classes, activation='softmax')(x)

        # Create model
        self.model = Model(inputs=input_layer, outputs=output)
        return self.model

    def compile_model(self, optimizer='adam', learning_rate=0.001):
        """Compile the model."""
        from tensorflow.keras.optimizers import Adam

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
```

### Step 2: Register Your Model

Edit `model_factory.py`:

```python
# At the top - import your model
from your_model import YourModel

# In AVAILABLE_MODELS dictionary
AVAILABLE_MODELS = {
    'eegnet': EEGNet,
    'yourmodel': YourModel,  # Add this line
}
```

### Step 3: Use Your Model

Edit any config file:

```yaml
model:
  architecture: "yourmodel" # That's it!
  chans: 3
  samples: 1000
  nb_classes: 2
  # Add any custom parameters your model needs
```

Run training:

```bash
python train_model_2b.py config_2b.yaml
```

---

## 📊 Model Comparison Workflow

### Best Practice for Testing New Models:

1. **Create separate configs** for each model:
   - `config_eegnet.yaml`
   - `config_deepconv.yaml`
   - `config_custom.yaml`

2. **Run systematic tests:**

   ```bash
   # Test all models
   python train_model_2b.py config_eegnet.yaml
   python train_model_2b.py config_deepconv.yaml
   python train_model_2b.py config_custom.yaml
   ```

3. **Compare results:**
   - Check `logs/evaluation_results_*.json`
   - Review `models/best_accuracy.json`
   - Analyze training curves in TensorBoard

4. **Select best model:**
   - Update your production config with winner
   - Deploy with confidence!

---

## 🔍 Troubleshooting

### Error: "Model type 'xyz' not found"

**Solution:** Make sure your model is registered in `model_factory.py`:

```python
AVAILABLE_MODELS = {
    'eegnet': EEGNet,
    'xyz': YourModelClass,  # Add this
}
```

### Error: "Missing parameter in config"

**Solution:** Check that all required parameters are in your config file:

```yaml
model:
  architecture: "eegnet"
  chans: 3 # Required
  samples: 1000 # Required
  nb_classes: 2 # Required
  # ... other parameters
```

### Model not training properly

**Solution:** Verify your model class has these methods:

- `__init__(config_path)`
- `build_model()`
- `compile_model(optimizer, learning_rate)`

---

## 📚 Advanced Usage

### List Available Models

```python
from model_factory import list_available_models

# See all registered models
models = list_available_models()
```

### Dynamic Model Registration

```python
from model_factory import ModelFactory

# Register a new model at runtime
ModelFactory.add_model('experimental', ExperimentalModel)
```

### Custom Model Parameters

Your model can use custom parameters from config:

```yaml
model:
  architecture: "custom"
  # Standard params
  chans: 3
  samples: 1000
  nb_classes: 2

  # Your custom params
  custom_param1: 128
  custom_param2: "relu"
  use_attention: true
```

Access in your model:

```python
def __init__(self, config_path):
    # ...
    self.custom_param1 = self.model_config.get('custom_param1', 64)
    self.use_attention = self.model_config.get('use_attention', False)
```

---

## 🎓 Example Models to Try

### Popular BCI Architectures:

1. **DeepConvNet** - Deep convolutional network
   - Best for: Large datasets, complex patterns
   - Deeper architecture with more parameters

2. **ShallowConvNet** - Shallow convolutional network
   - Best for: Simple patterns, frequency features
   - Fast training, fewer parameters

3. **EEG-Inception** - Inception-style architecture
   - Best for: Multi-scale temporal features
   - Parallel convolutional paths

4. **EEGNet-Attention** - EEGNet + attention mechanism
   - Best for: Channel importance, interpretability
   - Attention weights show important channels

### Implementing these is straightforward using the factory pattern!

---

## 🎯 Summary

**Benefits of Model Factory System:**

- ✅ Switch models by editing config only
- ✅ No code changes needed for model testing
- ✅ Easy A/B testing and comparison
- ✅ Systematic hyperparameter optimization
- ✅ Clean separation of concerns
- ✅ Easy to add new models
- ✅ Backwards compatible with existing code

**Key Files:**

- `model_factory.py` - Model creation engine
- `config_2b.yaml` - Configuration file
- `train_model_2b.py` - Training script (uses factory)
- `your_model.py` - Your custom models

**Key Parameter:**

```yaml
model:
  architecture: "eegnet" # ← Change this to switch models
```

---

## 📞 Need Help?

Check:

1. Model is registered in `AVAILABLE_MODELS` dictionary
2. Config has `architecture` parameter
3. Model class implements required methods
4. All required parameters are in config

For more examples, see the existing `eegnet_model.py` implementation!

---

**Happy Model Building! 🧠🚀**
