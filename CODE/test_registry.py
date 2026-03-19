#!/usr/bin/env python3
from model_factory import ModelFactory

try:
    info = ModelFactory.load_from_registry('alan')
    print('✅ Model loaded from registry!')
    print(f'Path: {info["path"]}')
    print(f'Threshold: {info["neutral_threshold"]}')
    print(f'Accuracy: {info["metadata"].get("val_accuracy", "N/A")}')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
