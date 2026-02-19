"""Quick demo of model configuration system"""
from model_factory import list_available_models

print("\n" + "="*60)
print("🎯 YOUR MODEL CONFIGURATION SYSTEM IS READY!")
print("="*60)

available = list_available_models()

print("\n📋 HOW TO USE:")
print("="*60)
print("\n1️⃣  SWITCH MODELS - Edit any config file:")
print("   ")
print("   model:")
print("     architecture: \"eegnet\"  ← Change this!")
print()
print("2️⃣  AVAILABLE OPTIONS:")
for model in available:
    print(f"   - {model}")
print()
print("3️⃣  RUN TRAINING:")
print("   python train_model_2b.py config_2b.yaml")
print()
print("="*60)
print("✅ NO CODE CHANGES NEEDED!")
print("="*60)
print()
