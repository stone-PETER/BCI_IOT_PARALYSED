"""
Test Script: Model Factory Functionality
Verifies that the model configuration system works correctly
"""

import sys
import numpy as np
from pathlib import Path

# Add CODE directory to path
code_dir = Path(__file__).parent
sys.path.insert(0, str(code_dir))

from model_factory import ModelFactory, list_available_models


def test_available_models():
    """Test listing available models."""
    print("\n" + "=" * 60)
    print("TEST 1: List Available Models")
    print("=" * 60)
    
    models = list_available_models()
    assert len(models) > 0, "No models available!"
    assert 'eegnet' in models, "EEGNet should be available"
    print("✓ Available models test passed\n")
    return models


def test_eegnet_creation():
    """Test creating EEGNet model."""
    print("=" * 60)
    print("TEST 2: Create EEGNet Model")
    print("=" * 60)
    
    try:
        factory = ModelFactory('config_2b.yaml')
        model_instance = factory.create_model(compile_model=False)
        
        assert model_instance is not None, "Model instance is None"
        assert hasattr(model_instance, 'model'), "Model instance has no 'model' attribute"
        assert model_instance.model is not None, "Keras model is None"
        
        # Check model shape
        input_shape = model_instance.model.input_shape
        output_shape = model_instance.model.output_shape
        
        print(f"✓ EEGNet model created successfully")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {output_shape}")
        print(f"  Total parameters: {model_instance.model.count_params():,}\n")
        
        return True
    except Exception as e:
        print(f"✗ EEGNet creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simplecnn_creation():
    """Test creating SimpleCNN model."""
    print("=" * 60)
    print("TEST 3: Create SimpleCNN Model")
    print("=" * 60)
    
    try:
        factory = ModelFactory('config_simplecnn.yaml')
        model_instance = factory.create_model(compile_model=False)
        
        assert model_instance is not None, "Model instance is None"
        assert hasattr(model_instance, 'model'), "Model instance has no 'model' attribute"
        assert model_instance.model is not None, "Keras model is None"
        
        # Check model shape
        input_shape = model_instance.model.input_shape
        output_shape = model_instance.model.output_shape
        
        print(f"✓ SimpleCNN model created successfully")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {output_shape}")
        print(f"  Total parameters: {model_instance.model.count_params():,}\n")
        
        return True
    except Exception as e:
        print(f"✗ SimpleCNN creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_inference():
    """Test model inference with dummy data."""
    print("=" * 60)
    print("TEST 4: Model Inference Test")
    print("=" * 60)
    
    try:
        # Create EEGNet model
        factory = ModelFactory('config_2b.yaml')
        model_instance = factory.create_model(compile_model=True)
        
        # Generate dummy data
        dummy_input = np.random.randn(8, 3, 1000, 1).astype(np.float32)
        
        # Run inference
        predictions = model_instance.model.predict(dummy_input, verbose=0)
        
        assert predictions.shape == (8, 2), f"Unexpected output shape: {predictions.shape}"
        assert np.allclose(predictions.sum(axis=1), 1.0, rtol=1e-5), "Outputs should sum to 1"
        
        print(f"✓ Model inference test passed")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Sample prediction: {predictions[0]}")
        print(f"  Prediction sum: {predictions[0].sum():.6f}\n")
        
        return True
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_switching():
    """Test switching between models via config."""
    print("=" * 60)
    print("TEST 5: Model Switching Test")
    print("=" * 60)
    
    try:
        # Test EEGNet
        print("\n  Creating EEGNet from config_2b.yaml...")
        factory1 = ModelFactory('config_2b.yaml')
        model1 = factory1.create_model(compile_model=False)
        params1 = model1.model.count_params()
        print(f"  ✓ EEGNet: {params1:,} parameters")
        
        # Test SimpleCNN
        print("\n  Creating SimpleCNN from config_simplecnn.yaml...")
        factory2 = ModelFactory('config_simplecnn.yaml')
        model2 = factory2.create_model(compile_model=False)
        params2 = model2.model.count_params()
        print(f"  ✓ SimpleCNN: {params2:,} parameters")
        
        print(f"\n✓ Model switching test passed")
        print(f"  Models have different architectures: {params1 != params2}\n")
        
        return True
    except Exception as e:
        print(f"✗ Model switching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  MODEL FACTORY TEST SUITE".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    
    results = []
    
    # Run tests
    try:
        test_available_models()
        results.append(("List Models", True))
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        results.append(("List Models", False))
    
    try:
        result = test_eegnet_creation()
        results.append(("EEGNet Creation", result))
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        results.append(("EEGNet Creation", False))
    
    try:
        result = test_simplecnn_creation()
        results.append(("SimpleCNN Creation", result))
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        results.append(("SimpleCNN Creation", False))
    
    try:
        result = test_model_inference()
        results.append(("Model Inference", result))
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        results.append(("Model Inference", False))
    
    try:
        result = test_model_switching()
        results.append(("Model Switching", result))
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        results.append(("Model Switching", False))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nYour model factory is working correctly!")
        print("\nNext steps:")
        print("1. Edit any config file and change 'architecture' parameter")
        print("2. Run: python train_model_2b.py <config_file>")
        print("3. Compare different models easily!\n")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Please check the error messages above.\n")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
