"""
Quick Training Test with Improvements
Test the enhanced training pipeline with data augmentation
"""

import os
import sys
from pathlib import Path

# Add CODE directory to path
code_dir = Path(__file__).parent / "CODE"
sys.path.insert(0, str(code_dir))

from train_model_2b import EEGNetTrainer2B


def test_augmentation():
    """Test data augmentation functionality."""
    print("🧪 Testing Data Augmentation")
    print("=" * 50)
    
    try:
        # Test with augmentation enabled
        trainer = EEGNetTrainer2B("config_2b.yaml")
        
        print(f"✅ Augmentation enabled: {trainer.data_loader.augmentation_config.get('enabled', False)}")
        print(f"📈 Augmentation factor: {trainer.data_loader.augmentation_config.get('augmentation_factor', 0)}")
        
        # Load a small sample to test augmentation
        print("\n🔄 Loading sample data...")
        data_splits = trainer.load_and_prepare_data()
        
        print(f"📊 Data shapes after loading:")
        print(f"  Training: {data_splits['X_train'].shape}")
        print(f"  Validation: {data_splits['X_val'].shape}")
        print(f"  Test: {data_splits['X_test'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing augmentation: {e}")
        return False


def quick_train_test():
    """Run a quick training test with 5 epochs."""
    print("\n🚀 Quick Training Test (5 epochs)")
    print("=" * 50)
    
    try:
        # Create temporary config with few epochs
        import yaml
        from tempfile import NamedTemporaryFile
        
        # Load base config
        with open("config_2b.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Modify for quick test
        config['training']['epochs'] = 5
        config['training']['early_stopping']['patience'] = 10
        config['augmentation']['augmentation_factor'] = 1  # Double the data
        
        # Write temporary config
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(config, tmp_file, indent=2)
            temp_config_path = tmp_file.name
        
        try:
            # Run training
            trainer = EEGNetTrainer2B(temp_config_path)
            results = trainer.run_complete_training()
            
            print(f"\n✅ Quick training completed!")
            print(f"📊 Test Accuracy: {results['test_accuracy']:.4f}")
            
            if results.get('is_new_best', False):
                print(f"🏆 NEW BEST ACCURACY!")
            else:
                current_best = results.get('current_best', 0.0)
                print(f"📈 Current best: {current_best:.4f}")
            
            return True
            
        finally:
            # Clean up temp file
            os.unlink(temp_config_path)
            
    except Exception as e:
        print(f"❌ Error in quick training: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_improvement_test():
    """Run comprehensive improvement test."""
    print("🔬 BCI Model Improvement Test")
    print("=" * 70)
    
    # Test 1: Augmentation functionality
    aug_success = test_augmentation()
    
    if aug_success:
        print("\n" + "="*50)
        print("🎯 Ready for Training!")
        print("="*50)
        print("\nNext steps:")
        print("1. Run hyperparameter sweep: python hyperparameter_sweep.py")
        print("2. Run quick training test below")
        print("3. Run full training: python CODE/train_model_2b.py")
        
        # Ask if user wants to run quick test
        response = input("\n🤔 Run quick training test (5 epochs)? (y/N): ")
        if response.lower() == 'y':
            quick_train_test()
        else:
            print("💡 Skipping quick test. Run full training when ready!")
    
    else:
        print("\n❌ Setup incomplete. Check error messages above.")


if __name__ == "__main__":
    run_improvement_test()