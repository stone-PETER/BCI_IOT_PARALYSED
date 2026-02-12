"""
Hyperparameter Sweep for EEGNet Training
Systematically test different configurations to improve accuracy
"""

import itertools
import json
import yaml
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import os

# Add CODE directory to path
code_dir = Path(__file__).parent / "CODE"
sys.path.insert(0, str(code_dir))

from train_model_2b import EEGNetTrainer2B


class HyperparameterSweep:
    """Systematic hyperparameter optimization for EEGNet."""
    
    def __init__(self):
        self.base_config_path = "CODE/config_2b.yaml"
        self.results_dir = Path("hyperparameter_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load base config
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def define_search_space(self):
        """Define hyperparameter search space."""
        return {
            'learning_rate': [0.0005, 0.001, 0.002],
            'batch_size': [16, 32, 64],
            'epochs': [400, 500],
            'early_stopping_patience': [75, 100],
            'F1': [8, 16],  # First Conv layer filters
            'F2': [16, 32], # Second Conv layer filters
            'dropoutRate': [0.3, 0.4, 0.5, 0.6],
            'kernLength': [32, 64, 128],
        }
    
    def create_config_variant(self, params, run_id):
        """Create a config file variant with specific parameters."""
        config = self.base_config.copy()
        
        # Update training parameters
        config['training']['learning_rate'] = params['learning_rate']
        config['training']['batch_size'] = params['batch_size']
        config['training']['epochs'] = params['epochs']
        config['training']['early_stopping']['patience'] = params['early_stopping_patience']
        
        # Update model parameters
        config['model']['F1'] = params['F1']
        config['model']['F2'] = params['F2']
        config['model']['dropoutRate'] = params['dropoutRate']
        config['model']['kernLength'] = params['kernLength']
        
        # Update paths for this run
        config_file = f"config_sweep_{run_id}.yaml"
        config_path = Path("CODE") / config_file
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=2)
        
        return config_path
    
    def run_single_experiment(self, params, run_id):
        """Run a single training experiment."""
        print(f"\n{'='*60}")
        print(f"🔬 Running Experiment {run_id}")
        print(f"{'='*60}")
        
        # Create config for this run
        config_path = self.create_config_variant(params, run_id)
        
        try:
            # Initialize trainer with custom config
            trainer = EEGNetTrainer2B(str(config_path))
            
            # Run training
            results = trainer.run_complete_training()
            
            # Save results
            experiment_results = {
                'run_id': run_id,
                'parameters': params,
                'results': results,
                'timestamp': datetime.now().isoformat(),
                'config_path': str(config_path)
            }
            
            results_file = self.results_dir / f"experiment_{run_id}.json"
            with open(results_file, 'w') as f:
                json.dump(experiment_results, f, indent=2)
            
            # Clean up config file
            config_path.unlink()
            
            print(f"✅ Experiment {run_id} completed: {results['test_accuracy']:.4f}")
            return experiment_results
            
        except Exception as e:
            print(f"❌ Experiment {run_id} failed: {e}")
            # Clean up config file
            if config_path.exists():
                config_path.unlink()
            return None
    
    def run_random_search(self, n_experiments=20):
        """Run random search over hyperparameter space."""
        import random
        
        search_space = self.define_search_space()
        all_results = []
        
        print(f"🎯 Starting Random Search with {n_experiments} experiments")
        print(f"📁 Results will be saved to: {self.results_dir}")
        
        for i in range(n_experiments):
            # Sample random parameters
            params = {}
            for param_name, param_values in search_space.items():
                params[param_name] = random.choice(param_values)
            
            print(f"\n📋 Experiment {i+1}/{n_experiments} Parameters:")
            for k, v in params.items():
                print(f"   {k}: {v}")
            
            # Run experiment
            result = self.run_single_experiment(params, f"random_{i+1:03d}")
            if result:
                all_results.append(result)
        
        # Save summary
        self.save_summary(all_results, "random_search")
        return all_results
    
    def run_grid_search(self, param_subset=None):
        """Run grid search (use with caution - can be very large!)."""
        search_space = self.define_search_space()
        
        if param_subset:
            search_space = {k: v for k, v in search_space.items() if k in param_subset}
        
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"🔍 Grid Search: {len(combinations)} total combinations")
        if len(combinations) > 50:
            print("⚠️  Warning: Very large search space! Consider using random search.")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                return []
        
        all_results = []
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            print(f"\n📋 Experiment {i+1}/{len(combinations)} Parameters:")
            for k, v in params.items():
                print(f"   {k}: {v}")
            
            result = self.run_single_experiment(params, f"grid_{i+1:03d}")
            if result:
                all_results.append(result)
        
        self.save_summary(all_results, "grid_search")
        return all_results
    
    def save_summary(self, results, search_type):
        """Save summary of all experiments."""
        if not results:
            print("❌ No results to summarize")
            return
        
        # Sort by accuracy
        results_sorted = sorted(results, key=lambda x: x['results']['test_accuracy'], reverse=True)
        
        summary = {
            'search_type': search_type,
            'total_experiments': len(results),
            'best_accuracy': results_sorted[0]['results']['test_accuracy'],
            'best_parameters': results_sorted[0]['parameters'],
            'all_results': results_sorted,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = self.results_dir / f"{search_type}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"📊 HYPERPARAMETER SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"🎯 Best Accuracy: {summary['best_accuracy']:.4f}")
        print(f"📝 Best Parameters:")
        for k, v in summary['best_parameters'].items():
            print(f"   {k}: {v}")
        print(f"💾 Summary saved to: {summary_file}")
        
    def run_quick_test(self):
        """Run a quick test with 3 promising parameter combinations."""
        promising_configs = [
            # Higher learning rate, more filters
            {
                'learning_rate': 0.002,
                'batch_size': 32,
                'epochs': 500,
                'early_stopping_patience': 100,
                'F1': 16,
                'F2': 32,
                'dropoutRate': 0.4,
                'kernLength': 64,
            },
            # Lower dropout, longer kernel
            {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 500,
                'early_stopping_patience': 100,
                'F1': 8,
                'F2': 16,
                'dropoutRate': 0.3,
                'kernLength': 128,
            },
            # Smaller batch, higher dropout
            {
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 400,
                'early_stopping_patience': 75,
                'F1': 16,
                'F2': 32,
                'dropoutRate': 0.6,
                'kernLength': 64,
            }
        ]
        
        all_results = []
        for i, params in enumerate(promising_configs):
            print(f"\n📋 Quick Test {i+1}/3 Parameters:")
            for k, v in params.items():
                print(f"   {k}: {v}")
            
            result = self.run_single_experiment(params, f"quick_{i+1}")
            if result:
                all_results.append(result)
        
        self.save_summary(all_results, "quick_test")
        return all_results


def main():
    """Main function to run hyperparameter optimization."""
    print("🧠 EEGNet Hyperparameter Optimization")
    print("=====================================")
    
    sweep = HyperparameterSweep()
    
    print("\nSelect optimization strategy:")
    print("1. Quick Test (3 promising configs)")
    print("2. Random Search (20 experiments)")
    print("3. Random Search (50 experiments)")
    print("4. Small Grid Search (learning rate + dropout)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        results = sweep.run_quick_test()
    elif choice == "2":
        results = sweep.run_random_search(20)
    elif choice == "3":
        results = sweep.run_random_search(50)
    elif choice == "4":
        # Limited grid search
        param_subset = ['learning_rate', 'dropoutRate', 'F1']
        results = sweep.run_grid_search(param_subset)
    else:
        print("Invalid choice!")
        return
    
    if results:
        print(f"\n✅ Optimization complete! Check {sweep.results_dir} for detailed results.")
    else:
        print("\n❌ No successful experiments completed.")


if __name__ == "__main__":
    main()