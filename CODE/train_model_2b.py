"""
Training Script for BCI Competition IV Dataset 2b
Binary motor imagery classification (Left hand vs Right hand)
3 EEG channels: C3, Cz, C4
"""

import numpy as np
import yaml
import logging
import os
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from bci4_2b_loader_v2 import BCI4_2B_Loader
from eegnet_model import EEGNet


class EEGNetTrainer2B:
    """Trainer for EEGNet on BCI IV 2b (binary classification)."""
    
    def __init__(self, config_path: str = "config_2b.yaml"):
        """Initialize trainer with configuration."""
        # Get script directory
        script_dir = Path(__file__).parent.resolve()
        config_path = script_dir / config_path
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.training_config = self.config['training']
        
        # Setup logging
        log_level = getattr(logging, self.config['logging']['level'])
        log_format = self.config['logging']['format']
        
        if self.config['logging']['save_to_file']:
            os.makedirs(self.config['paths']['logs_path'], exist_ok=True)
            log_file = os.path.join(
                self.config['paths']['logs_path'], 
                self.config['logging']['log_filename']
            )
            logging.basicConfig(
                level=log_level,
                format=log_format,
                handlers=[
                    logging.FileHandler(log_file, encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(level=log_level, format=log_format)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = BCI4_2B_Loader(str(config_path))
        self.eegnet = EEGNet(config_path)
        
        # Training state
        self.model = None
        self.history = None
        
        # Create directories
        os.makedirs(self.config['paths']['model_save_path'], exist_ok=True)
    
    def load_and_prepare_data(self):
        """Load and prepare BCI IV 2b data for training."""
        self.logger.info("Loading BCI Competition IV Dataset 2b...")
        
        # Load all subjects' training data (session T only)
        all_epochs, all_labels = self.data_loader.load_all_subjects(
            sessions=['T']  # Training session only
        )
        
        self.logger.info(f"Loaded {len(all_epochs)} total epochs")
        self.logger.info(f"Epochs shape: {all_epochs.shape}")
        
        # Convert labels to one-hot encoding (binary)
        y_categorical = to_categorical(all_labels, num_classes=2)
        
        # Reshape for EEGNet: (trials, channels, samples, 1)
        X = all_epochs.transpose(0, 2, 1)[..., np.newaxis]
        
        self.logger.info(f"Reshaped data: {X.shape}")
        
        # Split into train/val/test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y_categorical, 
            test_size=0.2, 
            random_state=42,
            stratify=all_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=0.2,
            random_state=42,
            stratify=np.argmax(y_train_val, axis=1)
        )
        
        self.logger.info(f"Data splits:")
        self.logger.info(f"  - Training: X={X_train.shape}, y={y_train.shape}")
        self.logger.info(f"  - Validation: X={X_val.shape}, y={y_val.shape}")
        self.logger.info(f"  - Test: X={X_test.shape}, y={y_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def create_callbacks(self):
        """Create training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=self.training_config['early_stopping']['patience'],
            restore_best_weights=self.training_config['early_stopping']['restore_best_weights'],
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.config['paths']['model_save_path'],
            'best_' + self.config['paths']['model_filename']
        )
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Reduce learning rate
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        log_dir = os.path.join(
            self.config['paths']['logs_path'],
            f"tensorboard_2b_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard)
        
        # CSV Logger
        csv_log_path = os.path.join(
            self.config['paths']['logs_path'],
            f"training_log_2b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        csv_logger = CSVLogger(csv_log_path, append=True)
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train_model(self, data_splits):
        """Train the EEGNet model."""
        self.logger.info("Building and compiling EEGNet for binary classification...")
        
        # Build model
        self.model = self.eegnet.build_model()
        self.eegnet.compile_model(
            learning_rate=self.training_config['learning_rate']
        )
        
        # Print model summary
        self.logger.info("Model Architecture:")
        self.logger.info(f"\n{self.eegnet.get_model_summary()}")
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train
        start_time = datetime.now()
        self.logger.info("Starting training...")
        
        self.history = self.model.fit(
            data_splits['X_train'], data_splits['y_train'],
            validation_data=(data_splits['X_val'], data_splits['y_val']),
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = datetime.now() - start_time
        self.logger.info(f"Training completed in: {training_time}")
        
        # Save final model
        final_model_path = self.eegnet.save_model()
        self.logger.info(f"Final model saved to: {final_model_path}")
        
        return self.history.history
    
    def evaluate_model(self, data_splits):
        """Evaluate the trained model."""
        self.logger.info("Evaluating model on test data...")
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(
            data_splits['X_test'], data_splits['y_test'],
            verbose=1
        )
        
        # Get predictions
        test_predictions = self.model.predict(data_splits['X_test'])
        test_pred_classes = np.argmax(test_predictions, axis=1)
        test_true_classes = np.argmax(data_splits['y_test'], axis=1)
        
        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        target_names = ['Left Hand', 'Right Hand']
        classification_rep = classification_report(
            test_true_classes, test_pred_classes,
            target_names=target_names,
            output_dict=True
        )
        
        confusion_mat = confusion_matrix(test_true_classes, test_pred_classes)
        
        # Log results
        self.logger.info(f"Test Results:")
        self.logger.info(f"  Test Loss: {test_loss:.4f}")
        self.logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"  Confusion Matrix:\n{confusion_mat}")
        
        # Save results
        import json
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat.tolist(),
            'target_names': target_names
        }
        
        results_path = os.path.join(
            self.config['paths']['logs_path'],
            f"evaluation_results_2b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_path}")
        
        return results
    
    def run_complete_training(self):
        """Run complete training pipeline."""
        self.logger.info("="*70)
        self.logger.info("BCI Competition IV Dataset 2b - Binary Classification")
        self.logger.info("Left Hand vs Right Hand Motor Imagery")
        self.logger.info("="*70)
        
        # Load data
        data_splits = self.load_and_prepare_data()
        
        # Train
        training_history = self.train_model(data_splits)
        
        # Evaluate
        results = self.evaluate_model(data_splits)
        
        self.logger.info("="*70)
        self.logger.info("Training Complete!")
        self.logger.info(f"Final Test Accuracy: {results['test_accuracy']:.2%}")
        self.logger.info(f"Expected for binary MI: 70-85%")
        self.logger.info("="*70)
        
        return results


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("BCI IV Dataset 2b - Binary Motor Imagery Classification")
    print("="*70)
    
    try:
        trainer = EEGNetTrainer2B()
        results = trainer.run_complete_training()
        
        print("\n" + "="*70)
        print("✅ Training completed successfully!")
        print(f"📊 Test Accuracy: {results['test_accuracy']:.2%}")
        print(f"🎯 Model saved to: models/best_eegnet_2class_bci2b.keras")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
