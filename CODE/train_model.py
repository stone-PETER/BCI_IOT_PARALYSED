"""
Training Script for EEGNet Motor Imagery Classification
Updated for unified multi-dataset training with BCI Competition III and IV data

This script handles the complete training pipeline for the EEGNet model
on motor imagery data from multiple BCI competition datasets.
"""

import numpy as np
import yaml
import logging
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Tuple, Optional
import json

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from bci4_2a_loader import BCI4_2A_Loader
from eegnet_model import EEGNet


class EEGNetTrainer:
    """
    Trainer class for EEGNet model on motor imagery data.
    
    Handles the complete training pipeline including data loading,
    model creation, training, validation, and evaluation.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Get script directory for relative path resolution
        from pathlib import Path
        script_dir = Path(__file__).parent.resolve()
        config_path = script_dir / config_path
        
        self.config_path = config_path  # Store config path for later use
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.training_config = self.config['training']
        
        # Setup logging
        log_level = getattr(logging, self.config['logging']['level'])
        log_format = self.config['logging']['format']
        
        if self.config['logging']['save_to_file']:
            # Create logs directory
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
        
        # Initialize components - use BCI IV 2a loader directly
        self.data_loader = BCI4_2A_Loader(str(config_path))
        self.eegnet = EEGNet(config_path)
        
        # Training state
        self.model = None
        self.history = None
        self.data_splits = None
        
        # Create model save directory
        os.makedirs(self.config['paths']['model_save_path'], exist_ok=True)
    
    def load_and_prepare_data(self) -> Dict[str, np.ndarray]:
        """
        Load and prepare data for training using BCI IV 2a loader.
        
        Returns:
            Dictionary containing train/validation/test splits
        """
        self.logger.info("Loading BCI Competition IV Dataset 2a...")
        
        try:
            # Load all subjects' training data
            all_epochs, all_labels = self.data_loader.load_all_subjects(session='T')
            
            self.logger.info(f"Loaded {len(all_epochs)} total epochs")
            self.logger.info(f"Epochs shape: {all_epochs.shape}")
            
            # Convert labels to one-hot encoding
            y_categorical = to_categorical(all_labels, num_classes=4)
            
            # Reshape for EEGNet: (trials, channels, samples, 1)
            X = all_epochs.transpose(0, 2, 1)[..., np.newaxis]
            
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
            
            dataset_info = {
                'total_epochs': len(all_epochs),
                'n_channels': all_epochs.shape[2],
                'n_samples': all_epochs.shape[1],
                'n_classes': 4
            }
            
            self.logger.info(f"Dataset information:")
            self.logger.info(f"  - Total epochs: {dataset_info['total_epochs']}")
            self.logger.info(f"  - Channels: {dataset_info['n_channels']}")
            self.logger.info(f"  - Samples per epoch: {dataset_info['n_samples']}")
            self.logger.info(f"  - Classes: {dataset_info['n_classes']}")
            
            self.logger.info(f"Data shapes:")
            self.logger.info(f"  - Training: X={X_train.shape}, y={y_train.shape}")
            self.logger.info(f"  - Validation: X={X_val.shape}, y={y_val.shape}")
            self.logger.info(f"  - Test: X={X_test.shape}, y={y_test.shape}")
            
            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'dataset_info': dataset_info
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load BCI IV 2a dataset: {e}")
            raise RuntimeError(f"Data loading failed: {e}")
        
    def create_callbacks(self) -> list:
        """
        Create training callbacks for monitoring and optimization.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=self.training_config['early_stopping']['patience'],
            restore_best_weights=self.training_config['early_stopping']['restore_best_weights'],
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint - use .keras extension
        checkpoint_path = os.path.join(
            self.config['paths']['model_save_path'],
            'best_' + self.config['paths']['model_filename'].replace('.h5', '.keras')
        )
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard logging
        log_dir = os.path.join(
            self.config['paths']['logs_path'],
            f"tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        # CSV logging
        csv_log_path = os.path.join(
            self.config['paths']['logs_path'],
            f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        csv_logger = CSVLogger(csv_log_path, append=True)
        callbacks.append(csv_logger)
        
        self.logger.info(f"Created {len(callbacks)} callbacks for training")
        
        return callbacks
    
    def train_model(self, data_splits: Optional[Dict] = None) -> Dict:
        """
        Train the EEGNet model.
        
        Args:
            data_splits: Data splits (uses self.data_splits if None)
            
        Returns:
            Training history
        """
        if data_splits is None:
            data_splits = self.data_splits
        
        if data_splits is None:
            raise ValueError("No data available. Call load_and_prepare_data() first.")
        
        self.logger.info("Starting EEGNet training...")
        
        # Build and compile model
        self.model = self.eegnet.build_model()
        self.eegnet.compile_model(
            learning_rate=self.training_config['learning_rate']
        )
        
        # Print model summary
        self.logger.info("Model Architecture:")
        self.logger.info(f"\n{self.eegnet.get_model_summary()}")
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        start_time = datetime.now()
        
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
    
    def evaluate_model(self, data_splits: Optional[Dict] = None) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            data_splits: Data splits (uses self.data_splits if None)
            
        Returns:
            Evaluation metrics
        """
        if data_splits is None:
            data_splits = self.data_splits
        
        if self.model is None:
            raise ValueError("No trained model available. Train model first.")
        
        self.logger.info("Evaluating model on test data...")
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(
            data_splits['X_test'], data_splits['y_test'],
            verbose=1
        )
        
        # Get predictions for detailed analysis
        test_predictions = self.model.predict(data_splits['X_test'])
        test_pred_classes = np.argmax(test_predictions, axis=1)
        test_true_classes = np.argmax(data_splits['y_test'], axis=1)
        
        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Determine class names based on number of classes
        n_classes = data_splits['y_test'].shape[1]
        if n_classes == 4:
            target_names = ['Left Hand', 'Right Hand', 'Foot', 'Tongue']
        elif n_classes == 2:
            target_names = ['Left Hand', 'Right Hand']
        else:
            target_names = [f'Class {i}' for i in range(n_classes)]
        
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
        
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat.tolist(),
            'predictions': test_predictions.tolist(),
            'true_classes': test_true_classes.tolist(),
            'predicted_classes': test_pred_classes.tolist(),
            'n_classes': n_classes,
            'target_names': target_names
        }
        
        # Save evaluation results
        eval_results_path = os.path.join(
            self.config['paths']['logs_path'],
            f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(eval_results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to: {eval_results_path}")
        
        return evaluation_results
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history (loss and accuracy curves).
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.history is None:
            raise ValueError("No training history available. Train model first.")
        
        history = self.history.history
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot training & validation accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path is None:
            save_path = os.path.join(
                self.config['paths']['logs_path'],
                f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def run_complete_training(self) -> Dict:
        """
        Run the complete training pipeline.
        
        Returns:
            Dictionary containing training results and evaluation metrics
        """
        self.logger.info("Starting complete EEGNet training pipeline...")
        
        # 1. Load and prepare data
        data_splits = self.load_and_prepare_data()
        
        # 2. Train model
        training_history = self.train_model(data_splits)
        
        # 3. Evaluate model
        evaluation_results = self.evaluate_model(data_splits)
        
        # 4. Plot training history
        self.plot_training_history()
        
        # 5. Save training summary
        training_summary = {
            'config': self.config,
            'data_info': data_splits.get('dataset_info', {}),
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'training_completed': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(
            self.config['paths']['logs_path'],
            f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved to: {summary_path}")
        self.logger.info("Complete training pipeline finished successfully!")
        
        return training_summary


def main():
    """Main function to run EEGNet training."""
    print("EEGNet Motor Imagery Training")
    print("=" * 40)
    
    try:
        # Initialize trainer
        trainer = EEGNetTrainer()
        
        # Run complete training pipeline
        results = trainer.run_complete_training()
        
        print("\n" + "=" * 40)
        print("Training completed successfully!")
        print(f"Final test accuracy: {results['evaluation_results']['test_accuracy']:.4f}")
        print(f"Number of classes: {results['evaluation_results']['n_classes']}")
        print(f"Model saved to: {trainer.config['paths']['model_save_path']}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()