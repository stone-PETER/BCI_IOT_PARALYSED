"""
Exploratory Data Analysis (EDA) for BCI Datasets
Analyzes both k3b and BCI Competition IV Dataset 2a

This script performs comprehensive EDA including:
- Dataset information and statistics
- Class distribution analysis
- Signal characteristics analysis
- Data quality assessment
- Visualization of key findings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")

class BCIDatasetAnalyzer:
    """
    Comprehensive analyzer for BCI datasets including k3b and BCI IV 2a.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Dataset paths
        self.k3b_path = "BCI/k3b"
        self.bci4_2a_path = "BCI/bci4_2a"
        
        # Class information
        self.class_names = {
            1: 'Left Hand',
            2: 'Right Hand', 
            3: 'Foot',
            4: 'Tongue'
        }
        
        # Results storage
        self.results = {}
        
    def analyze_k3b_dataset(self) -> Dict:
        """
        Analyze k3b dataset structure and characteristics.
        
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("=== Analyzing k3b Dataset ===")
        
        try:
            # Load k3b files
            classlabel_path = os.path.join(self.k3b_path, "k3b_HDR_Classlabel.txt")
            trig_path = os.path.join(self.k3b_path, "k3b_HDR_TRIG.txt")
            artifact_path = os.path.join(self.k3b_path, "k3b_HDR_ArtifactSelection.txt")
            
            # Read class labels
            with open(classlabel_path, 'r') as f:
                class_data = f.read().strip().split('\n')
            
            # Convert to numeric, handling NaN values
            class_labels = []
            for label in class_data:
                label = label.strip()
                if label == 'NaN' or label == '':
                    class_labels.append(np.nan)
                else:
                    try:
                        class_labels.append(int(float(label)))
                    except:
                        class_labels.append(np.nan)
            
            class_labels = np.array(class_labels)
            
            # Read triggers
            with open(trig_path, 'r') as f:
                trig_data = f.read().strip().split('\n')
            
            triggers = []
            for trig in trig_data:
                trig = trig.strip()
                if trig == 'NaN' or trig == '':
                    triggers.append(np.nan)
                else:
                    try:
                        triggers.append(int(float(trig)))
                    except:
                        triggers.append(np.nan)
            
            triggers = np.array(triggers)
            
            # Read artifact selection
            with open(artifact_path, 'r') as f:
                artifact_data = f.read().strip().split('\n')
                
            artifacts = []
            for art in artifact_data:
                art = art.strip()
                if art == 'NaN' or art == '':
                    artifacts.append(np.nan)
                else:
                    try:
                        artifacts.append(int(float(art)))
                    except:
                        artifacts.append(np.nan)
            
            artifacts = np.array(artifacts)
            
            # Get signal file info (k3b_s.txt is too large to load completely)
            signal_path = os.path.join(self.k3b_path, "k3b_s.txt")
            signal_size = os.path.getsize(signal_path)
            
            # Basic statistics
            total_trials = len(class_labels)
            valid_trials = np.sum(~np.isnan(class_labels))
            invalid_trials = np.sum(np.isnan(class_labels))
            
            # Class distribution (excluding NaN)
            valid_labels = class_labels[~np.isnan(class_labels)]
            unique_classes, class_counts = np.unique(valid_labels, return_counts=True)
            
            # Create results dictionary
            k3b_results = {
                'dataset_name': 'k3b',
                'total_trials': total_trials,
                'valid_trials': valid_trials,
                'invalid_trials': invalid_trials,
                'valid_percentage': (valid_trials / total_trials) * 100,
                'signal_file_size_mb': signal_size / (1024 * 1024),
                'class_distribution': dict(zip(unique_classes.astype(int), class_counts)),
                'class_labels': class_labels,
                'triggers': triggers,
                'artifacts': artifacts,
                'unique_classes': unique_classes,
                'class_counts': class_counts
            }
            
            # Log basic information
            self.logger.info(f"k3b Dataset Summary:")
            self.logger.info(f"  - Total trials: {total_trials}")
            self.logger.info(f"  - Valid trials: {valid_trials} ({valid_trials/total_trials*100:.1f}%)")
            self.logger.info(f"  - Invalid trials: {invalid_trials} ({invalid_trials/total_trials*100:.1f}%)")
            self.logger.info(f"  - Signal file size: {signal_size/(1024*1024):.1f} MB")
            
            # Class distribution
            self.logger.info(f"  - Class distribution:")
            for class_idx, count in zip(unique_classes, class_counts):
                class_name = self.class_names.get(int(class_idx), f"Class {int(class_idx)}")
                percentage = (count / valid_trials) * 100
                self.logger.info(f"    * {class_name}: {count} trials ({percentage:.1f}%)")
            
            return k3b_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing k3b dataset: {e}")
            return {'error': str(e)}
    
    def analyze_bci4_2a_dataset(self) -> Dict:
        """
        Analyze BCI Competition IV Dataset 2a structure and characteristics.
        
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("=== Analyzing BCI Competition IV Dataset 2a ===")
        
        try:
            from bci4_2a_loader import BCI4_2A_Loader
            
            # Initialize loader
            loader = BCI4_2A_Loader()
            
            # Get list of available files
            gdf_files = [f for f in os.listdir(self.bci4_2a_path) if f.endswith('.gdf')]
            training_files = [f for f in gdf_files if 'T.gdf' in f]
            evaluation_files = [f for f in gdf_files if 'E.gdf' in f]
            
            # Analyze each subject
            subject_results = {}
            all_training_labels = []
            all_evaluation_labels = []
            
            for subject_num in range(1, 10):  # A01 to A09
                subject_id = f"A{subject_num:02d}"
                
                try:
                    # Load training session
                    train_file = f"{subject_id}T.gdf"
                    if train_file in training_files:
                        epochs_train, labels_train = loader.load_subject(subject_id, 'T')
                        all_training_labels.extend(labels_train)
                    else:
                        epochs_train, labels_train = None, []
                    
                    # Load evaluation session  
                    eval_file = f"{subject_id}E.gdf"
                    if eval_file in evaluation_files:
                        epochs_eval, labels_eval = loader.load_subject(subject_id, 'E')
                        all_evaluation_labels.extend(labels_eval)
                    else:
                        epochs_eval, labels_eval = None, []
                    
                    # Store subject results
                    subject_results[subject_id] = {
                        'training_epochs': len(labels_train) if labels_train is not None else 0,
                        'evaluation_epochs': len(labels_eval) if labels_eval is not None else 0,
                        'training_labels': labels_train,
                        'evaluation_labels': labels_eval,
                        'training_shape': epochs_train.shape if epochs_train is not None else None,
                        'evaluation_shape': epochs_eval.shape if epochs_eval is not None else None
                    }
                    
                    self.logger.info(f"Subject {subject_id}: Train={len(labels_train)}, Eval={len(labels_eval)} epochs")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load subject {subject_id}: {e}")
                    subject_results[subject_id] = {'error': str(e)}
            
            # Overall statistics
            all_training_labels = np.array(all_training_labels)
            all_evaluation_labels = np.array(all_evaluation_labels)
            
            # Training data class distribution
            if len(all_training_labels) > 0:
                train_unique, train_counts = np.unique(all_training_labels, return_counts=True)
                train_class_dist = dict(zip(train_unique, train_counts))
            else:
                train_class_dist = {}
            
            # Evaluation data class distribution
            if len(all_evaluation_labels) > 0:
                eval_unique, eval_counts = np.unique(all_evaluation_labels, return_counts=True)
                eval_class_dist = dict(zip(eval_unique, eval_counts))
            else:
                eval_class_dist = {}
            
            # Create results dictionary
            bci4_2a_results = {
                'dataset_name': 'BCI Competition IV Dataset 2a',
                'num_subjects': len([s for s in subject_results.keys() if 'error' not in subject_results[s]]),
                'total_training_epochs': len(all_training_labels),
                'total_evaluation_epochs': len(all_evaluation_labels),
                'training_class_distribution': train_class_dist,
                'evaluation_class_distribution': eval_class_dist,
                'subject_results': subject_results,
                'training_labels': all_training_labels,
                'evaluation_labels': all_evaluation_labels,
                'available_files': {
                    'training': training_files,
                    'evaluation': evaluation_files
                }
            }
            
            # Log summary
            self.logger.info(f"BCI IV 2a Dataset Summary:")
            self.logger.info(f"  - Number of subjects: {len([s for s in subject_results.keys() if 'error' not in subject_results[s]])}/9")
            self.logger.info(f"  - Total training epochs: {len(all_training_labels)}")
            self.logger.info(f"  - Total evaluation epochs: {len(all_evaluation_labels)}")
            
            # Training class distribution
            if train_class_dist:
                self.logger.info(f"  - Training class distribution:")
                for class_idx, count in train_class_dist.items():
                    class_name = self.class_names.get(class_idx, f"Class {class_idx}")
                    percentage = (count / len(all_training_labels)) * 100
                    self.logger.info(f"    * {class_name}: {count} trials ({percentage:.1f}%)")
            
            # Evaluation class distribution
            if eval_class_dist:
                self.logger.info(f"  - Evaluation class distribution:")
                for class_idx, count in eval_class_dist.items():
                    class_name = self.class_names.get(class_idx, f"Class {class_idx}")
                    percentage = (count / len(all_evaluation_labels)) * 100
                    self.logger.info(f"    * {class_name}: {count} trials ({percentage:.1f}%)")
            
            return bci4_2a_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing BCI IV 2a dataset: {e}")
            return {'error': str(e)}
    
    def check_class_balance(self, class_distribution: Dict, dataset_name: str) -> Dict:
        """
        Check if the dataset is balanced across the 4 classes.
        
        Args:
            class_distribution: Dictionary with class counts
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with balance analysis results
        """
        if not class_distribution:
            return {'error': 'No class distribution data available'}
        
        # Convert to proper format if needed
        if isinstance(list(class_distribution.keys())[0], np.integer):
            class_distribution = {int(k): v for k, v in class_distribution.items()}
        
        total_samples = sum(class_distribution.values())
        num_classes = len(class_distribution)
        expected_per_class = total_samples / num_classes
        
        # Calculate balance metrics
        balance_results = {
            'dataset_name': dataset_name,
            'total_samples': total_samples,
            'num_classes': num_classes,
            'expected_per_class': expected_per_class,
            'class_percentages': {},
            'deviations_from_expected': {},
            'balance_score': 0,
            'is_balanced': False
        }
        
        deviations = []
        
        for class_idx, count in class_distribution.items():
            percentage = (count / total_samples) * 100
            deviation = abs(count - expected_per_class)
            deviation_percentage = (deviation / expected_per_class) * 100
            
            balance_results['class_percentages'][class_idx] = percentage
            balance_results['deviations_from_expected'][class_idx] = deviation_percentage
            deviations.append(deviation_percentage)
        
        # Calculate balance score (lower is better, 0 = perfectly balanced)
        balance_score = np.mean(deviations)
        balance_results['balance_score'] = balance_score
        
        # Determine if balanced (within 10% deviation from expected)
        balance_results['is_balanced'] = balance_score <= 10.0
        
        # Log balance information
        self.logger.info(f"\n{dataset_name} - Class Balance Analysis:")
        self.logger.info(f"  - Total samples: {total_samples}")
        self.logger.info(f"  - Expected per class: {expected_per_class:.1f}")
        self.logger.info(f"  - Balance score: {balance_score:.2f}% (lower is better)")
        self.logger.info(f"  - Is balanced: {'Yes' if balance_results['is_balanced'] else 'No'}")
        
        for class_idx, percentage in balance_results['class_percentages'].items():
            class_name = self.class_names.get(class_idx, f"Class {class_idx}")
            deviation = balance_results['deviations_from_expected'][class_idx]
            self.logger.info(f"    * {class_name}: {percentage:.1f}% (deviation: {deviation:.1f}%)")
        
        return balance_results
    
    def create_visualizations(self, k3b_results: Dict, bci4_2a_results: Dict):
        """
        Create comprehensive visualizations for both datasets.
        
        Args:
            k3b_results: k3b analysis results
            bci4_2a_results: BCI IV 2a analysis results
        """
        self.logger.info("Creating visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. k3b Class Distribution
        if 'class_distribution' in k3b_results:
            plt.subplot(3, 4, 1)
            classes = list(k3b_results['class_distribution'].keys())
            counts = list(k3b_results['class_distribution'].values())
            class_labels = [self.class_names.get(int(c), f"Class {int(c)}") for c in classes]
            
            bars = plt.bar(class_labels, counts, color=sns.color_palette("husl", len(classes)))
            plt.title('k3b Dataset - Class Distribution')
            plt.xlabel('Motor Imagery Class')
            plt.ylabel('Number of Trials')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom')
        
        # 2. k3b Valid vs Invalid Trials
        if 'valid_trials' in k3b_results:
            plt.subplot(3, 4, 2)
            labels = ['Valid Trials', 'Invalid Trials (NaN)']
            sizes = [k3b_results['valid_trials'], k3b_results['invalid_trials']]
            colors = ['lightgreen', 'lightcoral']
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('k3b Dataset - Trial Validity')
        
        # 3. BCI IV 2a Training Class Distribution
        if 'training_class_distribution' in bci4_2a_results and bci4_2a_results['training_class_distribution']:
            plt.subplot(3, 4, 3)
            classes = list(bci4_2a_results['training_class_distribution'].keys())
            counts = list(bci4_2a_results['training_class_distribution'].values())
            class_labels = [self.class_names.get(c, f"Class {c}") for c in classes]
            
            bars = plt.bar(class_labels, counts, color=sns.color_palette("husl", len(classes)))
            plt.title('BCI IV 2a - Training Class Distribution')
            plt.xlabel('Motor Imagery Class')
            plt.ylabel('Number of Trials')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom')
        
        # 4. BCI IV 2a Evaluation Class Distribution
        if 'evaluation_class_distribution' in bci4_2a_results and bci4_2a_results['evaluation_class_distribution']:
            plt.subplot(3, 4, 4)
            classes = list(bci4_2a_results['evaluation_class_distribution'].keys())
            counts = list(bci4_2a_results['evaluation_class_distribution'].values())
            class_labels = [self.class_names.get(c, f"Class {c}") for c in classes]
            
            bars = plt.bar(class_labels, counts, color=sns.color_palette("husl", len(classes)))
            plt.title('BCI IV 2a - Evaluation Class Distribution')
            plt.xlabel('Motor Imagery Class')
            plt.ylabel('Number of Trials')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom')
        
        # 5. Subject-wise Data Distribution for BCI IV 2a
        if 'subject_results' in bci4_2a_results:
            plt.subplot(3, 4, 5)
            subjects = []
            train_counts = []
            eval_counts = []
            
            for subject_id, results in bci4_2a_results['subject_results'].items():
                if 'error' not in results:
                    subjects.append(subject_id)
                    train_counts.append(results['training_epochs'])
                    eval_counts.append(results['evaluation_epochs'])
            
            if subjects:
                x = np.arange(len(subjects))
                width = 0.35
                
                plt.bar(x - width/2, train_counts, width, label='Training', alpha=0.8)
                plt.bar(x + width/2, eval_counts, width, label='Evaluation', alpha=0.8)
                
                plt.xlabel('Subject')
                plt.ylabel('Number of Trials')
                plt.title('BCI IV 2a - Trials per Subject')
                plt.xticks(x, subjects, rotation=45)
                plt.legend()
        
        # 6. Dataset Size Comparison
        plt.subplot(3, 4, 6)
        datasets = []
        total_trials = []
        
        if 'valid_trials' in k3b_results:
            datasets.append('k3b\n(Valid)')
            total_trials.append(k3b_results['valid_trials'])
        
        if 'total_training_epochs' in bci4_2a_results:
            datasets.append('BCI IV 2a\n(Training)')
            total_trials.append(bci4_2a_results['total_training_epochs'])
        
        if 'total_evaluation_epochs' in bci4_2a_results:
            datasets.append('BCI IV 2a\n(Evaluation)')
            total_trials.append(bci4_2a_results['total_evaluation_epochs'])
        
        if datasets:
            bars = plt.bar(datasets, total_trials, color=['skyblue', 'lightgreen', 'lightcoral'])
            plt.title('Dataset Size Comparison')
            plt.ylabel('Total Number of Trials')
            
            # Add value labels on bars
            for bar, count in zip(bars, total_trials):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(count), ha='center', va='bottom')
        
        # 7-12. Class Balance Heatmaps
        # k3b Balance
        if 'class_distribution' in k3b_results:
            plt.subplot(3, 4, 7)
            k3b_balance = self.check_class_balance(k3b_results['class_distribution'], 'k3b')
            
            if 'class_percentages' in k3b_balance:
                classes = list(k3b_balance['class_percentages'].keys())
                percentages = list(k3b_balance['class_percentages'].values())
                class_labels = [self.class_names.get(int(c), f"Class {int(c)}") for c in classes]
                
                # Create balance visualization
                colors = ['red' if abs(p - 25) > 5 else 'yellow' if abs(p - 25) > 2.5 else 'green' 
                         for p in percentages]
                
                bars = plt.bar(class_labels, percentages, color=colors, alpha=0.7)
                plt.axhline(y=25, color='black', linestyle='--', alpha=0.5, label='Expected (25%)')
                plt.title('k3b Class Balance')
                plt.ylabel('Percentage (%)')
                plt.xticks(rotation=45)
                plt.legend()
                
                # Add value labels
                for bar, pct in zip(bars, percentages):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{pct:.1f}%', ha='center', va='bottom')
        
        # BCI IV 2a Training Balance
        if 'training_class_distribution' in bci4_2a_results and bci4_2a_results['training_class_distribution']:
            plt.subplot(3, 4, 8)
            train_balance = self.check_class_balance(bci4_2a_results['training_class_distribution'], 'BCI IV 2a Training')
            
            if 'class_percentages' in train_balance:
                classes = list(train_balance['class_percentages'].keys())
                percentages = list(train_balance['class_percentages'].values())
                class_labels = [self.class_names.get(c, f"Class {c}") for c in classes]
                
                colors = ['red' if abs(p - 25) > 5 else 'yellow' if abs(p - 25) > 2.5 else 'green' 
                         for p in percentages]
                
                bars = plt.bar(class_labels, percentages, color=colors, alpha=0.7)
                plt.axhline(y=25, color='black', linestyle='--', alpha=0.5, label='Expected (25%)')
                plt.title('BCI IV 2a Training - Class Balance')
                plt.ylabel('Percentage (%)')
                plt.xticks(rotation=45)
                plt.legend()
                
                # Add value labels
                for bar, pct in zip(bars, percentages):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{pct:.1f}%', ha='center', va='bottom')
        
        # BCI IV 2a Evaluation Balance
        if 'evaluation_class_distribution' in bci4_2a_results and bci4_2a_results['evaluation_class_distribution']:
            plt.subplot(3, 4, 9)
            eval_balance = self.check_class_balance(bci4_2a_results['evaluation_class_distribution'], 'BCI IV 2a Evaluation')
            
            if 'class_percentages' in eval_balance:
                classes = list(eval_balance['class_percentages'].keys())
                percentages = list(eval_balance['class_percentages'].values())
                class_labels = [self.class_names.get(c, f"Class {c}") for c in classes]
                
                colors = ['red' if abs(p - 25) > 5 else 'yellow' if abs(p - 25) > 2.5 else 'green' 
                         for p in percentages]
                
                bars = plt.bar(class_labels, percentages, color=colors, alpha=0.7)
                plt.axhline(y=25, color='black', linestyle='--', alpha=0.5, label='Expected (25%)')
                plt.title('BCI IV 2a Evaluation - Class Balance')
                plt.ylabel('Percentage (%)')
                plt.xticks(rotation=45)
                plt.legend()
                
                # Add value labels
                for bar, pct in zip(bars, percentages):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{pct:.1f}%', ha='center', va='bottom')
        
        # 10. Data Quality Summary
        plt.subplot(3, 4, 10)
        quality_metrics = []
        quality_labels = []
        
        if 'valid_percentage' in k3b_results:
            quality_metrics.append(k3b_results['valid_percentage'])
            quality_labels.append('k3b\nValid %')
        
        if bci4_2a_results.get('num_subjects', 0) > 0:
            quality_metrics.append((bci4_2a_results['num_subjects'] / 9) * 100)
            quality_labels.append('BCI IV 2a\nSubjects %')
        
        if quality_metrics:
            colors = ['green' if m >= 80 else 'yellow' if m >= 60 else 'red' for m in quality_metrics]
            bars = plt.bar(quality_labels, quality_metrics, color=colors, alpha=0.7)
            plt.title('Data Quality Metrics')
            plt.ylabel('Percentage (%)')
            plt.ylim(0, 100)
            
            # Add value labels
            for bar, metric in zip(bars, quality_metrics):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{metric:.1f}%', ha='center', va='bottom')
        
        # 11. File Size Information
        plt.subplot(3, 4, 11)
        file_info = []
        
        if 'signal_file_size_mb' in k3b_results:
            file_info.append(('k3b Signal File', k3b_results['signal_file_size_mb']))
        
        # Get BCI IV 2a file sizes
        if os.path.exists(self.bci4_2a_path):
            gdf_files = [f for f in os.listdir(self.bci4_2a_path) if f.endswith('.gdf')]
            if gdf_files:
                total_size = sum(os.path.getsize(os.path.join(self.bci4_2a_path, f)) 
                               for f in gdf_files) / (1024 * 1024)
                file_info.append(('BCI IV 2a Total', total_size))
        
        if file_info:
            names, sizes = zip(*file_info)
            bars = plt.bar(names, sizes, color=['skyblue', 'lightgreen'])
            plt.title('Dataset File Sizes')
            plt.ylabel('Size (MB)')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, size in zip(bars, sizes):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{size:.1f} MB', ha='center', va='bottom')
        
        # 12. Summary Table (text)
        plt.subplot(3, 4, 12)
        plt.axis('off')
        
        summary_text = "Dataset Summary:\n\n"
        
        if 'class_distribution' in k3b_results:
            summary_text += "k3b Dataset:\n"
            summary_text += f"• Total: {k3b_results['total_trials']} trials\n"
            summary_text += f"• Valid: {k3b_results['valid_trials']} ({k3b_results['valid_percentage']:.1f}%)\n"
            
            balance_k3b = self.check_class_balance(k3b_results['class_distribution'], 'k3b')
            if 'is_balanced' in balance_k3b:
                status = "Balanced" if balance_k3b['is_balanced'] else "Imbalanced"
                summary_text += f"• Balance: {status}\n\n"
        
        if 'total_training_epochs' in bci4_2a_results:
            summary_text += "BCI IV 2a Dataset:\n"
            summary_text += f"• Training: {bci4_2a_results['total_training_epochs']} trials\n"
            summary_text += f"• Evaluation: {bci4_2a_results['total_evaluation_epochs']} trials\n"
            summary_text += f"• Subjects: {bci4_2a_results.get('num_subjects', 0)}/9\n"
            
            if bci4_2a_results['training_class_distribution']:
                train_balance = self.check_class_balance(bci4_2a_results['training_class_distribution'], 'Training')
                if 'is_balanced' in train_balance:
                    status = "Balanced" if train_balance['is_balanced'] else "Imbalanced"
                    summary_text += f"• Training Balance: {status}\n"
            
            if bci4_2a_results['evaluation_class_distribution']:
                eval_balance = self.check_class_balance(bci4_2a_results['evaluation_class_distribution'], 'Evaluation')
                if 'is_balanced' in eval_balance:
                    status = "Balanced" if eval_balance['is_balanced'] else "Imbalanced"
                    summary_text += f"• Evaluation Balance: {status}\n"
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        plt.title('Summary')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('bci_datasets_eda.png', dpi=300, bbox_inches='tight')
        self.logger.info("Visualization saved as 'bci_datasets_eda.png'")
        
        plt.show()
    
    def generate_report(self, k3b_results: Dict, bci4_2a_results: Dict) -> str:
        """
        Generate a comprehensive EDA report.
        
        Args:
            k3b_results: k3b analysis results
            bci4_2a_results: BCI IV 2a analysis results
            
        Returns:
            Formatted report string
        """
        report = "="*80 + "\n"
        report += "BCI DATASETS EXPLORATORY DATA ANALYSIS REPORT\n"
        report += "="*80 + "\n\n"
        
        # k3b Dataset Section
        report += "1. K3B DATASET ANALYSIS\n"
        report += "-"*50 + "\n"
        
        if 'error' in k3b_results:
            report += f"ERROR: {k3b_results['error']}\n\n"
        else:
            report += f"Dataset Overview:\n"
            report += f"  • Total trials: {k3b_results['total_trials']}\n"
            report += f"  • Valid trials: {k3b_results['valid_trials']} ({k3b_results['valid_percentage']:.1f}%)\n"
            report += f"  • Invalid trials: {k3b_results['invalid_trials']} ({100-k3b_results['valid_percentage']:.1f}%)\n"
            report += f"  • Signal file size: {k3b_results['signal_file_size_mb']:.1f} MB\n\n"
            
            report += f"Class Distribution:\n"
            for class_idx, count in k3b_results['class_distribution'].items():
                class_name = self.class_names.get(int(class_idx), f"Class {int(class_idx)}")
                percentage = (count / k3b_results['valid_trials']) * 100
                report += f"  • {class_name}: {count} trials ({percentage:.1f}%)\n"
            
            # Balance analysis
            balance_results = self.check_class_balance(k3b_results['class_distribution'], 'k3b')
            if 'balance_score' in balance_results:
                report += f"\nClass Balance Analysis:\n"
                report += f"  • Balance score: {balance_results['balance_score']:.2f}% (lower is better)\n"
                report += f"  • Is balanced: {'Yes' if balance_results['is_balanced'] else 'No'}\n"
                report += f"  • Expected per class: {balance_results['expected_per_class']:.1f} trials\n\n"
        
        # BCI IV 2a Dataset Section
        report += "2. BCI COMPETITION IV DATASET 2A ANALYSIS\n"
        report += "-"*50 + "\n"
        
        if 'error' in bci4_2a_results:
            report += f"ERROR: {bci4_2a_results['error']}\n\n"
        else:
            report += f"Dataset Overview:\n"
            report += f"  • Number of subjects: {bci4_2a_results['num_subjects']}/9\n"
            report += f"  • Total training epochs: {bci4_2a_results['total_training_epochs']}\n"
            report += f"  • Total evaluation epochs: {bci4_2a_results['total_evaluation_epochs']}\n\n"
            
            # Training data
            if bci4_2a_results['training_class_distribution']:
                report += f"Training Data Class Distribution:\n"
                total_train = bci4_2a_results['total_training_epochs']
                for class_idx, count in bci4_2a_results['training_class_distribution'].items():
                    class_name = self.class_names.get(class_idx, f"Class {class_idx}")
                    percentage = (count / total_train) * 100
                    report += f"  • {class_name}: {count} trials ({percentage:.1f}%)\n"
                
                # Training balance
                train_balance = self.check_class_balance(bci4_2a_results['training_class_distribution'], 'Training')
                if 'balance_score' in train_balance:
                    report += f"\nTraining Data Balance:\n"
                    report += f"  • Balance score: {train_balance['balance_score']:.2f}%\n"
                    report += f"  • Is balanced: {'Yes' if train_balance['is_balanced'] else 'No'}\n"
                    report += f"  • Expected per class: {train_balance['expected_per_class']:.1f} trials\n"
            
            # Evaluation data
            if bci4_2a_results['evaluation_class_distribution']:
                report += f"\nEvaluation Data Class Distribution:\n"
                total_eval = bci4_2a_results['total_evaluation_epochs']
                for class_idx, count in bci4_2a_results['evaluation_class_distribution'].items():
                    class_name = self.class_names.get(class_idx, f"Class {class_idx}")
                    percentage = (count / total_eval) * 100
                    report += f"  • {class_name}: {count} trials ({percentage:.1f}%)\n"
                
                # Evaluation balance
                eval_balance = self.check_class_balance(bci4_2a_results['evaluation_class_distribution'], 'Evaluation')
                if 'balance_score' in eval_balance:
                    report += f"\nEvaluation Data Balance:\n"
                    report += f"  • Balance score: {eval_balance['balance_score']:.2f}%\n"
                    report += f"  • Is balanced: {'Yes' if eval_balance['is_balanced'] else 'No'}\n"
                    report += f"  • Expected per class: {eval_balance['expected_per_class']:.1f} trials\n"
            
            # Subject-wise breakdown
            report += f"\nSubject-wise Data:\n"
            for subject_id, results in bci4_2a_results['subject_results'].items():
                if 'error' not in results:
                    report += f"  • {subject_id}: {results['training_epochs']} train, {results['evaluation_epochs']} eval\n"
                else:
                    report += f"  • {subject_id}: ERROR - {results['error']}\n"
        
        # Overall Comparison
        report += "\n3. DATASET COMPARISON\n"
        report += "-"*50 + "\n"
        
        # Size comparison
        report += "Dataset Sizes:\n"
        if 'valid_trials' in k3b_results:
            report += f"  • k3b: {k3b_results['valid_trials']} valid trials\n"
        if 'total_training_epochs' in bci4_2a_results:
            report += f"  • BCI IV 2a Training: {bci4_2a_results['total_training_epochs']} trials\n"
        if 'total_evaluation_epochs' in bci4_2a_results:
            report += f"  • BCI IV 2a Evaluation: {bci4_2a_results['total_evaluation_epochs']} trials\n"
        
        # Balance comparison
        report += "\nClass Balance Summary:\n"
        
        if 'class_distribution' in k3b_results:
            k3b_balance = self.check_class_balance(k3b_results['class_distribution'], 'k3b')
            if 'is_balanced' in k3b_balance:
                status = "✓ Balanced" if k3b_balance['is_balanced'] else "✗ Imbalanced"
                report += f"  • k3b: {status} (score: {k3b_balance['balance_score']:.2f}%)\n"
        
        if bci4_2a_results['training_class_distribution']:
            train_balance = self.check_class_balance(bci4_2a_results['training_class_distribution'], 'Training')
            if 'is_balanced' in train_balance:
                status = "✓ Balanced" if train_balance['is_balanced'] else "✗ Imbalanced"
                report += f"  • BCI IV 2a Training: {status} (score: {train_balance['balance_score']:.2f}%)\n"
        
        if bci4_2a_results['evaluation_class_distribution']:
            eval_balance = self.check_class_balance(bci4_2a_results['evaluation_class_distribution'], 'Evaluation')
            if 'is_balanced' in eval_balance:
                status = "✓ Balanced" if eval_balance['is_balanced'] else "✗ Imbalanced"
                report += f"  • BCI IV 2a Evaluation: {status} (score: {eval_balance['balance_score']:.2f}%)\n"
        
        # Recommendations
        report += "\n4. RECOMMENDATIONS\n"
        report += "-"*50 + "\n"
        
        # For k3b
        if 'valid_percentage' in k3b_results:
            if k3b_results['valid_percentage'] < 80:
                report += "k3b Dataset:\n"
                report += f"  • High invalid trial rate ({100-k3b_results['valid_percentage']:.1f}%) - investigate data quality\n"
                report += "  • Consider artifact removal or stricter preprocessing\n"
            
            if 'class_distribution' in k3b_results:
                k3b_balance = self.check_class_balance(k3b_results['class_distribution'], 'k3b')
                if not k3b_balance.get('is_balanced', True):
                    report += "  • Class imbalance detected - consider data augmentation or class weighting\n"
        
        # For BCI IV 2a
        if bci4_2a_results.get('num_subjects', 0) < 9:
            report += "\nBCI IV 2a Dataset:\n"
            report += "  • Some subjects failed to load - check file integrity\n"
        
        if bci4_2a_results['training_class_distribution']:
            train_balance = self.check_class_balance(bci4_2a_results['training_class_distribution'], 'Training')
            if not train_balance.get('is_balanced', True):
                report += "  • Training data is imbalanced - consider stratified sampling\n"
        
        if bci4_2a_results['evaluation_class_distribution']:
            eval_balance = self.check_class_balance(bci4_2a_results['evaluation_class_distribution'], 'Evaluation')
            if not eval_balance.get('is_balanced', True):
                report += "  • Evaluation data is imbalanced - may affect performance metrics\n"
        
        # General recommendations
        report += "\nGeneral Recommendations:\n"
        report += "  • Use stratified train-test splits to maintain class balance\n"
        report += "  • Apply data augmentation techniques for motor imagery\n"
        report += "  • Consider ensemble methods to handle class imbalance\n"
        report += "  • Monitor per-class performance metrics during training\n"
        
        report += "\n" + "="*80 + "\n"
        report += "END OF REPORT\n"
        report += "="*80 + "\n"
        
        return report
    
    def run_complete_analysis(self):
        """
        Run the complete EDA analysis on both datasets.
        """
        self.logger.info("Starting comprehensive BCI datasets EDA...")
        
        # Analyze both datasets
        k3b_results = self.analyze_k3b_dataset()
        bci4_2a_results = self.analyze_bci4_2a_dataset()
        
        # Store results
        self.results = {
            'k3b': k3b_results,
            'bci4_2a': bci4_2a_results
        }
        
        # Create visualizations
        self.create_visualizations(k3b_results, bci4_2a_results)
        
        # Generate and save report
        report = self.generate_report(k3b_results, bci4_2a_results)
        
        # Save report to file
        with open('bci_datasets_eda_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info("Report saved as 'bci_datasets_eda_report.txt'")
        
        # Print report to console
        print(report)
        
        return self.results


def main():
    """Main function to run the EDA analysis."""
    try:
        # Initialize analyzer
        analyzer = BCIDatasetAnalyzer()
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        print("\n" + "="*80)
        print("EDA ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Generated files:")
        print("  • bci_datasets_eda.png - Comprehensive visualizations")
        print("  • bci_datasets_eda_report.txt - Detailed analysis report")
        print("="*80)
        
        return results
        
    except Exception as e:
        logging.error(f"EDA analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()