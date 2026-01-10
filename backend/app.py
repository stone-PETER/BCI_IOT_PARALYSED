#!/usr/bin/env python3
"""
Real-time BCI Flask Backend with WebSocket Support
Provides REST API and WebSocket streaming for real-time EEG classification
"""

import logging
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np

# Import our modules
from simulator import EEGDataSimulator
from inference import BCIModelInference
from preProcessing import MOTOR_IMAGERY_CLASSES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class BCIBackend:
    """
    Real-time BCI Backend Server
    """
    
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'bci-realtime-secret-key'
        
        # Enable CORS for frontend communication
        CORS(self.app, origins=['http://localhost:3000'])
        
        # Initialize SocketIO
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins=['http://localhost:3000'],
            async_mode='threading'
        )
        
        # Initialize components
        self.simulator = EEGDataSimulator()
        self.inference_engine = BCIModelInference()
        
        # Session state
        self.is_streaming = False
        self.current_class = 0
        self.true_labels = []  # Track true labels for accuracy calculation
        self.session_start_time = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Register routes and events
        self._register_routes()
        self._register_socketio_events()
        
        self.logger.info("BCI Backend initialized")
    
    def _register_routes(self):
        """Register REST API routes"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time(),
                'components': {
                    'simulator': 'ready',
                    'inference_engine': 'ready',
                    'model_loaded': self.inference_engine.model is not None
                }
            })
        
        @self.app.route('/api/available-classes', methods=['GET'])
        def get_available_classes():
            """Get available motor imagery classes"""
            available = self.simulator.get_available_classes()
            return jsonify({
                'classes': available,
                'current_class': self.current_class,
                'current_class_name': MOTOR_IMAGERY_CLASSES.get(self.current_class, 'Unknown')
            })
        
        @self.app.route('/api/set-class', methods=['POST'])
        def set_motor_imagery_class():
            """Set current motor imagery class for simulation"""
            try:
                data = request.get_json()
                class_idx = data.get('class_index')
                
                if class_idx is None or class_idx not in range(4):
                    return jsonify({'error': 'Invalid class index'}), 400
                
                self.current_class = class_idx
                self.simulator.set_motor_imagery_class(class_idx)
                
                return jsonify({
                    'success': True,
                    'class_index': class_idx,
                    'class_name': MOTOR_IMAGERY_CLASSES[class_idx]
                })
                
            except Exception as e:
                self.logger.error(f"Set class error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/start-streaming', methods=['POST'])
        def start_streaming():
            """Start real-time EEG streaming and classification"""
            try:
                if self.is_streaming:
                    return jsonify({'error': 'Streaming already active'}), 400
                
                # Reset session
                self.inference_engine.reset_session()
                self.true_labels.clear()
                self.session_start_time = time.time()
                
                # Start inference engine
                self.inference_engine.start_realtime_processing(self._on_prediction)
                
                # Start simulator
                self.simulator.start_streaming(self._on_eeg_data)
                
                self.is_streaming = True
                
                # Notify clients
                self.socketio.emit('streaming_started', {
                    'timestamp': time.time(),
                    'current_class': self.current_class,
                    'class_name': MOTOR_IMAGERY_CLASSES[self.current_class]
                })
                
                return jsonify({
                    'success': True,
                    'message': 'Streaming started',
                    'current_class': self.current_class
                })
                
            except Exception as e:
                self.logger.error(f"Start streaming error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stop-streaming', methods=['POST'])
        def stop_streaming():
            """Stop real-time EEG streaming"""
            try:
                if not self.is_streaming:
                    return jsonify({'error': 'Streaming not active'}), 400
                
                # Stop components
                self.simulator.stop_streaming()
                self.inference_engine.stop_realtime_processing()
                
                self.is_streaming = False
                
                # Calculate session results
                session_results = self._calculate_session_results()
                
                # Notify clients
                self.socketio.emit('streaming_stopped', {
                    'timestamp': time.time(),
                    'session_results': session_results
                })
                
                return jsonify({
                    'success': True,
                    'message': 'Streaming stopped',
                    'session_results': session_results
                })
                
            except Exception as e:
                self.logger.error(f"Stop streaming error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/session-stats', methods=['GET'])
        def get_session_stats():
            """Get current session statistics"""
            try:
                performance_stats = self.inference_engine.get_performance_stats()
                accuracy_stats = self.inference_engine.get_session_accuracy(self.true_labels)
                
                return jsonify({
                    'performance': performance_stats,
                    'accuracy': accuracy_stats,
                    'session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
                    'true_labels_count': len(self.true_labels),
                    'is_streaming': self.is_streaming
                })
                
            except Exception as e:
                self.logger.error(f"Session stats error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _register_socketio_events(self):
        """Register WebSocket events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.logger.info(f"Client connected: {request.sid}")
            emit('connection_established', {
                'timestamp': time.time(),
                'available_classes': self.simulator.get_available_classes(),
                'current_class': self.current_class,
                'is_streaming': self.is_streaming
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_status')
        def handle_status_request():
            """Handle status request from client"""
            emit('status_update', {
                'is_streaming': self.is_streaming,
                'current_class': self.current_class,
                'class_name': MOTOR_IMAGERY_CLASSES.get(self.current_class, 'Unknown'),
                'timestamp': time.time()
            })
    
    def _on_eeg_data(self, chunk: np.ndarray, true_class: int):
        """
        Handle incoming EEG data from simulator
        
        Args:
            chunk: EEG data chunk (channels x samples)
            true_class: True motor imagery class
        """
        try:
            # Add to inference engine
            self.inference_engine.add_eeg_chunk(chunk)
            
            # Track true label
            self.true_labels.append(true_class)
            
            # Emit raw EEG data to clients (subset of channels for visualization)
            display_channels = [9, 10, 11, 12, 13, 14]  # C3, Cz, C4 area
            display_data = chunk[display_channels, :].tolist()
            
            self.socketio.emit('eeg_data', {
                'data': display_data,
                'channels': display_channels,
                'samples': chunk.shape[1],
                'true_class': true_class,
                'timestamp': time.time()
            })
            
        except Exception as e:
            self.logger.error(f"EEG data handling error: {e}")
    
    def _on_prediction(self, result: Dict):
        """
        Handle prediction result from inference engine
        
        Args:
            result: Prediction result dictionary
        """
        try:
            # Emit prediction to clients
            self.socketio.emit('prediction', result)
            
        except Exception as e:
            self.logger.error(f"Prediction handling error: {e}")
    
    def _calculate_session_results(self) -> Dict:
        """Calculate comprehensive session results"""
        try:
            performance_stats = self.inference_engine.get_performance_stats()
            accuracy_stats = self.inference_engine.get_session_accuracy(self.true_labels)
            
            # Calculate class distribution
            class_distribution = {}
            for class_idx in range(4):
                count = self.true_labels.count(class_idx)
                class_distribution[MOTOR_IMAGERY_CLASSES[class_idx]] = count
            
            # Get recent classification history
            recent_history = self.inference_engine.classification_history[-100:]  # Last 100 predictions
            
            session_duration = time.time() - self.session_start_time if self.session_start_time else 0
            
            return {
                'performance': performance_stats,
                'accuracy': accuracy_stats,
                'session_duration_seconds': session_duration,
                'total_samples': len(self.true_labels),
                'class_distribution': class_distribution,
                'recent_predictions': recent_history,
                'trained_model_comparison': {
                    'trained_accuracy': 0.7636,  # From model_accuracy_checker results
                    'trained_balanced_accuracy': 0.4764,  # From improved results
                    'session_accuracy': accuracy_stats.get('accuracy', 0),
                    'accuracy_difference': accuracy_stats.get('accuracy', 0) - 0.4764
                }
            }
            
        except Exception as e:
            self.logger.error(f"Session results calculation error: {e}")
            return {
                'performance': {},
                'accuracy': {},
                'session_duration_seconds': 0,
                'error': str(e)
            }
    
    def run(self, debug=True):
        """Run the Flask backend server"""
        self.logger.info(f"Starting BCI Backend on {self.host}:{self.port}")
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            use_reloader=False  # Avoid issues with threading
        )

# Create backend instance
backend = BCIBackend()

if __name__ == "__main__":
    backend.run(debug=True)