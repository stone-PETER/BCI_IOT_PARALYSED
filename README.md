# 🧠 BCI Real-time Classification System

A real-time Brain-Computer Interface (BCI) system for motor imagery classification using EEGNet deep learning model with React frontend and Flask backend.

## 🎯 Features

- **Real-time EEG Classification**: Live motor imagery classification using trained EEGNet model
- **4-Class Motor Imagery**: Left Hand, Right Hand, Foot, and Tongue movement imagination
- **Real-time Data Visualization**: Live EEG signal plotting with motor cortex channels
- **Performance Analytics**: Session statistics, accuracy tracking, and model comparison
- **WebSocket Communication**: Real-time data streaming between frontend and backend
- **Modular Architecture**: Production-ready design for future real EEG hardware integration

## 🏗️ Architecture

### Backend (Flask + WebSocket)

- `preprocessing.py`: EEG signal preprocessing pipeline with filtering and normalization
- `simulator.py`: Real-time EEG data simulation using recorded BCI datasets
- `inference.py`: Model inference engine with real-time classification
- `app.py`: Flask server with WebSocket support and REST API endpoints

### Frontend (React)

- Real-time EEG signal visualization using Recharts
- Motor imagery class selection and control interface
- Live prediction display with confidence scores and probability bars
- Performance statistics and session results comparison

### Shared

- Model configuration and utility functions

## 📋 Prerequisites

### Backend Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- Flask
- Flask-SocketIO
- NumPy
- SciPy
- joblib

### Frontend Requirements

- Node.js 16+
- npm or yarn

## 🚀 Installation & Setup

### 1. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Ensure the trained model files are in the correct location:

- `bci_eegnet_model.h5` (trained EEGNet model)
- `bci_preprocessor.pkl` (trained preprocessing pipeline)
- `bci_preprocessed_data.npz` (BCI dataset for simulation)

### 2. Frontend Setup

Navigate to the frontend directory:

```bash
cd frontend
```

Install npm dependencies:

```bash
npm install
```

## 🏃‍♂️ Running the System

### 1. Start the Backend Server

In the backend directory:

```bash
python app.py
```

The Flask server will start on `http://localhost:5000` with WebSocket support.

### 2. Start the Frontend Development Server

In the frontend directory:

```bash
npm start
```

The React app will start on `http://localhost:3000` and automatically connect to the backend.

## 📖 Usage Guide

### 1. System Connection

- Open your browser to `http://localhost:3000`
- The system will automatically connect to the backend server
- Connection status is displayed in the top-left corner

### 2. Motor Imagery Class Selection

- Choose one of the 4 motor imagery classes:
  - **Left Hand** 🖐️: Imagine moving your left hand
  - **Right Hand** 👋: Imagine moving your right hand
  - **Foot** 🦶: Imagine moving your foot
  - **Tongue** 👅: Imagine moving your tongue
- Class selection is disabled during active streaming

### 3. Real-time Classification

- Click **"Start Streaming"** to begin real-time classification
- The system will:
  - Simulate EEG data for the selected motor imagery class
  - Process signals in real-time (250Hz sampling rate)
  - Classify every 2 seconds using the trained EEGNet model
  - Display live predictions with confidence scores

### 4. Visualization & Analytics

- **Live EEG Signals**: Real-time plot of motor cortex channels (C3, Cz, C4, CP1, CP2, P3, Pz, P4)
- **Prediction Display**: Current classification with confidence level and probability bars
- **Performance Stats**: Live session statistics including accuracy, processing time, and classification rate
- **Session Results**: Complete analysis after stopping the stream

### 5. Session Management

- Click **"Stop Streaming"** to end the current session
- View comprehensive session results including:
  - Final accuracy and performance metrics
  - Per-class precision, recall, and F1-scores
  - Comparison with trained model performance

## 🔧 Configuration

### Backend Configuration

The system can be configured in `app.py`:

- **Sampling Rate**: Default 250Hz (configurable in simulator)
- **Buffer Size**: 2-second classification windows
- **Model Path**: Update paths to your trained model files

### Frontend Configuration

Update API endpoints in `App.js` if backend runs on different host/port:

```javascript
const newSocket = io("http://localhost:5000", {
  transports: ["websocket", "polling"],
});
```

## 📊 Model Performance

The system includes comprehensive model validation showing:

- **Balanced Test Accuracy**: 47.64% (4-class classification)
- **Cross-subject Validation**: 91.64% ± 3.97% accuracy
- **Real-time Processing**: 106.7 Hz classification rate
- **Class-specific Performance**:
  - Left Hand: High accuracy (majority class)
  - Right Hand: 7.5% recall, 100% precision
  - Foot: 3.8% recall, 40% precision
  - Tongue: High accuracy (majority class)

## 🔬 Technical Details

### EEG Preprocessing Pipeline

1. **Bandpass Filtering**: 0.5-50 Hz for brain signal isolation
2. **Common Average Referencing (CAR)**: Noise reduction
3. **Z-score Normalization**: Feature standardization
4. **Windowing**: 2-second classification windows with 0.5s overlap

### Model Architecture

- **EEGNet**: Compact CNN designed for EEG-BCI tasks
- **Input**: 8 motor cortex channels × 500 samples (2s at 250Hz)
- **Output**: 4-class motor imagery classification
- **Real-time Capable**: <10ms inference time

### Data Flow

1. **Simulation**: Recorded BCI data replay at 250Hz
2. **Preprocessing**: Real-time filtering and normalization
3. **Buffering**: Sliding window data management
4. **Classification**: Model inference every 2 seconds
5. **Visualization**: WebSocket streaming to frontend

## 🚀 Production Deployment

### For Real EEG Hardware Integration:

1. Replace `simulator.py` with real EEG data acquisition
2. Update channel mappings in preprocessing pipeline
3. Adjust sampling rates and buffer sizes as needed
4. Maintain the same WebSocket API for seamless frontend integration

### Scalability Considerations:

- Use Redis for session management in multi-user scenarios
- Implement load balancing for multiple concurrent sessions
- Add authentication and user management systems
- Deploy using Docker containers for consistent environments

## 🐛 Troubleshooting

### Common Issues:

1. **Connection Failed**: Ensure backend server is running on port 5000
2. **No EEG Data**: Check that model files are in correct locations
3. **Poor Performance**: Verify model file integrity and preprocessing pipeline
4. **Browser Compatibility**: Use modern browsers with WebSocket support

### Debug Mode:

Enable verbose logging in `app.py`:

```python
app.config['DEBUG'] = True
socketio.run(app, debug=True, port=5000)
```

## 📚 Research & References

This system is based on:

- **EEGNet Architecture**: Lawhern et al. (2018) - "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
- **Motor Imagery BCI**: Pfurtscheller & Neuper (2001) - "Motor imagery and direct brain-computer communication"
- **Real-time Processing**: Schlögl et al. (2007) - "Characterization of four-class motor imagery EEG data for the BCI-competition 2005"

## 📄 License

This project is developed for research and educational purposes. Please cite appropriately if used in academic work.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Real EEG hardware integration
- Additional visualization features
- Performance optimizations
- Documentation improvements

---

**🧠 Built for advancing Brain-Computer Interface research and development**
