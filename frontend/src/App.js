import React, { useState, useEffect, useRef } from "react";
import io from "socket.io-client";
import "./index.css";
import EEGChart from "./components/EEGChart";
import PredictionDisplay from "./components/PredictionDisplay";
import PerformanceStats from "./components/PerformanceStats";

const MOTOR_IMAGERY_CLASSES = {
  0: "Left Hand",
  1: "Right Hand",
  2: "Foot",
  3: "Tongue",
};

function App() {
  // Connection state
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  // Data state
  const [currentClass, setCurrentClass] = useState(0);
  const [availableClasses, setAvailableClasses] = useState({});
  const [eegData, setEegData] = useState([]);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [sessionResults, setSessionResults] = useState(null);
  const [error, setError] = useState(null);

  // Performance tracking
  const [performanceStats, setPerformanceStats] = useState({});
  const [predictionHistory, setPredictionHistory] = useState([]);

  // Refs for data management
  const eegBufferRef = useRef([]);
  const maxBufferSize = 1250; // 5 seconds at 250Hz

  useEffect(() => {
    // Initialize socket connection
    const newSocket = io("http://localhost:5000", {
      transports: ["websocket", "polling"],
    });

    setSocket(newSocket);

    // Connection event handlers
    newSocket.on("connect", () => {
      console.log("Connected to backend");
      setIsConnected(true);
      setError(null);
    });

    newSocket.on("disconnect", () => {
      console.log("Disconnected from backend");
      setIsConnected(false);
      setIsStreaming(false);
    });

    newSocket.on("connection_established", (data) => {
      console.log("Connection established:", data);
      setAvailableClasses(data.available_classes);
      setCurrentClass(data.current_class);
      setIsStreaming(data.is_streaming);
    });

    // Data event handlers
    newSocket.on("eeg_data", (data) => {
      // Add new EEG data to buffer
      const newSamples = data.data
        .map((channelData, channelIndex) =>
          channelData.map((value, sampleIndex) => ({
            time: Date.now() + sampleIndex * 4, // 4ms per sample at 250Hz
            [`ch${data.channels[channelIndex]}`]: value,
          }))
        )
        .flat();

      // Update buffer
      eegBufferRef.current = [...eegBufferRef.current, ...newSamples].slice(
        -maxBufferSize
      );
      setEegData([...eegBufferRef.current]);
    });

    newSocket.on("prediction", (prediction) => {
      setCurrentPrediction(prediction);
      setPredictionHistory((prev) => [...prev, prediction].slice(-100)); // Keep last 100
    });

    newSocket.on("streaming_started", (data) => {
      console.log("Streaming started:", data);
      setIsStreaming(true);
      setSessionResults(null);
      setError(null);
      // Clear data
      eegBufferRef.current = [];
      setEegData([]);
      setPredictionHistory([]);
      setCurrentPrediction(null);
    });

    newSocket.on("streaming_stopped", (data) => {
      console.log("Streaming stopped:", data);
      setIsStreaming(false);
      setSessionResults(data.session_results);
    });

    newSocket.on("status_update", (status) => {
      setIsStreaming(status.is_streaming);
      setCurrentClass(status.current_class);
    });

    // Error handling
    newSocket.on("connect_error", (error) => {
      console.error("Connection error:", error);
      setError("Failed to connect to backend server");
      setIsConnected(false);
    });

    // Cleanup
    return () => {
      newSocket.close();
    };
  }, []);

  // Periodically fetch performance stats during streaming
  useEffect(() => {
    if (!isStreaming) return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch("/api/session-stats");
        if (response.ok) {
          const stats = await response.json();
          setPerformanceStats(stats);
        }
      } catch (error) {
        console.error("Error fetching stats:", error);
      }
    }, 2000); // Update every 2 seconds

    return () => clearInterval(interval);
  }, [isStreaming]);

  const handleClassSelection = async (classIndex) => {
    if (isStreaming) return; // Can't change class while streaming

    try {
      const response = await fetch("/api/set-class", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ class_index: classIndex }),
      });

      if (response.ok) {
        const result = await response.json();
        setCurrentClass(result.class_index);
        setError(null);
      } else {
        const error = await response.json();
        setError(error.error || "Failed to set class");
      }
    } catch (error) {
      console.error("Error setting class:", error);
      setError("Network error");
    }
  };

  const handleStartStreaming = async () => {
    try {
      const response = await fetch("/api/start-streaming", {
        method: "POST",
      });

      if (response.ok) {
        setError(null);
      } else {
        const error = await response.json();
        setError(error.error || "Failed to start streaming");
      }
    } catch (error) {
      console.error("Error starting streaming:", error);
      setError("Network error");
    }
  };

  const handleStopStreaming = async () => {
    try {
      const response = await fetch("/api/stop-streaming", {
        method: "POST",
      });

      if (response.ok) {
        setError(null);
      } else {
        const error = await response.json();
        setError(error.error || "Failed to stop streaming");
      }
    } catch (error) {
      console.error("Error stopping streaming:", error);
      setError("Network error");
    }
  };

  const getStatusIndicator = () => {
    if (!isConnected) {
      return (
        <div className="status-indicator disconnected">
          <div className="status-dot"></div>
          Disconnected
        </div>
      );
    }

    if (isStreaming) {
      return (
        <div className="status-indicator streaming">
          <div className="status-dot"></div>
          Streaming - {MOTOR_IMAGERY_CLASSES[currentClass]}
        </div>
      );
    }

    return (
      <div className="status-indicator connected">
        <div className="status-dot"></div>
        Connected
      </div>
    );
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🧠 BCI Real-time Classification</h1>
        <p>
          Motor Imagery Brain-Computer Interface with Real-time EEG Processing
        </p>
      </header>

      <div className="main-content">
        {/* Control Panel */}
        <div className="control-panel">
          {getStatusIndicator()}

          {error && <div className="error-message">{error}</div>}

          {/* Motor Imagery Class Selection */}
          <div className="control-section">
            <h3>Motor Imagery Class</h3>
            <div className="class-buttons">
              {Object.entries(MOTOR_IMAGERY_CLASSES).map(([index, name]) => (
                <button
                  key={index}
                  className={`class-button ${
                    parseInt(index) === currentClass ? "selected" : ""
                  } ${isStreaming ? "disabled" : ""}`}
                  onClick={() => handleClassSelection(parseInt(index))}
                  disabled={isStreaming}
                >
                  {name}
                </button>
              ))}
            </div>
          </div>

          {/* Streaming Controls */}
          <div className="control-section">
            <h3>Streaming Control</h3>
            <div className="control-buttons">
              <button
                className="btn primary"
                onClick={handleStartStreaming}
                disabled={!isConnected || isStreaming}
              >
                Start Streaming
              </button>
              <button
                className="btn danger"
                onClick={handleStopStreaming}
                disabled={!isConnected || !isStreaming}
              >
                Stop Streaming
              </button>
            </div>
          </div>

          {/* Performance Stats */}
          {(isStreaming || sessionResults) && (
            <PerformanceStats
              performanceStats={performanceStats}
              sessionResults={sessionResults}
              isStreaming={isStreaming}
            />
          )}
        </div>

        {/* Visualization Panel */}
        <div className="visualization-panel">
          {/* Current Prediction */}
          {currentPrediction && (
            <PredictionDisplay prediction={currentPrediction} />
          )}

          {/* EEG Signal Visualization */}
          {eegData.length > 0 && (
            <div className="chart-container">
              <h3 className="chart-title">
                Live EEG Signals (Motor Cortex Channels)
              </h3>
              <EEGChart data={eegData} />
            </div>
          )}

          {/* Session Results */}
          {sessionResults && !isStreaming && (
            <div className="session-results">
              <h4>Session Complete</h4>
              <div className="results-grid">
                <div className="stat-card">
                  <p className="stat-value">
                    {(sessionResults.accuracy?.accuracy * 100 || 0).toFixed(1)}%
                  </p>
                  <p className="stat-label">Session Accuracy</p>
                </div>
                <div className="stat-card">
                  <p className="stat-value">
                    {sessionResults.performance?.total_classifications || 0}
                  </p>
                  <p className="stat-label">Total Predictions</p>
                </div>
                <div className="stat-card">
                  <p className="stat-value">
                    {(sessionResults.session_duration_seconds || 0).toFixed(1)}s
                  </p>
                  <p className="stat-label">Duration</p>
                </div>
                <div className="stat-card">
                  <p className="stat-value">
                    {(
                      sessionResults.performance?.avg_processing_time_ms || 0
                    ).toFixed(1)}
                    ms
                  </p>
                  <p className="stat-label">Avg Response Time</p>
                </div>
              </div>

              {sessionResults.trained_model_comparison && (
                <div style={{ marginTop: "15px", fontSize: "0.9rem" }}>
                  <strong>vs. Trained Model:</strong> Session{" "}
                  {(
                    sessionResults.trained_model_comparison
                      .accuracy_difference * 100
                  ).toFixed(1)}
                  %{" "}
                  {sessionResults.trained_model_comparison
                    .accuracy_difference >= 0
                    ? "better"
                    : "lower"}{" "}
                  than balanced test accuracy (
                  {(
                    sessionResults.trained_model_comparison
                      .trained_balanced_accuracy * 100
                  ).toFixed(1)}
                  %)
                </div>
              )}
            </div>
          )}

          {/* No data message */}
          {!isStreaming && eegData.length === 0 && !sessionResults && (
            <div
              style={{
                textAlign: "center",
                padding: "60px 20px",
                color: "#666",
                fontSize: "1.1rem",
              }}
            >
              <p>
                🎯 Select a motor imagery class and start streaming to begin
                real-time classification
              </p>
              <p style={{ fontSize: "0.9rem", marginTop: "10px" }}>
                The system will simulate EEG signals and classify them in
                real-time using the trained EEGNet model
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
