import React from "react";

const MOTOR_IMAGERY_CLASSES = {
  0: "Left Hand",
  1: "Right Hand",
  2: "Foot",
  3: "Tongue",
};

const CLASS_COLORS = {
  0: "#e74c3c", // Left Hand - Red
  1: "#3498db", // Right Hand - Blue
  2: "#2ecc71", // Foot - Green
  3: "#f39c12", // Tongue - Orange
};

const CLASS_ICONS = {
  0: "✋", // Left Hand
  1: "👋", // Right Hand
  2: "🦶", // Foot
  3: "👅", // Tongue
};

function PredictionDisplay({ prediction }) {
  if (!prediction) {
    return (
      <div className="prediction-display">
        <div className="prediction-header">
          <h3>Waiting for prediction...</h3>
        </div>
      </div>
    );
  }

  const {
    predicted_class,
    probabilities,
    confidence,
    processing_time_ms,
    timestamp,
  } = prediction;

  const predictedClassName = MOTOR_IMAGERY_CLASSES[predicted_class];
  const predictedColor = CLASS_COLORS[predicted_class];
  const predictedIcon = CLASS_ICONS[predicted_class];

  // Format timestamp
  const predictionTime = new Date(timestamp).toLocaleTimeString();

  // Get confidence level description
  const getConfidenceLevel = (conf) => {
    if (conf >= 0.8) return { level: "High", color: "#2ecc71" };
    if (conf >= 0.6) return { level: "Medium", color: "#f39c12" };
    return { level: "Low", color: "#e74c3c" };
  };

  const confidenceInfo = getConfidenceLevel(confidence);

  return (
    <div className="prediction-display">
      <div className="prediction-header">
        <h3>🎯 Real-time Classification</h3>
        <div className="prediction-time">{predictionTime}</div>
      </div>

      {/* Main Prediction */}
      <div className="main-prediction">
        <div
          className="predicted-class"
          style={{ borderLeft: `4px solid ${predictedColor}` }}
        >
          <div className="class-info">
            <span className="class-icon">{predictedIcon}</span>
            <span className="class-name">{predictedClassName}</span>
          </div>
          <div className="confidence-info">
            <span
              className="confidence-value"
              style={{ color: confidenceInfo.color }}
            >
              {(confidence * 100).toFixed(1)}%
            </span>
            <span
              className="confidence-level"
              style={{ color: confidenceInfo.color }}
            >
              {confidenceInfo.level}
            </span>
          </div>
        </div>
      </div>

      {/* Probability Bars */}
      <div className="probability-bars">
        <h4>Class Probabilities</h4>
        {Object.entries(MOTOR_IMAGERY_CLASSES).map(
          ([classIndex, className]) => {
            const prob = probabilities[parseInt(classIndex)] || 0;
            const color = CLASS_COLORS[parseInt(classIndex)];
            const icon = CLASS_ICONS[parseInt(classIndex)];
            const isWinner = parseInt(classIndex) === predicted_class;

            return (
              <div
                key={classIndex}
                className={`probability-bar ${isWinner ? "winner" : ""}`}
              >
                <div className="bar-label">
                  <span className="bar-icon">{icon}</span>
                  <span className="bar-name">{className}</span>
                  <span className="bar-value">{(prob * 100).toFixed(1)}%</span>
                </div>
                <div className="bar-container">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${prob * 100}%`,
                      backgroundColor: color,
                      opacity: isWinner ? 1 : 0.7,
                    }}
                  />
                </div>
              </div>
            );
          }
        )}
      </div>

      {/* Processing Stats */}
      <div className="processing-stats">
        <div className="stat-item">
          <span className="stat-label">Processing Time:</span>
          <span className="stat-value">{processing_time_ms.toFixed(1)}ms</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Classification Rate:</span>
          <span className="stat-value">
            {processing_time_ms > 0
              ? (1000 / processing_time_ms).toFixed(1)
              : "N/A"}{" "}
            Hz
          </span>
        </div>
      </div>
    </div>
  );
}

export default PredictionDisplay;
