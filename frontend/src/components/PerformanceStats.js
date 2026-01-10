import React from "react";

const MOTOR_IMAGERY_CLASSES = {
  0: "Left Hand",
  1: "Right Hand",
  2: "Foot",
  3: "Tongue",
};

function PerformanceStats({ performanceStats, sessionResults, isStreaming }) {
  if (!performanceStats && !sessionResults) {
    return null;
  }

  return (
    <div className="performance-stats">
      <h3>📊 Performance Statistics</h3>

      {/* Live Stats (during streaming) */}
      {isStreaming && performanceStats && (
        <div className="live-stats">
          <h4>Live Session</h4>

          {/* Basic Stats */}
          <div className="stats-grid">
            <div className="stat-card mini">
              <p className="stat-value">
                {performanceStats.total_classifications || 0}
              </p>
              <p className="stat-label">Predictions</p>
            </div>

            <div className="stat-card mini">
              <p className="stat-value">
                {performanceStats.session_duration
                  ? performanceStats.session_duration.toFixed(1)
                  : "0.0"}
                s
              </p>
              <p className="stat-label">Duration</p>
            </div>

            <div className="stat-card mini">
              <p className="stat-value">
                {performanceStats.avg_processing_time
                  ? performanceStats.avg_processing_time.toFixed(1)
                  : "0.0"}
                ms
              </p>
              <p className="stat-label">Avg Time</p>
            </div>

            <div className="stat-card mini">
              <p className="stat-value">
                {performanceStats.classification_rate
                  ? performanceStats.classification_rate.toFixed(1)
                  : "0.0"}{" "}
                Hz
              </p>
              <p className="stat-label">Rate</p>
            </div>
          </div>

          {/* Class Distribution */}
          {performanceStats.class_distribution && (
            <div className="class-distribution">
              <h5>Predictions by Class</h5>
              <div className="distribution-bars">
                {Object.entries(performanceStats.class_distribution).map(
                  ([classIndex, count]) => {
                    const className =
                      MOTOR_IMAGERY_CLASSES[parseInt(classIndex)];
                    const total = performanceStats.total_classifications || 1;
                    const percentage = (count / total) * 100;

                    return (
                      <div key={classIndex} className="distribution-item">
                        <div className="distribution-label">
                          <span>{className}</span>
                          <span>
                            {count} ({percentage.toFixed(1)}%)
                          </span>
                        </div>
                        <div className="distribution-bar">
                          <div
                            className="distribution-fill"
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                      </div>
                    );
                  }
                )}
              </div>
            </div>
          )}

          {/* Recent Accuracy */}
          {performanceStats.recent_accuracy !== undefined && (
            <div className="recent-accuracy">
              <h5>Recent Performance</h5>
              <div className="accuracy-display">
                <span className="accuracy-value">
                  {(performanceStats.recent_accuracy * 100).toFixed(1)}%
                </span>
                <span className="accuracy-label">
                  Last {performanceStats.accuracy_window_size || 10} predictions
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Session Results (after completion) */}
      {sessionResults && !isStreaming && (
        <div className="session-summary">
          <h4>Session Summary</h4>

          {/* Main Metrics */}
          <div className="stats-grid">
            <div className="stat-card">
              <p className="stat-value">
                {sessionResults.accuracy
                  ? (sessionResults.accuracy.accuracy * 100).toFixed(1)
                  : "0.0"}
                %
              </p>
              <p className="stat-label">Final Accuracy</p>
            </div>

            <div className="stat-card">
              <p className="stat-value">
                {sessionResults.performance?.total_classifications || 0}
              </p>
              <p className="stat-label">Total Predictions</p>
            </div>

            <div className="stat-card">
              <p className="stat-value">
                {sessionResults.session_duration_seconds
                  ? sessionResults.session_duration_seconds.toFixed(1)
                  : "0.0"}
                s
              </p>
              <p className="stat-label">Duration</p>
            </div>

            <div className="stat-card">
              <p className="stat-value">
                {sessionResults.performance?.avg_processing_time_ms
                  ? sessionResults.performance.avg_processing_time_ms.toFixed(1)
                  : "0.0"}
                ms
              </p>
              <p className="stat-label">Avg Response</p>
            </div>
          </div>

          {/* Confusion Matrix Summary */}
          {sessionResults.accuracy?.confusion_matrix && (
            <div className="confusion-summary">
              <h5>Classification Results</h5>
              <div className="per-class-stats">
                {Object.entries(
                  sessionResults.accuracy.per_class_metrics || {}
                ).map(([classIndex, metrics]) => {
                  const className = MOTOR_IMAGERY_CLASSES[parseInt(classIndex)];

                  return (
                    <div key={classIndex} className="class-metric">
                      <div className="class-name">{className}</div>
                      <div className="metrics">
                        <span>
                          Precision: {(metrics.precision * 100).toFixed(1)}%
                        </span>
                        <span>
                          Recall: {(metrics.recall * 100).toFixed(1)}%
                        </span>
                        <span>F1: {(metrics.f1_score * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Model Comparison */}
          {sessionResults.trained_model_comparison && (
            <div className="model-comparison">
              <h5>vs. Trained Model</h5>
              <div className="comparison-info">
                <div className="comparison-stat">
                  <span className="comparison-label">Session:</span>
                  <span className="comparison-value">
                    {(sessionResults.accuracy.accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="comparison-stat">
                  <span className="comparison-label">Trained (balanced):</span>
                  <span className="comparison-value">
                    {(
                      sessionResults.trained_model_comparison
                        .trained_balanced_accuracy * 100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
                <div className="comparison-stat">
                  <span className="comparison-label">Difference:</span>
                  <span
                    className="comparison-value"
                    style={{
                      color:
                        sessionResults.trained_model_comparison
                          .accuracy_difference >= 0
                          ? "#2ecc71"
                          : "#e74c3c",
                    }}
                  >
                    {sessionResults.trained_model_comparison
                      .accuracy_difference >= 0
                      ? "+"
                      : ""}
                    {(
                      sessionResults.trained_model_comparison
                        .accuracy_difference * 100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default PerformanceStats;
