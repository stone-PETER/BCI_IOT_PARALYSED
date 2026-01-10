import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

const EEG_CHANNELS = ["C3", "Cz", "C4", "CP1", "CP2", "P3", "Pz", "P4"]; // Motor cortex channels
const CHANNEL_COLORS = [
  "#e74c3c", // C3 - Red
  "#3498db", // Cz - Blue
  "#2ecc71", // C4 - Green
  "#f39c12", // CP1 - Orange
  "#9b59b6", // CP2 - Purple
  "#1abc9c", // P3 - Turquoise
  "#34495e", // Pz - Dark Blue
  "#e67e22", // P4 - Dark Orange
];

function EEGChart({ data }) {
  // Process data for chart display
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];

    // Get the last 2 seconds of data (500 samples at 250Hz)
    const displaySamples = 500;
    const recentData = data.slice(-displaySamples);

    // Create chart data with relative time
    return recentData.map((sample, index) => {
      const timeInSeconds = (index - displaySamples) / 250; // Convert to seconds from current time

      const chartPoint = { time: timeInSeconds };

      // Add available channel data
      EEG_CHANNELS.forEach((channel) => {
        const channelKey = `ch${channel}`;
        if (sample[channelKey] !== undefined) {
          chartPoint[channel] = sample[channelKey];
        }
      });

      return chartPoint;
    });
  }, [data]);

  // Custom tick formatter for time axis
  const formatXTick = (value) => {
    return `${value.toFixed(1)}s`;
  };

  // Custom tick formatter for amplitude axis
  const formatYTick = (value) => {
    return `${value.toFixed(1)}`;
  };

  // Calculate Y-axis domain
  const yDomain = useMemo(() => {
    if (chartData.length === 0) return [-100, 100];

    let min = Infinity;
    let max = -Infinity;

    chartData.forEach((point) => {
      EEG_CHANNELS.forEach((channel) => {
        if (point[channel] !== undefined) {
          min = Math.min(min, point[channel]);
          max = Math.max(max, point[channel]);
        }
      });
    });

    // Add some padding
    const padding = (max - min) * 0.1;
    return [min - padding, max + padding];
  }, [chartData]);

  if (chartData.length === 0) {
    return (
      <div
        style={{
          height: "400px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#f8f9fa",
          border: "1px dashed #dee2e6",
          borderRadius: "4px",
          color: "#6c757d",
        }}
      >
        <p>Waiting for EEG data...</p>
      </div>
    );
  }

  return (
    <div style={{ height: "400px", width: "100%" }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={chartData}
          margin={{
            top: 10,
            right: 30,
            left: 40,
            bottom: 60,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis
            dataKey="time"
            type="number"
            scale="linear"
            domain={["dataMin", "dataMax"]}
            tickFormatter={formatXTick}
            tick={{ fontSize: 12 }}
            axisLine={{ stroke: "#666" }}
            tickLine={{ stroke: "#666" }}
          />
          <YAxis
            domain={yDomain}
            tickFormatter={formatYTick}
            tick={{ fontSize: 12 }}
            axisLine={{ stroke: "#666" }}
            tickLine={{ stroke: "#666" }}
            label={{
              value: "Amplitude (μV)",
              angle: -90,
              position: "insideLeft",
              style: { textAnchor: "middle", fontSize: "12px" },
            }}
          />

          {/* Render line for each available channel */}
          {EEG_CHANNELS.map((channel, index) => {
            // Check if this channel has data
            const hasData = chartData.some(
              (point) => point[channel] !== undefined
            );
            if (!hasData) return null;

            return (
              <Line
                key={channel}
                type="monotone"
                dataKey={channel}
                stroke={CHANNEL_COLORS[index]}
                strokeWidth={1.5}
                dot={false}
                connectNulls={false}
                name={channel}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>

      {/* Channel Legend */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "15px",
          marginTop: "10px",
          justifyContent: "center",
          padding: "10px",
          background: "#f8f9fa",
          borderRadius: "4px",
        }}
      >
        {EEG_CHANNELS.map((channel, index) => {
          // Check if this channel has data
          const hasData = chartData.some(
            (point) => point[channel] !== undefined
          );

          return (
            <div
              key={channel}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "5px",
                opacity: hasData ? 1 : 0.3,
                fontSize: "12px",
              }}
            >
              <div
                style={{
                  width: "12px",
                  height: "2px",
                  backgroundColor: CHANNEL_COLORS[index],
                }}
              />
              <span>{channel}</span>
            </div>
          );
        })}
      </div>

      <div
        style={{
          textAlign: "center",
          fontSize: "11px",
          color: "#666",
          marginTop: "5px",
        }}
      >
        Motor Cortex Channels • Real-time at 250Hz • Last 2 seconds
      </div>
    </div>
  );
}

export default EEGChart;
