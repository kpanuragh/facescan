import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const handleStartCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      videoRef.current.srcObject = stream;
    } catch (err) {
      setError("Camera access denied: " + err.message);
    }
  };

  const handleCapture = async () => {
    if (!videoRef.current) return;

    setLoading(true);
    setError(null);

    try {
      // Capture 30 frames from video
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const frames = [];

      for (let i = 0; i < 30; i++) {
        ctx.drawImage(videoRef.current, 0, 0, 128, 128);
        const imageData = ctx.getImageData(0, 0, 128, 128);
        frames.push(imageData.data);
      }

      // Send to backend
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames })
      });

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError("Prediction error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderConfidence = (conf) => {
    const percent = Math.round(conf * 100);
    const color = conf > 0.8 ? '#4CAF50' : conf > 0.6 ? '#FFC107' : '#F44336';
    return (
      <div className="confidence-bar">
        <div
          className="confidence-fill"
          style={{ width: `${percent}%`, backgroundColor: color }}
        />
        <span className="confidence-text">{percent}%</span>
      </div>
    );
  };

  return (
    <div className="App">
      <header className="header">
        <h1>rPPG Clinical Model</h1>
        <p>Real-time Health Biomarker Estimation from Facial Video</p>
      </header>

      <main className="main">
        <section className="video-section">
          <h2>Webcam Capture</h2>
          <video
            ref={videoRef}
            width="640"
            height="480"
            autoPlay
            playsInline
            className="video-feed"
          />
          <canvas
            ref={canvasRef}
            width="128"
            height="128"
            style={{ display: 'none' }}
          />
          <div className="button-group">
            <button onClick={handleStartCamera} className="btn btn-primary">
              Start Camera
            </button>
            <button
              onClick={handleCapture}
              className="btn btn-success"
              disabled={loading}
            >
              {loading ? 'Processing...' : 'Capture & Analyze'}
            </button>
          </div>
        </section>

        {error && <div className="error-message">{error}</div>}

        {results && (
          <section className="results-section">
            <h2>Results</h2>
            <div className="metrics-grid">
              <div className="metric-card">
                <h3>Heart Rate</h3>
                <p className="metric-value">{results.heart_rate.toFixed(1)} bpm</p>
                {renderConfidence(results.heart_rate_confidence)}
              </div>
              <div className="metric-card">
                <h3>Respiratory Rate</h3>
                <p className="metric-value">
                  {results.respiratory_rate.toFixed(1)} breaths/min
                </p>
                {renderConfidence(results.respiratory_rate_confidence)}
              </div>
              <div className="metric-card">
                <h3>Blood Oxygen (SpO2)</h3>
                <p className="metric-value">{results.spo2.toFixed(1)} %</p>
                {renderConfidence(results.spo2_confidence)}
              </div>
            </div>
          </section>
        )}

        <section className="disclaimer">
          <h3>⚠️ Disclaimer</h3>
          <p>
            This tool is for research and wellness purposes only. Not intended for
            medical diagnosis or treatment. Always consult healthcare professionals
            for medical decisions.
          </p>
        </section>
      </main>
    </div>
  );
}

export default App;
