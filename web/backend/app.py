"""FastAPI server for rPPG model inference."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from pathlib import Path
from pydantic import BaseModel

app = FastAPI(title="rPPG Clinical Model API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and device
model = None
device = None


class PredictionResponse(BaseModel):
    """Prediction response with confidence scores."""
    heart_rate: float
    heart_rate_confidence: float
    respiratory_rate: float
    respiratory_rate_confidence: float
    spo2: float
    spo2_confidence: float


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        from src.models.architecture import rPPGModel
        model = rPPGModel(feature_dim=256, hidden_dim=128).to(device)

        # Load trained weights if available
        model_path = Path("models/trained_model.pt")
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            print("No trained model found. Using untrained model for demo.")

        model.eval()
        print("Model loaded and ready for inference.")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "device": str(device)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(frames_tensor: torch.Tensor):
    """
    Predict biomarkers from video frames.

    frames_tensor: (1, 30, 3, 128, 128) prepared tensor
    """
    if model is None:
        return PredictionResponse(
            heart_rate=0, heart_rate_confidence=0,
            respiratory_rate=0, respiratory_rate_confidence=0,
            spo2=0, spo2_confidence=0
        )

    try:
        frames_tensor = frames_tensor.to(device)

        with torch.no_grad():
            hr_val, hr_conf, rr_val, rr_conf, spo2_val, spo2_conf = model(frames_tensor)

        return PredictionResponse(
            heart_rate=float(hr_val[0].item()),
            heart_rate_confidence=float(hr_conf[0].item()),
            respiratory_rate=float(rr_val[0].item()),
            respiratory_rate_confidence=float(rr_conf[0].item()),
            spo2=float(spo2_val[0].item()),
            spo2_confidence=float(spo2_conf[0].item())
        )
    except Exception as e:
        print(f"Prediction error: {e}")
        return PredictionResponse(
            heart_rate=0, heart_rate_confidence=0,
            respiratory_rate=0, respiratory_rate_confidence=0,
            spo2=0, spo2_confidence=0
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
