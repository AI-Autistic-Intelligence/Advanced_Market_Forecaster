import os
import joblib
import torch
import numpy as np
import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from prometheus_fastapi_instrumentator import Instrumentator

from src.core_models.lstm import LSTMMarketPredictor
from src.api.config import settings

logger = structlog.get_logger()

app = FastAPI(title="Advanced Market Forecaster API", 
              description="Serving a PyTorch LSTM model for time-series forecasting",
              version="1.0.0")

Instrumentator().instrument(app).expose(app)

# Global variables to hold model and scaler
model = None
scaler = None
device = torch.device("cpu")

class PredictionRequest(BaseModel):
    # Expecting a flat list or list of lists for 24 time steps (close, volume)
    sequence: List[List[float]] 

class PredictionResponse(BaseModel):
    predicted_close: float

@app.on_event("startup")
def load_assets():
    global model, scaler, device
    
    logger.info("Starting up API, loading assets...")
    
    if os.path.exists(settings.scaler_path):
        scaler = joblib.load(settings.scaler_path)
        logger.info(f"Scaler loaded from {settings.scaler_path}")
    else:
        logger.warning(f"Scaler not found at {settings.scaler_path}. Inference may fail.")
        
    if os.path.exists(settings.model_path):
        model = LSTMMarketPredictor(input_dim=2)
        model.load_state_dict(torch.load(settings.model_path, map_location=device))
        model.eval()
        model.to(device)
        logger.info(f"Model loaded from {settings.model_path}")
    else:
        logger.warning(f"Model weights not found at {settings.model_path}. Please train the model first.")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "scaler_loaded": scaler is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or scaler is None:
        logger.error("Predict called but Model or Scaler not loaded.")
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded.")
    
    seq = np.array(request.sequence)
    if seq.shape != (settings.seq_length, 2):
        logger.warning(f"Invalid sequence shape: expected ({settings.seq_length}, 2), got {seq.shape}")
        raise HTTPException(status_code=400, detail=f"Expected sequence of shape ({settings.seq_length}, 2), got {seq.shape}")
    
    try:
        # Scale input
        scaled_seq = scaler.transform(seq)
        
        # Convert to tensor and add batch dimension
        tensor_seq = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(tensor_seq)
            pred_val = output.item()
            
        # Inverse transform the prediction
        # We need to create a dummy array to inverse transform just the 'close' price
        dummy = np.zeros((1, 2))
        dummy[0, 0] = pred_val
        inv_pred = scaler.inverse_transform(dummy)[0, 0]
        
        logger.info("Prediction successful", predicted_close=float(inv_pred))
        return PredictionResponse(predicted_close=float(inv_pred))
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
