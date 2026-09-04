import os
import pickle
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.models.architecture import LSTMMarketPredictor

app = FastAPI(title="Advanced Market Forecaster API", 
              description="Serving a PyTorch LSTM model for time-series forecasting",
              version="1.0.0")

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
    
    scaler_path = "models/scaler.pkl"
    model_path = "models/best_model.pt"
    
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        print(f"Warning: Scaler not found at {scaler_path}. Inference may fail.")
        
    if os.path.exists(model_path):
        model = LSTMMarketPredictor(input_dim=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
    else:
        print(f"Warning: Model weights not found at {model_path}. Please train the model first.")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "scaler_loaded": scaler is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded.")
    
    seq = np.array(request.sequence)
    if seq.shape != (24, 2):
        raise HTTPException(status_code=400, detail=f"Expected sequence of shape (24, 2), got {seq.shape}")
    
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
        
        return PredictionResponse(predicted_close=float(inv_pred))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
