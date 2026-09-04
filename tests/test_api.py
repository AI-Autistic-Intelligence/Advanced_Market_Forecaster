from fastapi.testclient import TestClient
from src.api.main import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    # Even if model isn't loaded (during basic CI test), it should return healthy 200
    assert "model_loaded" in data

def test_predict_invalid_shape():
    # Missing sequence
    response = client.post("/predict", json={})
    assert response.status_code == 422 # Pydantic validation error
    
    # Wrong shape (e.g., 2 steps instead of 24)
    response = client.post("/predict", json={"sequence": [[1.0, 100.0], [1.1, 110.0]]})
    assert response.status_code in [400, 503] # 503 if model not loaded, 400 if bad shape

def test_predict_mocked_model(monkeypatch):
    # We can mock the global model and scaler to test the endpoint logic without training
    import src.api.main as api_main
    
    class MockModel:
        def __call__(self, x):
            import torch
            return torch.tensor([[0.5]])
            
    class MockScaler:
        def transform(self, x):
            return x
        def inverse_transform(self, x):
            return x
            
    monkeypatch.setattr(api_main, "model", MockModel())
    monkeypatch.setattr(api_main, "scaler", MockScaler())
    
    # Provide exactly 24 steps
    dummy_sequence = [[float(i), float(i*10)] for i in range(24)]
    
    response = client.post("/predict", json={"sequence": dummy_sequence})
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_close" in data
    assert data["predicted_close"] == 0.5
