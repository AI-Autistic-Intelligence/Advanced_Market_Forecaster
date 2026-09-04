# Advanced Market Forecaster

An End-to-End MLOps project demonstrating a production-ready Machine Learning pipeline for Time-Series forecasting. 

This project aims to simulate a realistic Senior ML Engineer workflow by covering data generation, deep learning model architecture, model training, API serving, containerization, and automated testing.

## 🚀 Features

- **Synthetic Market Data Generator**: Generates simulated financial time-series data to train the model without relying on external APIs.
- **Deep Learning Model**: A PyTorch-based Long Short-Term Memory (LSTM) network designed for sequence prediction.
- **FastAPI Serving Layer**: A high-performance REST API for serving model predictions.
- **Comprehensive Testing**: Pytest suite for both model correctness and API endpoints.
- **Dockerized**: Easy deployment using Docker and Docker Compose.

## 📂 Project Structure

```
Advanced_Market_Forecaster/
│
├── src/
│   ├── data/
│   │   └── data_pipeline.py     # Data generation & preprocessing
│   ├── models/
│   │   ├── architecture.py      # PyTorch model definitions
│   │   └── train.py             # Training loop
│   └── api/
│       └── main.py              # FastAPI server
│
├── tests/
│   ├── test_model.py            # Unit tests for PyTorch models
│   └── test_api.py              # Integration tests for FastAPI
│
├── data/                        # Generated datasets (ignored in git)
├── models/                      # Serialized model weights (ignored in git)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker image definition
└── docker-compose.yml           # Compose configuration
```

## 🛠️ Setup & Installation

### Option 1: Local Python Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Docker

Make sure you have Docker installed.
```bash
docker-compose up --build
```

## 🏃‍♂️ Running the Pipeline

If running locally (Option 1):

**1. Generate Data:**
```bash
python -m src.data.data_pipeline
```
This will create `data/market_data.csv` and a scaler artifact.

**2. Train the Model:**
```bash
python -m src.models.train
```
This will train the LSTM and save weights to `models/best_model.pt`.

**3. Run the API:**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Access the Swagger UI at `http://localhost:8000/docs`.

## 🧪 Testing

Run the test suite using pytest:
```bash
pytest tests/ -v
```
