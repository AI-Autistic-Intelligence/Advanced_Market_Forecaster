# Advanced Market Forecaster

An End-to-End MLOps project demonstrating a production-ready Machine Learning pipeline for Time-Series forecasting. 

This project simulates a realistic Senior ML Engineer workflow by covering data generation, deep learning model architecture, model training, API serving, containerization, and interactive visualization.

## 🚀 Features

- **Synthetic Market Data Generator**: Generates simulated financial time-series data to train the model without relying on external APIs.
- **Deep Learning Model**: A PyTorch-based Long Short-Term Memory (LSTM) network designed for sequence prediction.
- **Robust API Serving**: A high-performance FastAPI server utilizing `gunicorn`, `joblib` for secure serialization, and `pydantic-settings` for external configuration.
- **Prometheus Metrics**: Automatic `/metrics` endpoint exposed via `prometheus-fastapi-instrumentator`.
- **Streamlit Dashboard**: A fully interactive web application with Manual, Step-by-Step, and Auto modes for real-time inference and truth verification against historical data.
- **Comprehensive Testing**: Pytest suite for both model correctness and API endpoints.
- **Production-Ready Docker**: Multi-stage, non-root Docker builds for security and efficiency.

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
│   ├── api/
│   │   ├── main.py              # FastAPI server
│   │   └── config.py            # Pydantic environment configurations
│   └── dashboard/
│       └── app.py               # Streamlit interactive UI
│
├── tests/
│   ├── test_model.py            # Unit tests for PyTorch models
│   └── test_api.py              # Integration tests for FastAPI
│
├── docs_theme/                  # Docusaurus documentation submodule
├── data/                        # Generated datasets (git ignored)
├── models/                      # Serialized model weights (git ignored)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker image definition (Multi-stage)
├── .dockerignore                # Docker exclusion rules
└── docker-compose.yml           # Compose configuration for API and Dashboard
```

## 🛠️ Setup & Installation

### Option 1: Docker (Recommended)

Make sure you have Docker and Docker Compose installed.

1. Generate the initial training data and model artifacts locally first (requires Python environment):
   ```bash
   pip install -r requirements.txt
   python -m src.models.train
   ```

2. Start the API and Dashboard services:
   ```bash
   docker-compose up --build
   ```

3. Access the services:
   - **Dashboard**: `http://localhost:8501`
   - **API Swagger UI**: `http://localhost:8000/docs`
   - **Prometheus Metrics**: `http://localhost:8000/metrics`

### Option 2: Local Python Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python -m src.data.data_pipeline
   python -m src.models.train
   ```
4. Start the API:
   ```bash
   python -m uvicorn src.api.main:app --port 8000
   ```
5. Start the Dashboard (in a new terminal):
   ```bash
   python -m streamlit run src.dashboard.app.py --server.port 8501
   ```

## 📚 Documentation

Detailed architectural and technical documentation is maintained within the `docs_theme` directory, which is configured as a [Docusaurus](https://docusaurus.io/) git submodule.

To initialize and view the documentation locally:

```bash
# Initialize the submodule if you haven't already
git submodule update --init

cd docs_theme
npm install
npm run start
```
*Note: Ensure you have Node.js installed to run Docusaurus.*

## 🧪 Testing

Run the test suite using pytest:
```bash
pytest tests/ -v
```
