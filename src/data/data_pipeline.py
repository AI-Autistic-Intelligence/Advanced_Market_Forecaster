import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def generate_synthetic_data(num_samples: int = 5000, save_path: str = "data/market_data.csv") -> pd.DataFrame:
    """
    Generate synthetic financial time series data mimicking price and volume.
    Uses a random walk with drift and volatility clustering (GARCH-like).
    """
    print(f"Generating {num_samples} synthetic samples...")
    np.random.seed(42)
    
    # Time index
    dates = pd.date_range(start="2020-01-01", periods=num_samples, freq="1H")
    
    # Generate prices using geometric brownian motion-like process
    returns = np.random.normal(loc=0.0001, scale=0.01, size=num_samples)
    
    # Introduce volatility clustering
    volatility = np.zeros(num_samples)
    volatility[0] = 0.01
    for i in range(1, num_samples):
        volatility[i] = np.sqrt(0.1 * 0.01**2 + 0.85 * volatility[i-1]**2 + 0.05 * returns[i-1]**2)
        returns[i] = np.random.normal(loc=0.0, scale=volatility[i])
        
    prices = 100.0 * np.exp(np.cumsum(returns))
    
    # Generate volume correlated with volatility
    volume = np.abs(np.random.normal(loc=1000, scale=200, size=num_samples)) * (1 + 10 * volatility)
    
    df = pd.DataFrame({
        "close": prices,
        "volume": volume
    }, index=dates)
    
    # Create target (e.g. price in 10 steps) for reference, though we'll do this in the dataloader
    # Save raw data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(f"Data saved to {save_path}")
    return df

def preprocess_data(csv_path: str = "data/market_data.csv", 
                    scaler_path: str = "models/scaler.pkl") -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Load data, scale it, and save the scaler for inference.
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"Data scaled. Scaler saved to {scaler_path}")
    return df_scaled, scaler

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a time series into sequences of length seq_length.
    X shape: (num_samples, seq_length, num_features)
    y shape: (num_samples, )
    We try to predict the 'close' price (column 0) at the next time step.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0] # Predicting the close price
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    generate_synthetic_data()
    preprocess_data()
