import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.data.data_pipeline import generate_synthetic_data, preprocess_data, create_sequences
from src.core_models.lstm import LSTMMarketPredictor

def train_model(
    data_path: str = "data/market_data.csv",
    model_save_path: str = "models/best_model.pt",
    seq_length: int = 24,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 10
):
    print("Preparing data for training...")
    if not os.path.exists(data_path):
        generate_synthetic_data(num_samples=5000, save_path=data_path)
        
    df_scaled, _ = preprocess_data(data_path, "models/scaler.joblib")
    
    X, y = create_sequences(df_scaled.values, seq_length)
    
    # Train/Val/Test split (70/15/15)
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = LSTMMarketPredictor(input_dim=X.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Saved new best model with Val Loss: {val_loss:.5f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    print(f"Training complete. Best model saved at {model_save_path}")

if __name__ == "__main__":
    train_model()
