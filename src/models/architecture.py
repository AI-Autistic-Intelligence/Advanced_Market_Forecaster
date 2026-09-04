import torch
import torch.nn as nn

class LSTMMarketPredictor(nn.Module):
    """
    LSTM-based time-series forecasting model.
    Takes a sequence of historical prices/volumes and predicts the next price step.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1, dropout: float = 0.2):
        super(LSTMMarketPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize hidden state and cell state automatically (zeros if not provided)
        out, _ = self.lstm(x)
        
        # Take the output of the last time step
        out = out[:, -1, :]
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
