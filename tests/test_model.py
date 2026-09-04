import torch
from src.core_models.lstm import LSTMMarketPredictor

def test_model_forward_pass():
    # Model configuration
    input_dim = 2
    hidden_dim = 64
    seq_length = 24
    batch_size = 16
    
    model = LSTMMarketPredictor(input_dim=input_dim, hidden_dim=hidden_dim)
    model.eval()
    
    # Create dummy input tensor matching expected shape (batch_size, seq_length, input_dim)
    dummy_input = torch.randn(batch_size, seq_length, input_dim)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # Output should be (batch_size, 1)
    assert output.shape == (batch_size, 1)
    
def test_model_output_scaling():
    # Just a sanity check that it runs with small batches
    model = LSTMMarketPredictor(input_dim=2)
    model.eval()
    
    dummy_input = torch.randn(1, 24, 2)
    with torch.no_grad():
        out = model(dummy_input)
        
    assert out.dim() == 2
    assert out.shape[1] == 1
