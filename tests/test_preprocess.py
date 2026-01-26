# tests/test_model.py
import os
import torch
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.model import SimpleCNN

MODEL_PATH = "models/simple_cnn.pt"

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not found, please train first")
def test_model_loading():
    # Initialize model and load weights
    model = SimpleCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Forward a dummy tensor
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    # Assertions
    assert out.shape == (1, 1)
    assert torch.all((out >= 0) & (out <= 1)), "Output not in [0,1] range"
