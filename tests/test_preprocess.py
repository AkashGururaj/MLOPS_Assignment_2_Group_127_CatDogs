import os
import torch
import pytest
from src.model.model import SimpleCNN

MODEL_PATH = "tests/simple_cnn_test.pt"

@pytest.fixture(scope="module")
def dummy_model():
    model = SimpleCNN()
    torch.save(model.state_dict(), MODEL_PATH)
    yield MODEL_PATH
    os.remove(MODEL_PATH)

def test_model_loading(dummy_model):
    model = SimpleCNN()
    model.load_state_dict(torch.load(dummy_model))
    model.eval()
    # Forward a dummy tensor
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 1)
    assert (0 <= out.detach().numpy()).all()  # BCELoss output >=0
