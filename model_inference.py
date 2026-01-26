import os
import torch
from PIL import Image
import io
from torchvision import transforms
import pandas as pd
from datetime import datetime
import mlflow

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to saved model
MODEL_PATH = "models/simple_cnn.pt"

# ===============================
# Load saved model
# ===============================
def load_model(model_path=MODEL_PATH):
    from src.model.model import SimpleCNN
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load model at module import
model = load_model(MODEL_PATH)


# ===============================
# Predict single image
# ===============================
def predict(file_bytes):
    # Load image
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        prob = model(tensor).item()

    # Label
    label = "Dog" if prob > 0.5 else "Cat"
    return label, prob


# ===============================
# Log predictions locally + optional MLflow
# ===============================
def log_predictions(label, prob, true_label=None):
    os.makedirs("predictions", exist_ok=True)
    timestamp = datetime.now().isoformat()
    
    # Save locally
    df = pd.DataFrame([{
        "timestamp": timestamp,
        "predicted_label": label,
        "predicted_prob": prob,
        "true_label": true_label
    }])
    file_path = "predictions/predictions_log.csv"
    df.to_csv(file_path, mode="a", header=not os.path.exists(file_path), index=False)

    # Optional MLflow logging
    mlflow.set_experiment("cats_vs_dogs_post_deployment")
    mlflow.log_metric("predicted_prob", prob)
    if true_label is not None:
        mlflow.log_metric("correct", int(label == true_label))
