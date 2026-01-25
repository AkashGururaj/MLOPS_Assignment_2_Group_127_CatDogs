import torch
from PIL import Image
import io
from torchvision import transforms
import os
import mlflow
import pandas as pd
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path):
    from src.model.model import SimpleCNN
    model = SimpleCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(tensor).item()
    label = "Dog" if prob > 0.5 else "Cat"
    return label, prob

def log_predictions(label, prob, true_label=None):
    os.makedirs("predictions", exist_ok=True)
    timestamp = datetime.now().isoformat()
    df = pd.DataFrame([{
        "timestamp": timestamp,
        "predicted_label": label,
        "predicted_prob": prob,
        "true_label": true_label
    }])
    df.to_csv("predictions/predictions_log.csv", mode="a",
              header=not os.path.exists("predictions/predictions_log.csv"), index=False)
    # Optional MLflow logging
    mlflow.set_experiment("cats_vs_dogs_post_deployment")
    mlflow.log_metric("predicted_prob", prob)
    if true_label is not None:
        mlflow.log_metric("correct", int(label == true_label))
