from fastapi import FastAPI, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.model.model import SimpleCNN

# Initialize FastAPI
app = FastAPI()

# Device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained SimpleCNN model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/simple_cnn.pt", map_location=device))
model.eval()  # important for inference

# Image transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile):
    try:
        # Open image
        img = Image.open(file.file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            prob = model(img_tensor).item()  # single value
        label = "dog" if prob > 0.5 else "cat"

        return {
            "prediction": label,
            "probability": prob
        }
    except Exception as e:
        return {"error": str(e)}
