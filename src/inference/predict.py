from fastapi import FastAPI, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.model.model import SimpleCNN

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/simple_cnn.pt", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(file.file).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(img).item()
    return {"prediction": "dog" if prob>0.5 else "cat", "probability": prob}
