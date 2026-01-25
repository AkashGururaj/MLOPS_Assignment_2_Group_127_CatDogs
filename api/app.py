import time
import logging
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
import os
import mlflow

from model_inference import load_model, predict, log_predictions

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

app = FastAPI(title="Cats vs Dogs Inference API")

# Load model at startup
model = load_model("models/simple_cnn.pt")

# Request counter
REQUEST_COUNT = 0

@app.middleware("http")
async def log_requests(request: Request, call_next):
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(
        f"Request {REQUEST_COUNT}: {request.method} {request.url} | "
        f"Status: {response.status_code} | Latency: {process_time:.4f}s"
    )
    return response

@app.get("/health")
async def health():
    return {"status": "ok", "requests_served": REQUEST_COUNT}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), true_label: str = None):
    try:
        bytes_data = await file.read()
        label, prob = predict(model, bytes_data)
        log_predictions(label, prob, true_label)
        return {"label": label, "probability": prob}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
