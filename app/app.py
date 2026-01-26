import os
import time
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from model_inference import predict, log_predictions  # no need to import load_model now

# ===============================
# Logging setup
# ===============================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ===============================
# FastAPI app
# ===============================
app = FastAPI(title="Cats vs Dogs Inference API")

# ===============================
# Request counter
# ===============================
REQUEST_COUNT = 0

@app.middleware("http")
async def log_requests(request, call_next):
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    logging.info(
        f"Request {REQUEST_COUNT}: {request.method} {request.url} | "
        f"Status: {response.status_code} | Latency: {latency:.4f}s"
    )
    return response

# ===============================
# Health check endpoint
# ===============================
@app.get("/health")
async def health():
    return {"status": "ok", "requests_served": REQUEST_COUNT}

# ===============================
# Prediction endpoint
# ===============================
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), true_label: str = None):
    try:
        file_bytes = await file.read()
        label, prob = predict(file_bytes)  # model already loaded inside model_inference
        log_predictions(label, prob, true_label)
        return {"predicted_label": label, "probability": prob}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
