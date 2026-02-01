import os
import time
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from sklearn.metrics import accuracy_score, precision_score, recall_score

from model_inference import predict, log_predictions


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "api.log")

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)

logger.addHandler(file_handler)

logger.info("API STARTED")


# FastAPI app

app = FastAPI(title="Cats vs Dogs Inference API")


# Global counters & buffers

REQUEST_COUNT = 0

PERF_TRACKING_BATCH = []
PERF_BATCH_SIZE = 50


# Middleware: request logging

@app.middleware("http")
async def log_requests(request, call_next):
    global REQUEST_COUNT
    REQUEST_COUNT += 1

    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    logger.info(
        f"Request {REQUEST_COUNT} | "
        f"{request.method} {request.url.path} | "
        f"Status={response.status_code} | "
        f"Latency={latency:.4f}s"
    )

    return response


# Health check

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "requests_served": REQUEST_COUNT
    }


# Prediction endpoint

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    true_label: str | None = None
):
    try:
        file_bytes = await file.read()

        predicted_label, probability = predict(file_bytes)
        log_predictions(predicted_label, probability, true_label)

        logger.info(
            f"Prediction | Pred={predicted_label} | "
            f"Prob={probability:.4f} | True={true_label}"
        )

        if true_label is not None:
            PERF_TRACKING_BATCH.append((true_label, predicted_label))

            if len(PERF_TRACKING_BATCH) >= PERF_BATCH_SIZE:
                evaluate_performance(PERF_TRACKING_BATCH)
                PERF_TRACKING_BATCH.clear()

        return {
            "predicted_label": predicted_label,
            "probability": probability
        }

    except Exception as e:
        logger.exception("Prediction error")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# Post-deployment performance evaluation

def evaluate_performance(batch):
    true_labels, pred_labels = zip(*batch)

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(
        true_labels, pred_labels,
        average="macro", zero_division=0
    )
    recall = recall_score(
        true_labels, pred_labels,
        average="macro", zero_division=0
    )

    logger.info(
        f"POST-DEPLOYMENT METRICS | "
        f"Samples={len(batch)} | "
        f"Accuracy={accuracy:.4f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f}"
    )
