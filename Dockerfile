# ===============================
# Base image
# ===============================
FROM python:3.10-slim

# ===============================
# Set working directory
# ===============================
WORKDIR /app

# ===============================
# Install minimal system deps
# ===============================
RUN apt-get clean && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Install Python deps
# ===============================
COPY requirements_api.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements_api.txt

# ===============================
# Copy source code
# ===============================
COPY src ./src
COPY api ./api

# ===============================
# Expose FastAPI port
# ===============================
EXPOSE 8000

# ===============================
# Run FastAPI
# ===============================
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
