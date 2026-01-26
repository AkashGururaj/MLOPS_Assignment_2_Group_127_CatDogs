# ===============================
# Base image
# ===============================
FROM python:3.11-bullseye

# ===============================
# Set working directory
# ===============================
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt . 
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# Copy project files
# ===============================
COPY . .

# ===============================
# Expose FastAPI port
# ===============================
EXPOSE 8000

# ===============================
# Start FastAPI with Uvicorn
# ===============================
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
