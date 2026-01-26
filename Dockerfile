# ===============================
# Base image
# ===============================
FROM python:3.11-slim

# ===============================
# Set working directory
# ===============================
WORKDIR /app

# ===============================
# Install system dependencies
# Needed for PyTorch, FastAPI, and other packages
# ===============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    libopenblas-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Copy requirements and install Python packages
# ===============================
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

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
