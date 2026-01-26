#!/bin/bash
set -e   # FAIL IMMEDIATELY if any command fails

API_URL="http://127.0.0.1:8000"

echo "🔍 Health check..."
curl -f "$API_URL/health"

echo "🧪 Prediction test..."
curl -f -X POST "$API_URL/predict" \
  -F "file=@tests/cat.jpg"

echo "Smoke tests passed"
