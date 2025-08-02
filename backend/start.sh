#!/bin/bash

# Render startup script for Enhanced Document Query API

echo "🚀 Starting Enhanced Document Query API on Render..."

# Install any additional system dependencies if needed
# (Render handles Python dependencies via requirements.txt)

# Start the FastAPI application with Gunicorn
exec gunicorn main:app \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --log-level info
