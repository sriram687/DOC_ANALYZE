#!/bin/bash

# Render startup script for Enhanced Document Query API
echo "🚀 Starting Enhanced Document Query API on Render..."

# Check if we're in the right directory
echo "📁 Current directory: $(pwd)"
echo "📂 Directory contents:"
ls -la

# Start the FastAPI application using our root main.py
echo "🎯 Starting application with Python..."
exec python main.py
