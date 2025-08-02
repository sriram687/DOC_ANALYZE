"""
Root-level main.py for Render deployment compatibility
This file imports and runs the actual application from backend/main.py
"""

import sys
import os

print("🚀 Initializing Enhanced Document Query API...")
print(f"📁 Current working directory: {os.getcwd()}")
print(f"🐍 Python version: {sys.version}")

# Add backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
print(f"📂 Backend path: {backend_path}")
sys.path.insert(0, backend_path)

# Change working directory to backend for relative imports
os.chdir(backend_path)
print(f"📁 Changed to backend directory: {os.getcwd()}")

# Import and run the actual application
print("📦 Importing FastAPI application...")
from main import app
print("✅ Application imported successfully!")

if __name__ == "__main__":
    import uvicorn
    import platform
    import asyncio

    # Set event loop policy for Windows compatibility
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Get port from environment (for Render deployment) or default to 3000
    port = int(os.getenv("PORT", 3000))
    host = "0.0.0.0"  # Always bind to all interfaces for production

    print("🚀 Starting Enhanced Document Query API...")
    print(f"🌐 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🏥 Health Check: http://{host}:{port}/health")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )
