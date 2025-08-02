"""
Root-level main.py for Render deployment compatibility
This file imports and runs the actual application from backend/main.py
"""

import sys
import os

# Add backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

# Change working directory to backend for relative imports
os.chdir(backend_path)

# Import and run the actual application
from main import app

if __name__ == "__main__":
    import uvicorn
    import platform
    import asyncio
    
    # Set event loop policy for Windows compatibility
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Get port from environment (for Render deployment) or default to 3000
    port = int(os.getenv("PORT", 3000))
    host = "0.0.0.0" if os.getenv("RENDER") else "127.0.0.1"

    print("🚀 Starting Enhanced Document Query API...")
    print(f"📚 Access API documentation at: http://{host}:{port}/docs")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
