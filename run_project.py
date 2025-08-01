#!/usr/bin/env python3
"""
Startup script for Enhanced Document Query API with LangChain
Handles dependency installation, configuration validation, and application startup
"""

import os
import sys
import subprocess
import asyncio
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required dependencies"""
    logger.info("📦 Installing dependencies...")
    
    try:
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.error(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Dependency installation timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error installing dependencies: {e}")
        return False

def check_environment():
    """Check environment configuration"""
    logger.info("🔍 Checking environment configuration...")
    
    required_vars = [
        'GEMINI_API_KEY',
        'PINECONE_API_KEY',
        'PINECONE_INDEX_NAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        logger.info("Please check your .env file and ensure all required variables are set")
        return False
    
    logger.info("✅ Environment configuration looks good")
    return True

async def test_components():
    """Test key components before starting the server"""
    logger.info("🧪 Testing key components...")
    
    try:
        # Test LangChain engine initialization
        from langchain_query_engine import LangChainQueryEngine
        
        engine = LangChainQueryEngine()
        await engine.initialize()
        
        logger.info("✅ LangChain Query Engine initialized successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")
        logger.info("Check your API keys and configuration")
        return False

def start_server():
    """Start the FastAPI server"""
    logger.info("🚀 Starting Enhanced Document Query API server...")
    
    try:
        # Import uvicorn
        import uvicorn
        
        # Start the server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=3000,
            reload=True,
            log_level="info"
        )
        
    except ImportError:
        logger.error("❌ uvicorn not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uvicorn"])
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        return False

def print_startup_info():
    """Print startup information"""
    print("\n" + "="*60)
    print("🚀 ENHANCED DOCUMENT QUERY API WITH LANGCHAIN")
    print("="*60)
    print("🔗 LangChain RAG Pipeline")
    print("🤖 Google Gemini AI")
    print("📊 Pinecone Vector Database")
    print("🗄️ PostgreSQL (Optional)")
    print("⚡ Redis Caching (Optional)")
    print("="*60)
    print("\n📍 Server will be available at:")
    print("   • Main API: http://localhost:3000")
    print("   • Documentation: http://localhost:3000/docs")
    print("   • Health Check: http://localhost:3000/health")
    print("\n🔧 Key Endpoints:")
    print("   • /ask-document - Main LangChain-powered queries")
    print("   • /ask-document-langchain - Explicit LangChain endpoint")
    print("   • /analyze-document - Document analysis")
    print("   • /suggest-queries - AI-generated suggestions")
    print("\n" + "="*60)

async def main():
    """Main startup function"""
    print_startup_info()
    
    # Step 1: Install dependencies
    if not install_dependencies():
        logger.error("❌ Failed to install dependencies. Exiting.")
        sys.exit(1)
    
    # Step 2: Check environment
    if not check_environment():
        logger.error("❌ Environment check failed. Please fix configuration.")
        sys.exit(1)
    
    # Step 3: Test components
    if not await test_components():
        logger.error("❌ Component tests failed. Please check configuration.")
        sys.exit(1)
    
    # Step 4: Start server
    logger.info("🎉 All checks passed! Starting server...")
    time.sleep(2)  # Brief pause for user to read
    
    start_server()

def quick_start():
    """Quick start without full testing (for development)"""
    logger.info("⚡ Quick start mode - minimal checks")
    
    # Basic environment check
    if not os.path.exists('.env'):
        logger.warning("⚠️ No .env file found. Please create one with your API keys.")
    
    # Start server directly
    start_server()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_start()
    else:
        asyncio.run(main())
