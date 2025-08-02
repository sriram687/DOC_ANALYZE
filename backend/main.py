"""
Enhanced Document Query API - Main Application
Clean, production-ready FastAPI application with LangChain RAG pipeline
"""

import asyncio
import logging
import time
import uuid
import hashlib
import os
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Local imports
from src.models.models import QueryResponse, CleanQueryResponse
from src.services.document_processor import AdvancedDocumentProcessor
from src.services.langchain_query_engine import LangChainQueryEngine
from config.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDocumentQueryAPI:
    """Simplified API class focused on LangChain RAG functionality"""
    
    def __init__(self):
        self.document_processor = AdvancedDocumentProcessor()
        self.langchain_engine = LangChainQueryEngine()
        self._initialized = False
        logger.info("🔗 Using LangChain Query Engine")

    async def initialize(self):
        """Initialize the API components"""
        if not self._initialized:
            await self.langchain_engine.initialize()
            self._initialized = True
            logger.info("API components initialized successfully")

    async def process_query_langchain(self, query: str, file: UploadFile) -> QueryResponse:
        """Enhanced query processing using LangChain RAG pipeline"""
        start_time = time.time()
        
        # Ensure initialization
        await self.initialize()
        
        try:
            # Step 1: Document Processing
            logger.info("📄 Extracting text from document...")
            document_text = await self.document_processor.process_document(file)
            
            if not document_text or len(document_text.strip()) < 10:
                return QueryResponse(
                    query=query,
                    answer="The document appears to be empty or could not be processed.",
                    conditions=[],
                    evidence=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Generate document ID and metadata
            document_id = str(uuid.uuid4())
            content_hash = hashlib.sha256(document_text.encode()).hexdigest()
            
            metadata = {
                "filename": file.filename,
                "content_hash": content_hash,
                "upload_time": datetime.now().isoformat(),
                "file_size": len(document_text)
            }
            
            logger.info(f"📝 Document processed: {len(document_text)} characters")
            
            # Step 3: Add document to LangChain vector store
            logger.info("🔗 Adding document to LangChain vector store...")
            await self.langchain_engine.add_document(document_text, document_id, metadata)
            
            # Step 4: Query the document using LangChain
            logger.info("🤖 Processing query with LangChain RAG...")
            response = await self.langchain_engine.query_document(query, document_id)
            
            logger.info(f"✅ Query processed successfully in {response.processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in LangChain query processing: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return QueryResponse(
                query=query,
                answer=f"I encountered an error while processing your query: {str(e)}. Please try again or contact support.",
                conditions=[],
                evidence=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    async def process_query_clean(self, query: str, file: UploadFile) -> CleanQueryResponse:
        """Process query and return clean, formatted response without evidence"""
        # Use the LangChain method but return clean response
        full_response = await self.process_query_langchain(query, file)
        
        return CleanQueryResponse(
            query=full_response.query,
            answer=full_response.answer,
            conditions=full_response.conditions,
            confidence=full_response.confidence,
            processing_time=full_response.processing_time
        )

# =============================================================================
# FastAPI Application Setup
# =============================================================================

# Initialize API with LangChain
enhanced_api = EnhancedDocumentQueryAPI()
app = FastAPI(
    title="Enhanced Document Query API with LangChain",
    description="AI-powered document analysis using LangChain RAG, Google Gemini, and Pinecone",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/ask-document", response_model=QueryResponse)
async def ask_document(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """Main endpoint for document queries using LangChain RAG"""
    return await enhanced_api.process_query_langchain(query, file)

@app.post("/ask-document-clean", response_model=CleanQueryResponse)
async def ask_document_clean(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """Clean, formatted document queries using LangChain RAG"""
    return await enhanced_api.process_query_clean(query, file)

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "model": "gemini-2.5-pro",
        "langchain": "enabled",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Document Query API with LangChain",
        "endpoints": {
            "POST /ask-document": "Submit document and query (LangChain RAG)",
            "POST /ask-document-clean": "Submit document and query (Clean format)",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "supported_formats": [".pdf", ".docx", ".eml", ".txt", ".png", ".jpg"],
        "features": [
            "LangChain RAG Pipeline",
            "Google Gemini 2.5 Pro",
            "Pinecone Vector Store",
            "Professional Response Formatting"
        ]
    }

# =============================================================================
# Application Startup
# =============================================================================

if __name__ == "__main__":
    import platform
    
    # Set event loop policy for Windows compatibility
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        logger.info("🔧 Set Windows event loop policy for compatibility")

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Warning: GEMINI_API_KEY not set or using placeholder value")

    # Get port from environment (for Render deployment) or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0" if os.getenv("RENDER") else "127.0.0.1"

    print("🚀 Starting Enhanced Document Query API...")
    print(f"📚 Access API documentation at: http://{host}:{port}/docs")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
