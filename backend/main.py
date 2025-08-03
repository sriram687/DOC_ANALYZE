"""
Enhanced Document Query API - Main Application
Clean, production-ready FastAPI application with LangChain RAG pipeline
"""

import asyncio
import json
import logging
import re
import time
import uuid
import hashlib
import os
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
# Document Analysis Functions
# =============================================================================

async def process_document_questions(content: str, questions: list) -> list:
    """
    Process document content and questions using LangChain + Gemini
    Returns a list of 10 answers extracted from the document
    """
    try:
        logger.info("🧠 Starting document analysis with Gemini")

        # Create a concise prompt for the LLM
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

        prompt = f"""Extract answers from this document for 10 questions. Return ONLY a JSON object.

DOCUMENT:
{content}

QUESTIONS:
{questions_text}

Return exactly this JSON format:
{{"answers": ["answer1", "answer2", "answer3", "answer4", "answer5", "answer6", "answer7", "answer8", "answer9", "answer10"]}}

Extract exact text from document. If not found, use "Information not available"."""

        # Use direct Google Generative AI client for better control
        import google.generativeai as genai
        from config.config import config

        # Configure and create direct Gemini client
        genai.configure(api_key=config.GEMINI_API_KEY)
        # Use Gemini 1.5 Pro for more predictable token usage
        model = genai.GenerativeModel("gemini-1.5-pro")

        # Generate response with low temperature for factual accuracy
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Very low for factual accuracy
                top_p=0.8,
                top_k=40,
                max_output_tokens=8192,  # Much higher for Gemini 2.5 Pro thoughts
            )
        )

        # Extract the response text
        response_text = response.text.strip() if response.text else ""
        logger.info(f"🤖 Gemini response received: {len(response_text)} characters")

        # Log the actual response for debugging
        if response_text:
            logger.info(f"📝 Response preview: {response_text[:200]}...")
        else:
            logger.error("❌ Empty response from Gemini")
            # Try to get more info about the response
            logger.error(f"Response object: {response}")
            if hasattr(response, 'candidates'):
                logger.error(f"Candidates: {response.candidates}")
            if hasattr(response, 'prompt_feedback'):
                logger.error(f"Prompt feedback: {response.prompt_feedback}")

        # Parse JSON response
        if not response_text:
            logger.error("❌ Empty response from Gemini - using fallback answers")
            answers = ["Information not available in the document"] * 10
        else:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_response = json.loads(json_str)
                answers = parsed_response.get("answers", [])
            else:
                # Fallback: try to parse the entire response as JSON
                parsed_response = json.loads(response_text)
                answers = parsed_response.get("answers", [])

        # Validate we have exactly 10 answers
        if len(answers) != 10:
            logger.error(f"❌ Expected 10 answers, got {len(answers)}")
            # Pad or trim to ensure exactly 10 answers
            if len(answers) < 10:
                answers.extend(["Information not available in the document"] * (10 - len(answers)))
            else:
                answers = answers[:10]

        logger.info(f"✅ Successfully processed {len(answers)} answers")
        return answers

    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parsing error: {str(e)}")
        # Return fallback answers
        return ["Error processing document - JSON parsing failed"] * 10

    except Exception as e:
        logger.error(f"❌ Document processing error: {str(e)}")
        # Return fallback answers
        return ["Error processing document"] * 10

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

@app.post("/webhook")
async def webhook_listener(request: Request):
    """
    HackRx Document Analysis Webhook
    Processes document content and questions for evaluation

    Expected payload:
    {
        "type": "document_analysis",
        "content": "Full document text...",
        "questions": ["Q1", "Q2", ..., "Q10"],
        "filename": "document.pdf",
        "user_id": "user123",
        "timestamp": "2025-08-03T18:00:00Z"
    }

    Returns:
    {
        "answers": ["Answer1", "Answer2", ..., "Answer10"]
    }
    """
    try:
        # Parse JSON payload
        payload = await request.json()

        # Log webhook data
        logger.info(f"🔗 Webhook received from {request.client.host}")
        logger.info(f"📦 Processing document analysis request")

        # Check if this is a document analysis request
        webhook_type = payload.get("type", "unknown")

        if webhook_type == "document_analysis":
            # Extract required fields
            content = payload.get("content", "")
            questions = payload.get("questions", [])
            filename = payload.get("filename", "document.pdf")
            user_id = payload.get("user_id", "anonymous")

            # Validate input
            if not content:
                return JSONResponse(
                    content={"error": "Document content is required"},
                    status_code=400
                )

            if not questions or len(questions) != 10:
                return JSONResponse(
                    content={"error": "Exactly 10 questions are required"},
                    status_code=400
                )

            logger.info(f"📄 Processing document: {filename}")
            logger.info(f"👤 User: {user_id}")
            logger.info(f"❓ Questions: {len(questions)}")

            # Initialize the API if not already done
            await enhanced_api.initialize()

            # Process document and questions using LangChain
            answers = await process_document_questions(content, questions)

            # Return only the answers array as required by HackRx evaluator
            return JSONResponse(
                content={"answers": answers},
                status_code=200
            )

        else:
            # Handle other webhook types (generic processing)
            logger.info(f"🔔 Generic webhook processed: {webhook_type}")

            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Webhook received and processed",
                    "webhook_type": webhook_type,
                    "timestamp": datetime.now().isoformat()
                },
                status_code=200
            )

    except Exception as e:
        logger.error(f"❌ Webhook processing error: {str(e)}")
        return JSONResponse(
            content={
                "error": "Webhook processing failed",
                "details": str(e)
            },
            status_code=500
        )

@app.get("/webhook/test")
async def webhook_test():
    """Test endpoint to verify webhook functionality"""
    return {
        "message": "Webhook endpoint is active",
        "webhook_url": "https://doc-analyze.onrender.com/webhook",
        "methods": ["POST"],
        "content_type": "application/json",
        "supported_types": {
            "document_analysis": "HackRx document analysis with 10 questions",
            "generic": "Generic webhook processing"
        },
        "example_payloads": {
            "document_analysis": {
                "type": "document_analysis",
                "filename": "policy.pdf",
                "content": "Document content here...",
                "questions": [
                    "Question 1?",
                    "Question 2?",
                    "... (10 questions total)"
                ],
                "user_id": "user123",
                "timestamp": datetime.now().isoformat()
            },
            "generic": {
                "type": "test",
                "message": "Hello webhook!",
                "timestamp": datetime.now().isoformat()
            }
        },
        "expected_response": {
            "document_analysis": {
                "answers": ["Answer 1", "Answer 2", "... (10 answers total)"]
            },
            "generic": {
                "status": "success",
                "message": "Webhook received and processed"
            }
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Document Query API with LangChain",
        "endpoints": {
            "POST /ask-document": "Submit document and query (LangChain RAG)",
            "POST /ask-document-clean": "Submit document and query (Clean format)",
            "POST /webhook": "Generic webhook endpoint for external services",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "supported_formats": [".pdf", ".docx", ".eml", ".txt", ".png", ".jpg"],
        "features": [
            "LangChain RAG Pipeline",
            "Google Gemini 2.5 Pro",
            "Pinecone Vector Store",
            "Professional Response Formatting",
            "Webhook Integration"
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
