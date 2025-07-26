# Document Query API System with Google Gemini
# Full-stack implementation for processing legal, insurance, HR, and compliance documents

import os
import json
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging

# Core dependencies
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Document processing
import  fitz  # pymupdf
from docx import Document
import email
from email.policy import default

# Google AI and embeddings
import google.generativeai as genai
from google.cloud import aiplatform
import numpy as np
import faiss

# Text processing
import re
from sentence_transformers import SentenceTransformer
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Configuration and Models
# =============================================================================

@dataclass
class Config:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_CHUNKS: int = 10
    CONFIDENCE_THRESHOLD: float = 0.7

config = Config()

class QueryRequest(BaseModel):
    query: str

class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class ClauseMatch(BaseModel):
    clause_id: str
    text: str
    relevance_score: float
    metadata: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    query: str
    answer: str
    conditions: List[str]
    evidence: List[Dict[str, Any]]
    confidence: float
    processing_time: float

# =============================================================================
# Document Ingestion Module
# =============================================================================

class DocumentProcessor:
    """Handles extraction of text from various document formats"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.eml', '.txt']
    
    async def process_document(self, file: UploadFile) -> str:
        """Extract text from uploaded document"""
        if file.size > config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in self.supported_formats:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        content = await file.read()
        
        try:
            if file_extension == '.pdf':
                return self._extract_pdf_text(content)
            elif file_extension == '.docx':
                return self._extract_docx_text(content)
            elif file_extension == '.eml':
                return self._extract_email_text(content)
            elif file_extension == '.txt':
                return content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error processing {file_extension} file: {str(e)}")
            raise HTTPException(status_code=422, detail=f"Error processing file: {str(e)}")
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF using PyMuPDF"""
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return self._clean_text(text)
    
    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX file"""
        from io import BytesIO
        doc = Document(BytesIO(content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return self._clean_text(text)
    
    def _extract_email_text(self, content: bytes) -> str:
        """Extract text from email file"""
        msg = email.message_from_bytes(content, policy=default)
        text = ""
        
        # Extract basic headers
        text += f"Subject: {msg.get('Subject', 'No Subject')}\n"
        text += f"From: {msg.get('From', 'Unknown')}\n"
        text += f"Date: {msg.get('Date', 'Unknown')}\n\n"
        
        # Extract body
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    text += part.get_content() + "\n"
        else:
            text += msg.get_content() + "\n"
        
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/\\\$\%\&\*\+\=\@\#]', ' ', text)
        return text.strip()

# =============================================================================
# LLM Parser Module (Gemini)
# =============================================================================

class GeminiParser:
    """Handles Gemini API interactions for query parsing and logic evaluation"""
    
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    async def parse_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract structured intent from natural language query"""
        prompt = f"""
        Analyze the following natural language query and extract structured information:
        
        Query: "{query}"
        
        Please return a JSON object with the following structure:
        {{
            "intent": "coverage_check|policy_lookup|compliance_check|general_inquiry",
            "target": "specific item/procedure/condition being asked about",
            "focus": ["list", "of", "key", "aspects", "to", "focus", "on"],
            "question_type": "yes_no|conditions|explanation|comparison",
            "entities": ["list", "of", "named", "entities", "found"]
        }}
        
        Only return the JSON object, no additional text.
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500
                )
            )
            return json.loads(response.text.strip())
        except Exception as e:
            logger.error(f"Error parsing query intent: {str(e)}")
            # Fallback basic parsing
            return {
                "intent": "general_inquiry",
                "target": query[:50],
                "focus": ["coverage", "conditions"],
                "question_type": "explanation",
                "entities": []
            }
    
    async def evaluate_clause_relevance(self, query: str, chunks: List[str]) -> List[Dict[str, Any]]:
        """Use Gemini to assess and rank chunk relevance"""
        prompt = f"""
        Query: "{query}"
        
        Please evaluate the relevance of each text chunk to the query and extract key information:
        
        Text Chunks:
        {chr(10).join([f"CHUNK_{i}: {chunk}" for i, chunk in enumerate(chunks)])}
        
        For each chunk, return a JSON array with objects containing:
        {{
            "chunk_id": "CHUNK_X",
            "relevance_score": 0.0-1.0,
            "key_phrases": ["relevant", "phrases", "found"],
            "contains_conditions": true/false,
            "summary": "brief summary of relevant content"
        }}
        
        Only return the JSON array, no additional text.
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2000
                )
            )
            return json.loads(response.text.strip())
        except Exception as e:
            logger.error(f"Error evaluating clause relevance: {str(e)}")
            # Fallback scoring
            return [{"chunk_id": f"CHUNK_{i}", "relevance_score": 0.5, 
                    "key_phrases": [], "contains_conditions": False, 
                    "summary": "Could not analyze"} for i in range(len(chunks))]
    
    async def generate_final_answer(self, query: str, relevant_chunks: List[Dict[str, Any]], 
                                  query_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final structured answer based on relevant chunks"""
        chunks_text = "\n".join([f"CLAUSE_{chunk['chunk_id']}: {chunk['text']}" 
                                for chunk in relevant_chunks])
        
        prompt = f"""
        Based on the following relevant document clauses, provide a comprehensive answer to the user's query.
        
        Query: "{query}"
        Query Intent: {json.dumps(query_intent)}
        
        Relevant Clauses:
        {chunks_text}
        
        Please provide a response in the following JSON format:
        {{
            "answer": "Direct answer to the query",
            "conditions": ["list", "of", "any", "conditions", "or", "requirements"],
            "evidence": [
                {{
                    "clause_id": "unique_identifier",
                    "text": "relevant excerpt from the clause",
                    "relevance": "why this clause is relevant"
                }}
            ],
            "confidence": 0.0-1.0,
            "caveats": ["any", "limitations", "or", "uncertainties"]
        }}
        
        Be thorough but concise. If the answer cannot be determined from the provided clauses, state that clearly.
        Only return the JSON object, no additional text.
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1500
                )
            )
            return json.loads(response.text.strip())
        except Exception as e:
            logger.error(f"Error generating final answer: {str(e)}")
            return {
                "answer": "Unable to process the query due to technical issues.",
                "conditions": [],
                "evidence": [],
                "confidence": 0.0,
                "caveats": ["Technical error occurred during processing"]
            }

# =============================================================================
# Embedding and Search Module
# =============================================================================

class EmbeddingSearchEngine:
    """Handles document chunking, embedding, and similarity search"""
    
    def __init__(self):
        # Use sentence-transformers for embeddings (alternative to text-embedding-gecko)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.chunks: List[DocumentChunk] = []
    
    def chunk_document(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Split document into semantic chunks"""
        # Simple sentence-based chunking with overlap
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        chunk_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > config.CHUNK_SIZE and current_chunk:
                # Create chunk
                chunk_id = f"{document_id}_{len(chunks)}"
                chunk = DocumentChunk(
                    id=chunk_id,
                    text=current_chunk.strip(),
                    metadata={
                        "document_id": document_id,
                        "chunk_index": len(chunks),
                        "sentence_count": len(chunk_sentences)
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = chunk_sentences[-2:] if len(chunk_sentences) >= 2 else chunk_sentences
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                chunk_sentences = overlap_sentences + [sentence]
            else:
                current_chunk += " " + sentence
                chunk_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunk_id = f"{document_id}_{len(chunks)}"
            chunk = DocumentChunk(
                id=chunk_id,
                text=current_chunk.strip(),
                metadata={
                    "document_id": document_id,
                    "chunk_index": len(chunks),
                    "sentence_count": len(chunk_sentences)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def add_document(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Process document and add to search index"""
        chunks = self.chunk_document(text, document_id)
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = await asyncio.to_thread(self.embedding_model.encode, texts)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
            self.chunks.append(chunk)
        
        logger.info(f"Added {len(chunks)} chunks to search index for document {document_id}")
        return chunks
    
    async def search_similar_chunks(self, query: str, k: int = None) -> List[DocumentChunk]:
        """Find most similar chunks to query"""
        if k is None:
            k = config.TOP_K_CHUNKS
        
        if not self.chunks:
            return []
        
        # Generate query embedding
        query_embedding = await asyncio.to_thread(self.embedding_model.encode, [query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return matching chunks
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and score > 0.3:  # Minimum similarity threshold
                chunk = self.chunks[idx]
                results.append(chunk)
        
        return results

# =============================================================================
# Main API Application
# =============================================================================

class DocumentQueryAPI:
    """Main API class orchestrating all components"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.gemini_parser = GeminiParser()
        self.search_engine = EmbeddingSearchEngine()
        
    async def process_query(self, query: str, file: UploadFile) -> QueryResponse:
        """Main processing pipeline"""
        import time
        start_time = time.time()
        
        try:
            # Step 1: Document Ingestion
            logger.info("Extracting text from document...")
            document_text = await self.document_processor.process_document(file)
            
            # Step 2: Create search index
            document_id = str(uuid.uuid4())
            logger.info("Creating embeddings and search index...")
            await self.search_engine.add_document(document_text, document_id)
            
            # Step 3: Parse query intent
            logger.info("Parsing query intent...")
            query_intent = await self.gemini_parser.parse_query_intent(query)
            
            # Step 4: Retrieve relevant chunks
            logger.info("Searching for relevant document chunks...")
            relevant_chunks = await self.search_engine.search_similar_chunks(query)
            
            if not relevant_chunks:
                return QueryResponse(
                    query=query,
                    answer="No relevant information found in the document.",
                    conditions=[],
                    evidence=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Step 5: Evaluate chunk relevance with Gemini
            logger.info("Evaluating chunk relevance...")
            chunk_texts = [chunk.text for chunk in relevant_chunks]
            relevance_scores = await self.gemini_parser.evaluate_clause_relevance(query, chunk_texts)
            
            # Step 6: Prepare data for final answer generation
            scored_chunks = []
            for i, chunk in enumerate(relevant_chunks):
                score_data = relevance_scores[i] if i < len(relevance_scores) else {}
                scored_chunks.append({
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "relevance_score": score_data.get("relevance_score", 0.5),
                    "metadata": chunk.metadata
                })
            
            # Sort by relevance and take top chunks
            scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
            top_chunks = scored_chunks[:5]  # Top 5 most relevant
            
            # Step 7: Generate final answer
            logger.info("Generating final answer...")
            final_result = await self.gemini_parser.generate_final_answer(
                query, top_chunks, query_intent
            )
            
            # Step 8: Format response
            processing_time = time.time() - start_time
            
            return QueryResponse(
                query=query,
                answer=final_result.get("answer", "Unable to generate answer"),
                conditions=final_result.get("conditions", []),
                evidence=final_result.get("evidence", []),
                confidence=final_result.get("confidence", 0.0),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# =============================================================================
# FastAPI Application Setup
# =============================================================================

# Initialize API
api_handler = DocumentQueryAPI()
app = FastAPI(
    title="Document Query API",
    description="AI-powered document analysis using Google Gemini",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask-document", response_model=QueryResponse)
async def ask_document(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Main endpoint for document queries
    
    - **query**: Natural language question about the document
    - **file**: Document file (PDF, DOCX, EML, TXT)
    """
    return await api_handler.process_query(query, file)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "gemini-1.5-pro"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document Query API with Google Gemini",
        "endpoints": {
            "POST /ask-document": "Submit document and query",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "supported_formats": [".pdf", ".docx", ".eml", ".txt"]
    }

# =============================================================================
# CLI Runner (for development)
# =============================================================================

if __name__ == "__main__":
    # Ensure required environment variables
    if not config.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable is required")
        print("Please set it with your Google AI API key")
        exit(1)
    
    print("Starting Document Query API...")
    print(f"Supported formats: {DocumentProcessor().supported_formats}")
    print("Access API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",  # Assuming this file is named main.py
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )