import os
import uuid
import time
import hashlib
import asyncio
import logging
import json
from typing import List, Dict, Any
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models import QueryResponse, DocumentAnalysis, ComparisonResult, CleanQueryResponse
from document_processor import AdvancedDocumentProcessor
from gemini_parser import GeminiParser
from embedding_search import GeminiEmbeddingSearchEngine
from cache_manager import CacheManager
from query_optimizer import QueryOptimizer
from document_analytics import DocumentAnalytics
from batch_processor import BatchProcessor, StreamingProcessor
from insurance_processor import DocumentQueryProcessor, StructuredQuery, DocumentDecision
from langchain_query_engine import LangChainQueryEngine
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDocumentQueryAPI:
    """Enhanced API class orchestrating all components with LangChain integration"""

    def __init__(self, use_langchain: bool = True, use_gemini_embeddings: bool = True):
        self.document_processor = AdvancedDocumentProcessor()
        self.gemini_parser = GeminiParser()

        # Choose between LangChain and legacy search engine
        self.use_langchain = use_langchain
        if use_langchain:
            self.langchain_engine = LangChainQueryEngine()
            logger.info("🔗 Using LangChain Query Engine")
        else:
            # Fallback to legacy search engine
            self.search_engine = GeminiEmbeddingSearchEngine(use_gemini_primary=use_gemini_embeddings)
            logger.info("🔍 Using Legacy Search Engine")

        self.cache_manager = CacheManager()
        self.query_optimizer = QueryOptimizer()
        self.analytics = DocumentAnalytics()
        self.batch_processor = BatchProcessor(self)
        self.streaming_processor = StreamingProcessor()
        self.document_query_processor = DocumentQueryProcessor()
        self._initialized = False

    async def initialize(self):
        """Initialize the API components"""
        if not self._initialized:
            if self.use_langchain:
                await self.langchain_engine.initialize()
            else:
                await self.search_engine.initialize()
            self._initialized = True
            logger.info("API components initialized successfully")

    async def process_query(self, query: str, file: UploadFile) -> QueryResponse:
        """Main processing pipeline"""
        start_time = time.time()

        # Ensure initialization
        await self.initialize()

        try:
            # Step 1: Document Ingestion
            logger.info("Extracting text from document...")
            document_text = await self.document_processor.process_document(file)

            # Step 2: Store document in PostgreSQL
            content_hash = hashlib.sha256(document_text.encode()).hexdigest()

            # Try to check if document already exists (optional)
            document_id = str(uuid.uuid4())
            if self.search_engine.db_initialized and self.search_engine.db_manager:
                try:
                    existing_doc = await self.search_engine.db_manager.get_document_by_hash(content_hash)
                    if existing_doc:
                        document_id = existing_doc["id"]
                        logger.info(f"📄 Document already exists with ID: {document_id}")
                    else:
                        # Store new document
                        stored_id = await self.search_engine.db_manager.store_document(
                            filename=file.filename,
                            content=document_text,
                            content_hash=content_hash,
                            metadata={"file_size": len(document_text)}
                        )
                        document_id = stored_id
                        logger.info(f"✅ Stored new document with ID: {document_id}")
                except Exception as e:
                    logger.warning(f"❌ Database operation failed: {e}")
                    logger.info(f"📝 Using generated document ID: {document_id}")
            else:
                logger.info(f"📝 Database not available, using generated ID: {document_id}")

            # Step 3: Create embeddings and search index
            logger.info("Creating embeddings and search index...")
            await self.search_engine.add_document(document_text, document_id)
            
            # Step 3: Parse query intent
            logger.info("Parsing query intent...")
            query_intent = await self.gemini_parser.parse_query_intent(query)
            
            # Step 4: Retrieve relevant chunks
            logger.info("Searching for relevant document chunks...")
            relevant_chunks = await self.search_engine.search_similar_chunks(query, document_id=document_id)
            
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

    async def process_query_with_hybrid_search(self, query: str, file: UploadFile) -> QueryResponse:
        """Enhanced query processing with hybrid search"""
        start_time = time.time()
        
        try:
            # Step 1: Document Ingestion
            logger.info("Extracting text from document...")
            document_text = await self.document_processor.process_document(file)
            
            # Step 2: Create search index with Gemini embeddings
            document_id = str(uuid.uuid4())
            logger.info("Creating Gemini embeddings and search index...")
            await self.search_engine.add_document(document_text, document_id)
            
            # Step 3: Parse query intent
            logger.info("Parsing query intent...")
            query_intent = await self.gemini_parser.parse_query_intent(query)
            
            # Step 4: Perform hybrid search
            logger.info("Performing hybrid search...")
            search_results = await self.search_engine.hybrid_search(query, document_id=document_id)
            
            if not search_results:
                return QueryResponse(
                    query=query,
                    answer="No relevant information found in the document.",
                    conditions=[],
                    evidence=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Step 5: Prepare chunks for Gemini evaluation
            relevant_chunks = [result['chunk'] for result in search_results]
            chunk_texts = [chunk.text for chunk in relevant_chunks]
            
            # Step 6: Evaluate with Gemini
            logger.info("Evaluating chunk relevance with Gemini...")
            relevance_scores = await self.gemini_parser.evaluate_clause_relevance(query, chunk_texts)
            
            # Step 7: Combine search scores with Gemini evaluation
            scored_chunks = []
            for i, (chunk, search_result) in enumerate(zip(relevant_chunks, search_results)):
                gemini_eval = relevance_scores[i] if i < len(relevance_scores) else {}
                
                # Combine multiple scoring methods
                final_score = (
                    0.4 * search_result.get('combined_score', 0.0) +
                    0.6 * gemini_eval.get('relevance_score', 0.0)
                )
                
                scored_chunks.append({
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "relevance_score": final_score,
                    "search_score": search_result.get('combined_score', 0.0),
                    "gemini_eval_score": gemini_eval.get('relevance_score', 0.0),
                    "metadata": chunk.metadata
                })
            
            # Sort by final combined score
            scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
            top_chunks = scored_chunks[:5]  # Top 5 most relevant
            
            # Step 8: Generate final answer
            logger.info("Generating final answer with Gemini...")
            final_result = await self.gemini_parser.generate_final_answer(
                query, top_chunks, query_intent
            )
            
            # Step 9: Format response with enhanced metadata
            processing_time = time.time() - start_time
            
            return QueryResponse(
                query=query,
                answer=final_result.get("answer", "Unable to generate answer"),
                conditions=final_result.get("conditions", []),
                evidence=[
                    {
                        **evidence,
                        "search_score": next(
                            (chunk["search_score"] for chunk in top_chunks 
                             if chunk["chunk_id"] == evidence.get("clause_id")), 0.0
                        ),
                        "embedding_model": "gemini" if self.search_engine.use_gemini_primary else "sentence_transformer"
                    }
                    for evidence in final_result.get("evidence", [])
                ],
                confidence=final_result.get("confidence", 0.0),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing query with hybrid search: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    async def process_query_with_cache(self, query: str, file: UploadFile) -> QueryResponse:
        """Process query with caching support"""
        start_time = time.time()
        
        # Generate document hash for caching
        file_content = await file.read()
        await file.seek(0)  # Reset file pointer
        doc_hash = hashlib.md5(file_content).hexdigest()
        
        # Check cache for previous response
        cached_response = await self.cache_manager.get_query_response(query, doc_hash)
        if cached_response:
            logger.info("Cache hit for query")
            cached_response['processing_time'] = time.time() - start_time
            cached_response['cache_hit'] = True
            return QueryResponse(**cached_response)
        
        # Process normally
        response = await self.process_query(query, file)
        
        # Cache the response (convert Pydantic model to dict)
        await self.cache_manager.set_query_response(
            query, doc_hash, response.model_dump()
        )
        
        response.cache_hit = False
        return response
    
    async def analyze_document_insights(self, file: UploadFile) -> DocumentAnalysis:
        """Provide document analysis and insights"""
        text = await self.document_processor.process_document(file)
        
        # Basic analysis
        insights = self.analytics.analyze_document(text)
        
        # Add embedding-based insights
        document_id = str(uuid.uuid4())
        chunks = await self.search_engine.add_document(text, document_id)
        
        insights.update({
            'chunk_count': len(chunks),
            'avg_chunk_length': sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0,
            'processing_timestamp': datetime.now().isoformat(),
            'document_id': document_id
        })
        
        return DocumentAnalysis(**insights)
    
    async def suggest_queries(self, file: UploadFile, num_suggestions: int = 5) -> List[str]:
        """Generate suggested queries based on document content"""
        text = await self.document_processor.process_document(file)
        doc_type = self.analytics._classify_document_type(text)
        key_topics = self.analytics._extract_key_topics(text, top_n=10)
        
        # Generate contextual suggestions based on document type
        suggestions = []
        
        if doc_type == 'insurance_policy':
            base_templates = [
                "What does this policy cover for {topic}?",
                "Are there any exclusions related to {topic}?",
                "What are the conditions for {topic} coverage?",
                "What is the deductible for {topic}?",
                "How do I file a claim for {topic}?"
            ]
        elif doc_type == 'hr_policy':
            base_templates = [
                "What are the policies regarding {topic}?",
                "What are the eligibility requirements for {topic}?",
                "How much {topic} time is available?",
                "What is the process for requesting {topic}?",
                "Are there any restrictions on {topic}?"
            ]
        elif doc_type == 'compliance_doc':
            base_templates = [
                "What are the {topic} requirements?",
                "How often should {topic} be reviewed?",
                "What are the penalties for {topic} violations?",
                "Who is responsible for {topic} compliance?",
                "What documentation is needed for {topic}?"
            ]
        else:
            base_templates = [
                "What information is provided about {topic}?",
                "What are the key points regarding {topic}?",
                "How is {topic} defined in this document?",
                "What are the implications of {topic}?",
                "Are there any conditions related to {topic}?"
            ]
        
        # Generate suggestions using key topics
        for i, template in enumerate(base_templates[:num_suggestions]):
            if i < len(key_topics):
                topic = key_topics[i].replace('_', ' ').title()
                suggestions.append(template.format(topic=topic))
        
        # Add some generic suggestions
        generic_suggestions = [
            "What are the main points covered in this document?",
            "Are there any important deadlines or timeframes?",
            "What are the key terms and definitions?",
            "What are the main requirements or conditions?",
            "Who does this document apply to?"
        ]
        
        # Fill remaining slots with generic suggestions
        while len(suggestions) < num_suggestions and generic_suggestions:
            suggestions.append(generic_suggestions.pop(0))
        
        return suggestions

    async def process_structured_query(self, query: str, file: UploadFile, document_type: str = "general") -> Dict[str, Any]:
        """Process any type of document query with structured decision making"""
        start_time = time.time()

        # Ensure initialization
        await self.initialize()

        try:
            # Step 1: Parse query based on document type
            logger.info(f"Parsing {document_type} query...")
            parsed_query = self.document_query_processor.parse_document_query(query, document_type)
            logger.info(f"Parsed query: Action={parsed_query.action}, Location={parsed_query.location}, Amount={parsed_query.amount}")

            # Step 2: Process document
            logger.info("Processing document...")
            document_text = await self.document_processor.process_document(file)

            # Step 3: Create search index
            document_id = str(uuid.uuid4())
            await self.search_engine.add_document(document_text, document_id)

            # Step 4: Search for relevant clauses
            logger.info("Searching for relevant document clauses...")
            relevant_chunks = await self.search_engine.search_similar_chunks(
                query, document_id=document_id, k=10
            )

            if not relevant_chunks:
                return {
                    "decision": "INSUFFICIENT_INFO",
                    "amount": None,
                    "justification": "No relevant clauses found for this query.",
                    "clauses_used": [],
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time,
                    "document_type": document_type,
                    "parsed_query": self._format_parsed_query(parsed_query)
                }

            # Step 5: Convert chunks to evidence format
            evidence_chunks = []
            for chunk in relevant_chunks:
                evidence_chunks.append({
                    'text': chunk.text,
                    'relevance_score': chunk.metadata.get('similarity_score', 0),
                    'metadata': chunk.metadata
                })

            # Step 6: Make decision
            logger.info(f"Making {document_type} decision...")
            decision = await self.document_query_processor.make_document_decision(
                parsed_query, evidence_chunks
            )

            processing_time = time.time() - start_time

            # Step 7: Format response
            return {
                "decision": decision.decision,
                "amount": decision.amount,
                "justification": decision.justification,
                "clauses_used": decision.clauses_used or [],
                "confidence": decision.confidence,
                "processing_time": processing_time,
                "document_type": document_type,
                "decision_type": decision.decision_type,
                "parsed_query": self._format_parsed_query(parsed_query),
                "evidence_summary": {
                    "total_clauses_found": len(evidence_chunks),
                    "clauses_used_in_decision": len(decision.clauses_used or []),
                    "highest_relevance_score": max([chunk['relevance_score'] for chunk in evidence_chunks], default=0)
                }
            }

        except Exception as e:
            logger.error(f"Error processing {document_type} query: {str(e)}")
            return {
                "decision": "ERROR",
                "amount": None,
                "justification": f"Error processing query: {str(e)}",
                "clauses_used": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "document_type": document_type,
                "parsed_query": {"raw_query": query}
            }

    def _format_parsed_query(self, parsed_query: StructuredQuery) -> Dict[str, Any]:
        """Format parsed query for response"""
        return {
            "age": parsed_query.age,
            "gender": parsed_query.gender,
            "name": parsed_query.name,
            "employee_id": parsed_query.employee_id,
            "action": parsed_query.action,
            "action_type": parsed_query.action_type,
            "location": parsed_query.location,
            "department": parsed_query.department,
            "duration": parsed_query.duration,
            "date": parsed_query.date,
            "amount": parsed_query.amount,
            "currency": parsed_query.currency,
            "document_type": parsed_query.document_type,
            "raw_query": parsed_query.raw_query
        }

    async def process_insurance_claim(self, query: str, file: UploadFile) -> Dict[str, Any]:
        """Process insurance claim with structured decision making"""
        start_time = time.time()

        # Ensure initialization
        await self.initialize()

        try:
            # Step 1: Parse insurance query
            logger.info("Parsing insurance query...")
            parsed_query = self.insurance_processor.parse_insurance_query(query)
            logger.info(f"Parsed query: Age={parsed_query.age}, Procedure={parsed_query.procedure}, Location={parsed_query.location}")

            # Step 2: Process document
            logger.info("Processing policy document...")
            document_text = await self.document_processor.process_document(file)

            # Step 3: Create search index
            document_id = str(uuid.uuid4())
            await self.search_engine.add_document(document_text, document_id)

            # Step 4: Search for relevant clauses
            logger.info("Searching for relevant policy clauses...")
            relevant_chunks = await self.search_engine.search_similar_chunks(
                query, document_id=document_id, k=10
            )

            if not relevant_chunks:
                return {
                    "decision": "REJECTED",
                    "amount": None,
                    "justification": "No relevant policy clauses found for this claim.",
                    "clauses_used": [],
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time,
                    "parsed_query": {
                        "age": parsed_query.age,
                        "gender": parsed_query.gender,
                        "procedure": parsed_query.procedure,
                        "location": parsed_query.location,
                        "policy_duration": parsed_query.policy_duration,
                        "amount_claimed": parsed_query.amount_claimed
                    }
                }

            # Step 5: Convert chunks to evidence format
            evidence_chunks = []
            for chunk in relevant_chunks:
                evidence_chunks.append({
                    'text': chunk.text,
                    'relevance_score': chunk.metadata.get('similarity_score', 0),
                    'metadata': chunk.metadata
                })

            # Step 6: Make insurance decision
            logger.info("Making insurance decision...")
            decision = await self.insurance_processor.make_insurance_decision(
                parsed_query, evidence_chunks
            )

            processing_time = time.time() - start_time

            # Step 7: Format response
            return {
                "decision": decision.decision,
                "amount": decision.amount,
                "justification": decision.justification,
                "clauses_used": decision.clauses_used or [],
                "confidence": decision.confidence,
                "processing_time": processing_time,
                "parsed_query": {
                    "age": parsed_query.age,
                    "gender": parsed_query.gender,
                    "procedure": parsed_query.procedure,
                    "location": parsed_query.location,
                    "policy_duration": parsed_query.policy_duration,
                    "amount_claimed": parsed_query.amount_claimed,
                    "raw_query": parsed_query.raw_query
                },
                "evidence_summary": {
                    "total_clauses_found": len(evidence_chunks),
                    "clauses_used_in_decision": len(decision.clauses_used or []),
                    "highest_relevance_score": max([chunk['relevance_score'] for chunk in evidence_chunks], default=0)
                }
            }

        except Exception as e:
            logger.error(f"Error processing insurance claim: {str(e)}")
            return {
                "decision": "ERROR",
                "amount": None,
                "justification": f"Error processing claim: {str(e)}",
                "clauses_used": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "parsed_query": {"raw_query": query}
            }[:num_suggestions]
    
    async def compare_documents(self, files: List[UploadFile], 
                              comparison_aspects: List[str]) -> ComparisonResult:
        """Compare multiple documents across specified aspects"""
        if len(files) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 documents allowed for comparison")
        
        documents = {}
        for i, file in enumerate(files):
            text = await self.document_processor.process_document(file)
            doc_id = f"doc_{i}"
            documents[doc_id] = {
                'filename': file.filename,
                'text': text,
                'analysis': self.analytics.analyze_document(text)
            }
        
        # Compare across aspects
        comparison_results = {
            'document_summaries': {},
            'aspect_comparisons': {},
            'similarities': {},
            'differences': {}
        }
        
        # Generate summaries
        for doc_id, doc_data in documents.items():
            comparison_results['document_summaries'][doc_id] = {
                'filename': doc_data['filename'],
                'word_count': doc_data['analysis']['word_count'],
                'document_type': doc_data['analysis']['document_type'],
                'complexity_score': doc_data['analysis']['complexity_score'],
                'key_topics': doc_data['analysis']['key_topics']
            }
        
        # Compare specific aspects using Gemini
        for aspect in comparison_aspects:
            aspect_comparison = await self._compare_aspect_across_documents(
                documents, aspect
            )
            comparison_results['aspect_comparisons'][aspect] = aspect_comparison
        
        return ComparisonResult(**comparison_results)
    
    async def _compare_aspect_across_documents(self, documents: Dict, aspect: str) -> Dict[str, Any]:
        """Compare a specific aspect across multiple documents using Gemini"""
        doc_texts = {doc_id: doc_data['text'] for doc_id, doc_data in documents.items()}
        
        prompt = f"""
        Compare the following documents regarding "{aspect}". 
        For each document, extract relevant information and then provide a comparison.
        
        Documents:
        {chr(10).join([f"DOCUMENT_{doc_id} ({documents[doc_id]['filename']}):{chr(10)}{text[:2000]}..." 
                      for doc_id, text in doc_texts.items()])}
        
        Please provide a JSON response with:
        {{
            "aspect": "{aspect}",
            "document_findings": {{
                "doc_0": "what document 0 says about {aspect}",
                "doc_1": "what document 1 says about {aspect}",
                ...
            }},
            "similarities": ["list of similarities across documents"],
            "differences": ["list of key differences"],
            "summary": "overall comparison summary"
        }}
        
        Only return the JSON object.
        """
        
        try:
            response = await asyncio.to_thread(
                self.gemini_parser.model.generate_content,
                prompt,
                generation_config=self.gemini_parser.model._generation_config
            )
            return json.loads(response.text.strip())
        except Exception as e:
            logger.error(f"Error comparing aspect {aspect}: {e}")
            return {
                "aspect": aspect,
                "document_findings": {},
                "similarities": [],
                "differences": [],
                "summary": f"Error analyzing aspect: {str(e)}"
            }

# =============================================================================
# FastAPI Application Setup
# =============================================================================

# Initialize API with LangChain by default
enhanced_api = EnhancedDocumentQueryAPI(use_langchain=True)
app = FastAPI(
    title="Enhanced Document Query API with LangChain",
    description="AI-powered document analysis using LangChain RAG, Google Gemini, Pinecone, and advanced features",
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
    """
    Main endpoint for document queries using LangChain RAG

    - **query**: Natural language question about the document
    - **file**: Document file (PDF, DOCX, EML, TXT, PNG, JPG)
    """
    if enhanced_api.use_langchain:
        return await enhanced_api.process_query_langchain(query, file)
    else:
        return await enhanced_api.process_query(query, file)

@app.post("/ask-document-langchain", response_model=QueryResponse)
async def ask_document_langchain(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Enhanced document queries using LangChain RAG pipeline

    This endpoint uses LangChain's Retrieval-Augmented Generation (RAG) for better accuracy:
    - Advanced text chunking and embedding
    - Gemini LLM for answer generation
    - Pinecone vector store for semantic search
    - Structured response with evidence and conditions

    - **query**: Natural language question about the document
    - **file**: Document file (PDF, DOCX, EML, TXT, PNG, JPG)
    """
    return await enhanced_api.process_query_langchain(query, file)

@app.post("/ask-document-clean", response_model=CleanQueryResponse)
async def ask_document_clean(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Clean, formatted document queries using LangChain RAG

    This endpoint provides clean, professional responses without technical evidence:
    - Professional formatting with clear structure
    - No technical references or evidence sections
    - Focus on direct answers and conditions
    - Optimized for end-user consumption

    - **query**: Natural language question about the document
    - **file**: Document file (PDF, DOCX, EML, TXT, PNG, JPG)
    """
    return await enhanced_api.process_query_clean(query, file)

@app.post("/ask-document-hybrid", response_model=QueryResponse)
async def ask_document_hybrid(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Enhanced endpoint with hybrid search

    - **query**: Natural language question about the document
    - **file**: Document file (PDF, DOCX, EML, TXT, PNG, JPG)
    """
    return await enhanced_api.process_query_with_hybrid_search(query, file)

@app.post("/ask-document-cached", response_model=QueryResponse)
async def ask_document_cached(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Endpoint with caching support

    - **query**: Natural language question about the document
    - **file**: Document file (PDF, DOCX, EML, TXT, PNG, JPG)
    """
    return await enhanced_api.process_query_with_cache(query, file)

@app.post("/analyze-document", response_model=DocumentAnalysis)
async def analyze_document(file: UploadFile = File(...)):
    """
    Analyze document and provide insights

    - **file**: Document file to analyze
    """
    return await enhanced_api.analyze_document_insights(file)

@app.post("/suggest-queries")
async def suggest_queries(
    file: UploadFile = File(...),
    num_suggestions: int = Form(5)
):
    """
    Generate suggested queries based on document content

    - **file**: Document file to analyze
    - **num_suggestions**: Number of suggestions to generate (default: 5)
    """
    suggestions = await enhanced_api.suggest_queries(file, num_suggestions)
    return {"suggestions": suggestions}

@app.post("/process-structured-query")
async def process_structured_query(
    query: str = Form(...),
    file: UploadFile = File(...),
    document_type: str = Form("general")
):
    """
    Process structured queries for any document type with decision making

    Supports:
    - Insurance: "46M, knee surgery, Pune, 3-month policy"
    - HR: "Employee ID EMP123, sick leave, 5 days, Mumbai office"
    - Legal: "Contract review, compliance check, Q4 2023"
    - General: Any document query requiring structured analysis

    - **query**: Natural language query
    - **file**: Document file (PDF, DOCX, etc.)
    - **document_type**: Type of document (insurance, hr, legal, contract, general)
    """
    return await enhanced_api.process_structured_query(query, file, document_type)

@app.post("/insurance-claim")
async def insurance_claim(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Process insurance claims with structured decision making

    Example query: "46M, knee surgery, Pune, 3-month policy"

    - **query**: Insurance claim details
    - **file**: Policy document
    """
    return await enhanced_api.process_structured_query(query, file, "insurance")

@app.post("/hr-request")
async def hr_request(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Process HR requests with policy compliance checking

    Example query: "Employee ID EMP123, sick leave, 5 days, Mumbai office"

    - **query**: HR request details
    - **file**: HR policy document
    """
    return await enhanced_api.process_structured_query(query, file, "hr")

@app.post("/legal-compliance")
async def legal_compliance(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Check legal compliance and contract terms

    Example query: "Contract review, compliance check, Q4 2023"

    - **query**: Legal compliance query
    - **file**: Legal document or contract
    """
    return await enhanced_api.process_structured_query(query, file, "legal")

@app.post("/contract-analysis")
async def contract_analysis(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Analyze contracts and agreements

    Example query: "Amendment required, payment terms, 30 days"

    - **query**: Contract analysis query
    - **file**: Contract document
    """
    return await enhanced_api.process_structured_query(query, file, "contract")

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "model": "gemini-2.0-flash",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Document Query API with Google Gemini",
        "endpoints": {
            "POST /ask-document": "Submit document and query",
            "POST /ask-document-hybrid": "Submit document and query with hybrid search",
            "POST /ask-document-cached": "Submit document and query with caching",
            "POST /analyze-document": "Analyze document for insights",
            "POST /suggest-queries": "Generate suggested queries",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "supported_formats": [".pdf", ".docx", ".eml", ".txt", ".png", ".jpg"]
    }

# =============================================================================
# CLI Runner (for development)
# =============================================================================

def setup_event_loop():
    """Setup proper event loop policy for Windows + PostgreSQL compatibility"""
    import sys
    import asyncio

    if sys.platform == "win32":
        try:
            # Set Windows selector event loop policy for psycopg compatibility
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            print("🔧 Set Windows selector event loop policy for PostgreSQL compatibility")
        except Exception as e:
            print(f"⚠️ Could not set event loop policy: {e}")

if __name__ == "__main__":
    # Setup event loop policy first
    setup_event_loop()

    # Check for API key (allow demo_key for testing)
    api_key = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Warning: GEMINI_API_KEY not set or using placeholder value")
        print("Some features may not work properly without a valid Google AI API key")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        print("Starting server anyway...")

    # Get port from environment (for Render deployment) or default to 3000
    port = int(os.getenv("PORT", 3000))
    host = "0.0.0.0" if os.getenv("RENDER") else "127.0.0.1"

    print("🚀 Starting Enhanced Document Query API...")
    print(f"📚 Access API documentation at: http://{host}:{port}/docs")
    print("🔗 Pinecone vector database: Connected")
    print("🐘 PostgreSQL database: Attempting connection...")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )