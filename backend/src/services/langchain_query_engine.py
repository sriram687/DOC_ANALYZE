"""
LangChain-based Query Engine for Enhanced Document Query API
Provides robust document Q&A using LangChain's RAG (Retrieval-Augmented Generation) pipeline
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import uuid

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Pinecone
from pinecone import Pinecone

# Local imports
from config import config
from models import QueryResponse, DocumentChunk

logger = logging.getLogger(__name__)

class AsyncLoggingHandler(AsyncCallbackHandler):
    """Custom callback handler for logging LangChain operations"""
    
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        logger.info(f"🔗 Starting chain: {serialized.get('name', 'Unknown')}")
    
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        logger.info("✅ Chain completed successfully")
    
    async def on_chain_error(self, error: Exception, **kwargs) -> None:
        logger.error(f"❌ Chain error: {error}")

class LangChainQueryEngine:
    """
    Advanced query engine using LangChain for document Q&A
    Integrates Gemini LLM, Gemini embeddings, and Pinecone vector store
    """
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = None
        self.qa_chain = None
        self.callback_handler = AsyncLoggingHandler()
        self._initialized = False
        
        # Configuration
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.top_k = config.TOP_K_CHUNKS
        
    async def initialize(self):
        """Initialize LangChain components"""
        if self._initialized:
            return
            
        logger.info("🚀 Initializing LangChain Query Engine...")
        
        try:
            # Initialize Gemini LLM
            self.llm = GoogleGenerativeAI(
                model=config.GEMINI_CHAT_MODEL,
                google_api_key=config.GEMINI_API_KEY,
                temperature=0.3,
                max_output_tokens=1500
            )
            logger.info("✅ Gemini LLM initialized")
            
            # Initialize Gemini embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=config.GEMINI_EMBEDDING_MODEL,
                google_api_key=config.GEMINI_API_KEY
            )
            logger.info("✅ Gemini embeddings initialized")
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            logger.info("✅ Text splitter initialized")
            
            # Initialize Pinecone vector store
            await self._initialize_vector_store()
            
            # Create QA chain
            self._create_qa_chain()
            
            self._initialized = True
            logger.info("🎉 LangChain Query Engine initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LangChain Query Engine: {e}")
            raise
    
    async def _initialize_vector_store(self):
        """Initialize Pinecone vector store"""
        try:
            # Initialize Pinecone
            pc = Pinecone(api_key=config.PINECONE_API_KEY)
            
            # Check if index exists
            if not pc.has_index(config.PINECONE_INDEX_NAME):
                logger.warning(f"⚠️ Pinecone index '{config.PINECONE_INDEX_NAME}' does not exist")
                logger.info("Creating new index...")
                
                from pinecone import ServerlessSpec
                pc.create_index(
                    name=config.PINECONE_INDEX_NAME,
                    dimension=768,  # Gemini embedding dimension
                    metric=config.PINECONE_METRIC,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=config.PINECONE_ENVIRONMENT or "us-east-1"
                    )
                )
                
                # Wait for index to be ready
                import time
                time.sleep(30)
            
            # Initialize vector store
            self.vector_store = PineconeVectorStore(
                index_name=config.PINECONE_INDEX_NAME,
                embedding=self.embeddings,
                pinecone_api_key=config.PINECONE_API_KEY
            )
            logger.info("✅ Pinecone vector store initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            raise
    
    def _create_qa_chain(self):
        """Create the question-answering chain"""
        
        # Custom prompt template for insurance/financial documents
        prompt_template = """
        You are an expert insurance advisor analyzing policy documents.
        Use the following context to provide a clear, professional answer.

        Context:
        {context}

        Question: {question}

        Instructions:
        1. Provide a direct, clear answer in a professional tone
        2. Structure your response with clear headings using markdown (##, ###)
        3. Use bullet points or numbered lists for conditions and requirements
        4. Be specific about coverage amounts, limits, and exclusions
        5. If something is not covered, explain why clearly
        6. If you cannot find specific information, state this clearly
        7. Keep the response concise but comprehensive
        8. Do not include references to "context" or "document" in your answer

        Format your answer professionally as if speaking directly to a policyholder.

        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        # Create the chain using LCEL (LangChain Expression Language)
        self.qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("✅ QA chain created successfully")
    
    async def add_document(self, text: str, document_id: str, metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """Add document to vector store"""
        await self.initialize()
        
        logger.info(f"📄 Processing document {document_id}...")
        start_time = time.time()
        
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"📝 Created {len(chunks)} chunks")
            
            # Create Document objects with metadata
            documents = []
            document_chunks = []
            
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_id": f"{document_id}_{i}",
                    **(metadata or {})
                }
                
                # LangChain Document
                doc = Document(
                    page_content=chunk,
                    metadata=doc_metadata
                )
                documents.append(doc)
                
                # Our DocumentChunk for compatibility
                doc_chunk = DocumentChunk(
                    id=doc_metadata["chunk_id"],
                    text=chunk,
                    metadata=doc_metadata,
                    document_id=document_id
                )
                document_chunks.append(doc_chunk)
            
            # Add to vector store
            await asyncio.to_thread(
                self.vector_store.add_documents,
                documents
            )
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Document processed in {processing_time:.2f}s")
            
            return document_chunks
            
        except Exception as e:
            logger.error(f"❌ Error adding document: {e}")
            raise
    
    async def query_document(self, query: str, document_id: Optional[str] = None) -> QueryResponse:
        """Query the document using LangChain RAG pipeline"""
        await self.initialize()
        
        logger.info(f"🔍 Processing query: {query}")
        start_time = time.time()
        
        try:
            # If document_id is specified, filter retrieval
            if document_id:
                # Update retriever with filter
                retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": self.top_k,
                        "filter": {"document_id": document_id}
                    }
                )
                
                # Recreate chain with filtered retriever
                prompt_template = """
                You are an expert assistant for analyzing insurance and financial documents. 
                Use the following pieces of context to answer the question at the end.
                
                Context:
                {context}
                
                Question: {question}
                
                Instructions:
                1. Provide a clear, direct answer based on the context
                2. If the answer involves conditions or requirements, list them clearly
                3. Include relevant evidence from the document
                4. If you cannot find the answer in the context, say so clearly
                5. Be specific about coverage amounts, limits, and exclusions
                6. For insurance queries, focus on eligibility, coverage, and claim procedures
                
                Answer:
                """
                
                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | PROMPT
                    | self.llm
                    | StrOutputParser()
                )
            else:
                chain = self.qa_chain
            
            # Execute the chain
            answer = await asyncio.to_thread(chain.invoke, query)

            # Clean up the answer
            answer = self._clean_answer(answer)

            # Extract conditions from answer using improved logic
            conditions = self._extract_conditions(answer)
            
            processing_time = time.time() - start_time

            # Calculate confidence based on answer quality
            confidence = self._calculate_confidence(answer)

            response = QueryResponse(
                query=query,
                answer=answer,
                conditions=conditions,
                evidence=[],  # Remove evidence for cleaner response
                confidence=confidence,
                processing_time=processing_time
            )
            
            logger.info(f"✅ Query processed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            
            # Return error response
            return QueryResponse(
                query=query,
                answer=f"I encountered an error while processing your query: {str(e)}",
                conditions=[],
                evidence=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    async def search_similar_chunks(self, query: str, k: int = None, document_id: str = None) -> List[DocumentChunk]:
        """Search for similar chunks (compatibility method)"""
        await self.initialize()
        
        k = k or self.top_k
        
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": k,
                    **({"filter": {"document_id": document_id}} if document_id else {})
                }
            )
            
            docs = await asyncio.to_thread(retriever.get_relevant_documents, query)
            
            chunks = []
            for doc in docs:
                chunk = DocumentChunk(
                    id=doc.metadata.get("chunk_id", str(uuid.uuid4())),
                    text=doc.page_content,
                    metadata=doc.metadata,
                    document_id=doc.metadata.get("document_id", "")
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error searching chunks: {e}")
            return []

    def _clean_answer(self, answer: str) -> str:
        """Clean and format the answer for better presentation"""
        if not answer:
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."

        # Remove any references to "context" or "document"
        answer = answer.replace("Based on the context provided", "Based on the policy information")
        answer = answer.replace("the context", "the policy")
        answer = answer.replace("the document", "the policy")
        answer = answer.replace("According to the document", "According to the policy")
        answer = answer.replace("The document states", "The policy states")

        # Clean up any markdown formatting issues
        answer = answer.replace("**", "")  # Remove bold formatting for now

        # Ensure proper spacing after periods
        import re
        answer = re.sub(r'\.([A-Z])', r'. \1', answer)

        return answer.strip()

    def _extract_conditions(self, answer: str) -> List[str]:
        """Extract conditions and requirements from the answer"""
        conditions = []

        if not answer:
            return conditions

        # Look for common condition indicators
        condition_indicators = [
            "must", "required", "condition", "eligibility", "prerequisite",
            "provided that", "subject to", "only if", "except", "unless"
        ]

        sentences = answer.split('. ')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in condition_indicators):
                # Clean up the sentence
                if sentence and not sentence.endswith('.'):
                    sentence += '.'
                conditions.append(sentence)

        # Remove duplicates while preserving order
        seen = set()
        unique_conditions = []
        for condition in conditions:
            if condition not in seen:
                seen.add(condition)
                unique_conditions.append(condition)

        return unique_conditions[:5]  # Limit to top 5 conditions

    def _calculate_confidence(self, answer: str) -> float:
        """Calculate confidence score based on answer quality"""
        if not answer:
            return 0.0

        # Base confidence
        confidence = 0.7

        # Increase confidence for detailed answers
        if len(answer) > 100:
            confidence += 0.1

        # Increase confidence for structured answers
        if any(marker in answer for marker in ["##", "###", "•", "-", "1.", "2."]):
            confidence += 0.1

        # Decrease confidence for uncertain language
        uncertainty_phrases = [
            "cannot find", "not mentioned", "unclear", "uncertain",
            "may be", "might be", "possibly", "perhaps"
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.3

        # Increase confidence for specific information
        specific_indicators = [
            "rs.", "rupees", "percent", "%", "days", "months", "years",
            "covered", "excluded", "eligible", "claim"
        ]
        if any(indicator in answer.lower() for indicator in specific_indicators):
            confidence += 0.1

        return min(max(confidence, 0.0), 1.0)  # Ensure between 0 and 1
