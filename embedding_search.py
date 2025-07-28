import re
import time
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional

import google.generativeai as genai

from models import DocumentChunk
from config import config
from database_manager import DatabaseManager

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Create dummy classes for when ML libraries are not available
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, *args, **kwargs):
            return []

logger = logging.getLogger(__name__)

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not available. Install with: pip install pinecone-client")

class GeminiEmbeddingSearchEngine:
    """Enhanced search engine using Gemini embeddings with Pinecone vector database"""

    def __init__(self, use_gemini_primary: bool = True):
        """
        Initialize with Gemini as primary embedding model and Pinecone for vector storage

        Args:
            use_gemini_primary: Whether to use Gemini as primary embedding model
        """
        self.use_gemini_primary = use_gemini_primary

        # Gemini embedding configuration
        self.gemini_model_name = "models/text-embedding-004"  # Latest Gemini embedding model
        self.gemini_dimension = 768  # Gemini embedding dimension

        # Initialize database manager (lazy initialization)
        self.db_manager = None
        self.db_initialized = False

        # Initialize Pinecone
        self.pc = None
        self.index = None
        if PINECONE_AVAILABLE and config.PINECONE_API_KEY:
            try:
                self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
                self._initialize_pinecone_index()
                logger.info("Pinecone initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {e}")
                self.pc = None
        else:
            logger.warning("Pinecone not available or API key not set")

        # Fallback to sentence transformers
        if ML_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                self.sentence_transformer_dimension = 384
                logger.info("Sentence transformer loaded as fallback")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.sentence_transformer = None
        else:
            self.sentence_transformer = None

        # Set dimension based on primary model
        if self.use_gemini_primary:
            self.dimension = self.gemini_dimension
            logger.info("Using Gemini embeddings as primary")
        else:
            self.dimension = self.sentence_transformer_dimension if ML_AVAILABLE else 768
            logger.info("Using Sentence Transformer embeddings as primary")

        self.chunks: List[DocumentChunk] = []

        # Rate limiting for Gemini API
        self.last_gemini_call = 0
        self.min_call_interval = 1.0  # Minimum 1 second between calls

    def _initialize_pinecone_index(self):
        """Initialize or connect to Pinecone index"""
        try:
            index_name = config.PINECONE_INDEX_NAME

            # Check if index exists
            if not self.pc.has_index(index_name):
                logger.info(f"Creating Pinecone index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=self.gemini_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=config.PINECONE_ENVIRONMENT
                    )
                )
                # Wait for index to be ready
                import time
                time.sleep(10)

            # Connect to the index
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            self.index = None

    async def initialize(self):
        """Initialize the search engine and database"""
        if not self.db_initialized:
            try:
                # Lazy initialize database manager
                if self.db_manager is None:
                    self.db_manager = DatabaseManager()

                await self.db_manager.initialize()
                self.db_initialized = True
                logger.info("✅ PostgreSQL database initialized successfully")
            except Exception as e:
                logger.warning(f"❌ Database initialization failed: {e}")
                logger.warning("Continuing without database storage")
                self.db_initialized = False
                self.db_manager = None
        
    async def _get_gemini_embeddings(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """
        Get embeddings from Gemini with rate limiting and error handling
        
        Args:
            texts: List of texts to embed
            task_type: Task type for Gemini embedding
        
        Returns:
            numpy array of embeddings
        """
        embeddings = []
        
        for text in texts:
            try:
                # Rate limiting
                current_time = time.time()
                time_since_last_call = current_time - self.last_gemini_call
                if time_since_last_call < self.min_call_interval:
                    await asyncio.sleep(self.min_call_interval - time_since_last_call)
                
                # Get embedding from Gemini (title only for retrieval_document)
                if task_type == "retrieval_document":
                    result = await asyncio.to_thread(
                        genai.embed_content,
                        model=self.gemini_model_name,
                        content=text,
                        task_type=task_type,
                        title="Document chunk"
                    )
                else:
                    # For retrieval_query, don't include title
                    result = await asyncio.to_thread(
                        genai.embed_content,
                        model=self.gemini_model_name,
                        content=text,
                        task_type=task_type
                    )
                
                embeddings.append(result['embedding'])
                self.last_gemini_call = time.time()
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error getting Gemini embedding for text: {str(e)[:100]}... Error: {e}")
                
                # Fallback to sentence transformer if available
                if self.sentence_transformer is not None:
                    logger.info("Falling back to sentence transformer for this text")
                    fallback_embedding = await asyncio.to_thread(
                        self.sentence_transformer.encode, [text]
                    )
                    # Pad or truncate to match Gemini dimension
                    fallback_embedding = fallback_embedding[0]
                    if len(fallback_embedding) < self.gemini_dimension:
                        # Pad with zeros
                        if ML_AVAILABLE:
                            padded = np.zeros(self.gemini_dimension)
                            padded[:len(fallback_embedding)] = fallback_embedding
                            embeddings.append(padded.tolist())
                        else:
                            padded = [0.0] * self.gemini_dimension
                            padded[:len(fallback_embedding)] = fallback_embedding
                            embeddings.append(padded)
                    else:
                        # Truncate
                        embeddings.append(fallback_embedding[:self.gemini_dimension].tolist())
                else:
                    # Return zero embedding as last resort
                    embeddings.append([0.0] * self.gemini_dimension)

        return embeddings if not ML_AVAILABLE else np.array(embeddings)
    
    async def _get_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from sentence transformer"""
        if self.sentence_transformer is None:
            raise ValueError("Sentence transformer not available")

        embeddings = await asyncio.to_thread(self.sentence_transformer.encode, texts)
        return embeddings.tolist() if ML_AVAILABLE else embeddings
    
    async def add_document(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Process document and add to Pinecone index and PostgreSQL"""
        start_time = time.time()

        # Chunk the document
        chunks = self.chunk_document(text, document_id)
        logger.info(f"Created {len(chunks)} chunks for document {document_id}")

        # Get texts for embedding
        texts = [chunk.text for chunk in chunks]

        # Get embeddings based on primary model
        try:
            if self.use_gemini_primary:
                logger.info("Generating Gemini embeddings...")
                embeddings = await self._get_gemini_embeddings(texts, "retrieval_document")
            else:
                logger.info("Generating Sentence Transformer embeddings...")
                embeddings = await self._get_sentence_transformer_embeddings(texts)

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Fallback to alternative method
            if self.use_gemini_primary and self.sentence_transformer is not None:
                logger.info("Falling back to sentence transformer")
                embeddings = await self._get_sentence_transformer_embeddings(texts)
            else:
                raise e

        # Prepare vectors for Pinecone
        vectors_to_upsert = []
        chunk_data_for_db = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate unique ID for Pinecone
            vector_id = f"{document_id}_{i}_{uuid.uuid4().hex[:8]}"

            # Prepare vector for Pinecone
            if isinstance(embedding, list):
                embedding_list = embedding
            else:
                embedding_list = embedding.tolist()

            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding_list,
                "metadata": {
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk.text[:1000],  # Limit metadata text size
                    "chunk_id": chunk.id
                }
            })

            # Prepare chunk data for database
            chunk_data_for_db.append({
                "content": chunk.text,
                "metadata": {
                    "chunk_index": i,
                    "start_char": getattr(chunk, 'start_char', 0),
                    "end_char": getattr(chunk, 'end_char', len(chunk.text))
                },
                "embedding_id": vector_id
            })

            # Store embedding in chunk object
            chunk.embedding = embedding_list
            self.chunks.append(chunk)

        # Upsert vectors to Pinecone
        if self.index and vectors_to_upsert:
            try:
                self.index.upsert(vectors=vectors_to_upsert)
                logger.info(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone")
            except Exception as e:
                logger.error(f"Failed to upsert vectors to Pinecone: {e}")

        # Store chunks in PostgreSQL (optional)
        if self.db_initialized and self.db_manager:
            try:
                await self.db_manager.store_document_chunks(document_id, chunk_data_for_db)
                logger.info(f"✅ Stored {len(chunk_data_for_db)} chunks in PostgreSQL")
            except Exception as e:
                logger.warning(f"❌ Failed to store chunks in PostgreSQL: {e}")
        else:
            logger.info("📝 Database not available, storing only in Pinecone")

        processing_time = time.time() - start_time
        logger.info(f"Added {len(chunks)} chunks with embeddings in {processing_time:.2f}s")

        return chunks
    
    async def search_similar_chunks(self, query: str, k: int = None, document_id: str = None) -> List[DocumentChunk]:
        """Find most similar chunks to query using Pinecone vector search"""
        if k is None:
            k = config.TOP_K_CHUNKS

        if not self.index:
            logger.warning("Pinecone index not available")
            return []

        start_time = time.time()

        # Generate query embedding
        try:
            if self.use_gemini_primary:
                query_embeddings = await self._get_gemini_embeddings([query], "retrieval_query")
            else:
                query_embeddings = await self._get_sentence_transformer_embeddings([query])
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            # Try fallback
            if self.use_gemini_primary and self.sentence_transformer is not None:
                query_embeddings = await self._get_sentence_transformer_embeddings([query])
            else:
                return []

        # Get query embedding vector
        if isinstance(query_embeddings, list):
            query_vector = query_embeddings[0] if len(query_embeddings) > 0 else query_embeddings
        else:
            query_vector = query_embeddings[0].tolist()

        # Prepare query filter
        query_filter = {}
        if document_id:
            query_filter["document_id"] = document_id

        # Search in Pinecone
        try:
            search_results = self.index.query(
                vector=query_vector,
                top_k=k,
                include_metadata=True,
                filter=query_filter if query_filter else None
            )

            # Convert Pinecone results to DocumentChunk objects
            results = []
            for match in search_results.matches:
                if match.score > config.CONFIDENCE_THRESHOLD:
                    # Create DocumentChunk from Pinecone metadata
                    chunk = DocumentChunk(
                        id=match.metadata.get("chunk_id", match.id),
                        document_id=match.metadata.get("document_id", ""),
                        text=match.metadata.get("text", ""),
                        metadata={
                            **match.metadata,
                            "similarity_score": float(match.score),
                            "pinecone_id": match.id
                        }
                    )
                    results.append(chunk)

            search_time = time.time() - start_time
            logger.info(f"Found {len(results)} relevant chunks in Pinecone in {search_time:.2f}s")

            return results

        except Exception as e:
            logger.error(f"Error searching in Pinecone: {e}")
            return []
    
    def chunk_document(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Split document into semantic chunks with enhanced logic"""
        # Enhanced chunking strategy
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk_size = len(current_chunk) + len(sentence) + 1
            
            if potential_chunk_size > config.CHUNK_SIZE and current_chunk:
                # Create chunk
                chunk_id = f"{document_id}_{len(chunks)}"
                chunk = DocumentChunk(
                    id=chunk_id,
                    text=current_chunk.strip(),
                    metadata={
                        "document_id": document_id,
                        "chunk_index": len(chunks),
                        "sentence_count": len(current_sentences),
                        "word_count": len(current_chunk.split()),
                        "start_sentence": current_sentences[0][:50] + "..." if current_sentences else "",
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_size = min(2, len(current_sentences))
                if overlap_size > 0:
                    overlap_sentences = current_sentences[-overlap_size:]
                    current_chunk = " ".join(overlap_sentences) + " " + sentence
                    current_sentences = overlap_sentences + [sentence]
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunk_id = f"{document_id}_{len(chunks)}"
            chunk = DocumentChunk(
                id=chunk_id,
                text=current_chunk.strip(),
                metadata={
                    "document_id": document_id,
                    "chunk_index": len(chunks),
                    "sentence_count": len(current_sentences),
                    "word_count": len(current_chunk.split()),
                    "start_sentence": current_sentences[0][:50] + "..." if current_sentences else "",
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Enhanced sentence splitting"""
        # Simple but effective sentence splitting
        # Split on sentence endings followed by whitespace and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                clean_sentences.append(sentence)

        return clean_sentences
    
    async def hybrid_search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both Gemini and sentence transformer embeddings
        Then combine and rank results
        """
        if k is None:
            k = config.TOP_K_CHUNKS
            
        results = []
        
        # Search with Gemini embeddings
        try:
            gemini_results = await self.search_similar_chunks(query, k)
            for chunk in gemini_results:
                results.append({
                    'chunk': chunk,
                    'gemini_score': chunk.metadata.get('similarity_score', 0.0),
                    'sentence_transformer_score': 0.0,
                    'source': 'gemini'
                })
        except Exception as e:
            logger.error(f"Gemini search failed: {e}")
        
        # If we have sentence transformer available, also search with it
        if self.sentence_transformer is not None:
            try:
                # Temporarily switch to sentence transformer mode
                original_primary = self.use_gemini_primary
                self.use_gemini_primary = False
                
                st_results = await self.search_similar_chunks(query, k)
                
                # Restore original setting
                self.use_gemini_primary = original_primary
                
                # Add sentence transformer scores
                for chunk in st_results:
                    # Find if this chunk already exists from Gemini search
                    existing = next((r for r in results if r['chunk'].id == chunk.id), None)
                    if existing:
                        existing['sentence_transformer_score'] = chunk.metadata.get('similarity_score', 0.0)
                    else:
                        results.append({
                            'chunk': chunk,
                            'gemini_score': 0.0,
                            'sentence_transformer_score': chunk.metadata.get('similarity_score', 0.0),
                            'source': 'sentence_transformer'
                        })
                        
            except Exception as e:
                logger.error(f"Sentence transformer search failed: {e}")
        
        # Combine scores and rank
        for result in results:
            # Weighted combination of scores
            combined_score = (
                0.7 * result['gemini_score'] + 
                0.3 * result['sentence_transformer_score']
            )
            result['combined_score'] = combined_score
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top k results
        return results[:k]