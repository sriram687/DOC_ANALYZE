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
        self.gemini_dimension = 768 # Gemini embedding dimension

        # Pinecone configuration (use config values)
        self.pinecone_dimension = config.PINECONE_DIMENSION  # Your index dimension (512)
        self.pinecone_metric = config.PINECONE_METRIC  # Your index metric (cosine)

        # Initialize database manager (lazy initialization)
        self.db_manager = None
        self.db_initialized = False

        # Initialize Pinecone
        self.pc = None
        self.index = None
        self.pinecone_available = False

        if PINECONE_AVAILABLE and config.PINECONE_API_KEY and config.PINECONE_API_KEY != "your_pinecone_api_key_here":
            try:
                self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
                self._initialize_pinecone_index()
                if self.index:
                    self.pinecone_available = True
                    logger.info("✅ Pinecone initialized successfully")
                else:
                    logger.warning("⚠️ Pinecone client created but index connection failed")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Pinecone: {e}")
                logger.info("🔄 Continuing without Pinecone - using local embeddings only")
                self.pc = None
                self.index = None
                self.pinecone_available = False
        else:
            logger.warning("⚠️ Pinecone not configured - check API key in .env file")

        # Fallback to sentence transformers
        if ML_AVAILABLE:
            try:
                # Try different models in order of preference, with offline capability
                models_to_try = [
                    'all-MiniLM-L6-v2',  # Smaller, more likely to be cached
                    'paraphrase-MiniLM-L6-v2',  # Alternative small model
                ]

                self.sentence_transformer = None
                for model_name in models_to_try:
                    try:
                        # Try to load model with offline mode if available
                        self.sentence_transformer = SentenceTransformer(model_name)
                        self.sentence_transformer_dimension = self.sentence_transformer.get_sentence_embedding_dimension()
                        logger.info(f"Sentence transformer loaded: {model_name} (dim: {self.sentence_transformer_dimension})")
                        break
                    except Exception as model_error:
                        logger.warning(f"Could not load {model_name}: {model_error}")
                        continue

                if self.sentence_transformer is None:
                    logger.warning("No sentence transformer models could be loaded")

            except Exception as e:
                logger.warning(f"Could not initialize sentence transformers: {e}")
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
                    dimension=self.pinecone_dimension,  # Use config dimension (1024)
                    metric=self.pinecone_metric,  # Use config metric (cosine)
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

    def _pad_embedding_to_dimension(self, embedding: List[float], target_dimension: int) -> List[float]:
        """Pad or truncate embedding to match target dimension"""
        if len(embedding) == target_dimension:
            return embedding
        elif len(embedding) < target_dimension:
            # Pad with zeros
            padding = [0.0] * (target_dimension - len(embedding))
            return embedding + padding
        else:
            # Truncate to target dimension
            return embedding[:target_dimension]

    async def initialize(self):
        """Initialize the search engine and database"""
        if not hasattr(self, 'db_initialized'):
            self.db_initialized = False

        if not self.db_initialized:
            try:
                # Lazy initialize database manager
                if self.db_manager is None:
                    from database_manager import DatabaseManager
                    self.db_manager = DatabaseManager()

                await self.db_manager.initialize()
                self.db_initialized = True
                logger.info("✅ PostgreSQL database initialized successfully")
            except ImportError as e:
                logger.warning(f"❌ Database manager not available: {e}")
                logger.info("🔄 Continuing without database storage")
                self.db_initialized = False
                self.db_manager = None
            except Exception as e:
                logger.warning(f"❌ Database initialization failed: {e}")
                logger.info("🔄 Continuing without database storage")
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
                
                # Pad embedding to match Pinecone dimension
                embedding = result['embedding']
                padded_embedding = self._pad_embedding_to_dimension(embedding, self.pinecone_dimension)
                embeddings.append(padded_embedding)
                self.last_gemini_call = time.time()
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error getting Gemini embedding for text: {str(e)[:100]}... Error: {e}")

                # Fallback to sentence transformer if available
                if self.sentence_transformer is not None:
                    logger.info("Falling back to sentence transformer for this text")
                    try:
                        fallback_embedding = await asyncio.to_thread(
                            self.sentence_transformer.encode, [text]
                        )
                        # Pad or truncate to match Pinecone dimension
                        fallback_embedding = fallback_embedding[0].tolist() if hasattr(fallback_embedding[0], 'tolist') else fallback_embedding[0]
                        padded_embedding = self._pad_embedding_to_dimension(fallback_embedding, self.pinecone_dimension)
                        embeddings.append(padded_embedding)
                    except Exception as fallback_error:
                        logger.error(f"Fallback embedding also failed: {fallback_error}")
                        # Return zero embedding as last resort
                        zero_embedding = [0.0] * self.pinecone_dimension
                        embeddings.append(zero_embedding)
                else:
                    # Return zero embedding as last resort
                    zero_embedding = [0.0] * self.pinecone_dimension
                    embeddings.append(zero_embedding)

        return embeddings
    
    async def _get_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from sentence transformer"""
        if self.sentence_transformer is None:
            raise ValueError("Sentence transformer not available")

        embeddings = await asyncio.to_thread(self.sentence_transformer.encode, texts)
        embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

        # Pad all embeddings to match Pinecone dimension
        padded_embeddings = []
        for embedding in embeddings_list:
            padded_embedding = self._pad_embedding_to_dimension(embedding, self.pinecone_dimension)
            padded_embeddings.append(padded_embedding)

        return padded_embeddings
    
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
                try:
                    embeddings = await self._get_sentence_transformer_embeddings(texts)
                except Exception as fallback_error:
                    logger.error(f"Sentence transformer fallback also failed: {fallback_error}")
                    # Create dummy embeddings to ensure chunks are still stored
                    logger.info("Creating dummy embeddings to store chunks locally")
                    embeddings = [[0.0] * self.pinecone_dimension for _ in texts]
            else:
                # Create dummy embeddings to ensure chunks are still stored
                logger.info("No embeddings available - storing chunks with dummy embeddings for local search")
                embeddings = [[0.0] * self.pinecone_dimension for _ in texts]

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

            # Store embedding in chunk object and add to local storage
            chunk.embedding = embedding_list
            chunk.vector_id = vector_id  # Store the vector ID for reference
            self.chunks.append(chunk)

            # Ensure chunk has document_id for local search
            if not hasattr(chunk, 'document_id'):
                chunk.document_id = document_id

        # Upsert vectors to Pinecone (if available)
        if self.index and self.pinecone_available and vectors_to_upsert:
            try:
                logger.info(f"📤 Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
                self.index.upsert(vectors=vectors_to_upsert)
                logger.info(f"✅ Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone")

                # Verify the upsert worked
                stats = self.index.describe_index_stats()
                logger.info(f"📊 Pinecone index now has {stats.total_vector_count} total vectors")

            except Exception as e:
                logger.error(f"❌ Failed to upsert vectors to Pinecone: {e}")
                import traceback
                traceback.print_exc()
        elif vectors_to_upsert:
            logger.info(f"📝 Stored {len(vectors_to_upsert)} vectors locally (Pinecone not available)")
        else:
            logger.warning("⚠️ No vectors to upsert!")

        # Store chunks in PostgreSQL (optional) with improved error handling
        if hasattr(self, 'db_initialized') and self.db_initialized and self.db_manager:
            try:
                stored_ids = await self.db_manager.store_document_chunks(document_id, chunk_data_for_db)
                if stored_ids:
                    logger.info(f"✅ Stored {len(stored_ids)} chunks in PostgreSQL")
                else:
                    logger.warning("⚠️ No chunks were stored in PostgreSQL")
            except Exception as e:
                logger.warning(f"❌ Failed to store chunks in PostgreSQL: {e}")
                # Continue without database storage - Pinecone is still available
        else:
            logger.info("📝 Database not available, using Pinecone + local storage only")

        processing_time = time.time() - start_time
        logger.info(f"✅ Added {len(chunks)} chunks with embeddings in {processing_time:.2f}s")
        logger.info(f"📊 Total chunks in memory: {len(self.chunks)}")

        # Log storage status
        if self.pinecone_available:
            logger.info(f"📝 Database not available, storing only in Pinecone")
        else:
            logger.info(f"📝 Database not available, stored locally for text search")

        return chunks
    
    async def search_similar_chunks(self, query: str, k: int = None, document_id: str = None) -> List[DocumentChunk]:
        """Find most similar chunks to query using Pinecone vector search"""
        if k is None:
            k = config.TOP_K_CHUNKS

        if not self.index or not self.pinecone_available:
            logger.warning("⚠️ Pinecone not available - using local similarity search")
            logger.info(f"   Index available: {self.index is not None}")
            logger.info(f"   Pinecone available: {self.pinecone_available}")
            logger.info(f"   Total local chunks: {len(self.chunks)}")
            return await self._local_similarity_search(query, k, document_id)

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

        # Prepare query filter for Pinecone
        query_filter = None
        if document_id:
            query_filter = {"document_id": {"$eq": document_id}}
            logger.info(f"🔍 Filtering by document_id: {document_id}")

        # Search in Pinecone
        try:
            logger.info(f"🔍 Searching Pinecone with query vector (dim: {len(query_vector)})")

            search_results = self.index.query(
                vector=query_vector,
                top_k=k,
                include_metadata=True,
                filter=query_filter
            )

            logger.info(f"📊 Pinecone returned {len(search_results.matches)} matches")

            # Convert Pinecone results to DocumentChunk objects
            results = []
            # Use a lower threshold for cosine similarity (0.3 instead of 0.7)
            min_threshold = 0.3

            for i, match in enumerate(search_results.matches):
                logger.info(f"   Match {i+1}: Score={match.score:.3f}, ID={match.id}")

                if match.score > min_threshold:
                    # Create DocumentChunk from Pinecone metadata
                    chunk = DocumentChunk(
                        id=match.metadata.get("chunk_id", match.id),
                        text=match.metadata.get("text", ""),
                        metadata={
                            **match.metadata,
                            "similarity_score": float(match.score),
                            "pinecone_id": match.id
                        },
                        document_id=match.metadata.get("document_id", "")
                    )
                    results.append(chunk)
                    logger.info(f"   ✅ Added chunk: {chunk.text[:100]}...")
                else:
                    logger.info(f"   ❌ Score {match.score:.3f} below threshold {min_threshold}")

            search_time = time.time() - start_time
            logger.info(f"✅ Found {len(results)} relevant chunks in Pinecone in {search_time:.2f}s")

            # If no results with high threshold, try with very low threshold
            if len(results) == 0 and len(search_results.matches) > 0:
                logger.warning("🔄 No results with normal threshold, trying with lower threshold...")
                for match in search_results.matches[:3]:  # Take top 3 regardless of score
                    chunk = DocumentChunk(
                        id=match.metadata.get("chunk_id", match.id),
                        text=match.metadata.get("text", ""),
                        metadata={
                            **match.metadata,
                            "similarity_score": float(match.score),
                            "pinecone_id": match.id
                        },
                        document_id=match.metadata.get("document_id", "")
                    )
                    results.append(chunk)
                logger.info(f"📝 Added {len(results)} chunks with relaxed threshold")

            return results

        except Exception as e:
            logger.error(f"❌ Error searching in Pinecone: {e}")
            import traceback
            traceback.print_exc()
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
    
    async def hybrid_search(self, query: str, k: int = None, document_id: str = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both Gemini and sentence transformer embeddings
        Then combine and rank results
        """
        if k is None:
            k = config.TOP_K_CHUNKS

        results = []

        # Search with Gemini embeddings
        try:
            gemini_results = await self.search_similar_chunks(query, k, document_id)
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

    async def _local_similarity_search(self, query: str, k: int, document_id: str = None) -> List[DocumentChunk]:
        """Fallback local similarity search when Pinecone is not available"""
        try:
            logger.info(f"🔍 Starting local search for query: '{query}' (total chunks: {len(self.chunks)})")

            # Simple text-based similarity using keyword matching
            query_words = set(query.lower().split())

            # Filter chunks by document_id if specified
            chunks_to_search = self.chunks
            if document_id:
                chunks_to_search = [chunk for chunk in self.chunks if hasattr(chunk, 'document_id') and chunk.document_id == document_id]
                logger.info(f"🔍 Filtered to {len(chunks_to_search)} chunks for document_id: {document_id}")

            if not chunks_to_search:
                logger.warning(f"❌ No chunks found for search (document_id: {document_id})")
                # If no chunks for specific document, try all chunks
                if document_id:
                    logger.info("🔄 Trying search across all chunks...")
                    chunks_to_search = self.chunks

                if not chunks_to_search:
                    logger.warning("❌ No chunks available at all")
                    logger.info("💡 This might mean:")
                    logger.info("   1. Document wasn't processed properly")
                    logger.info("   2. Chunks weren't stored")
                    logger.info("   3. Pinecone connection failed during storage")
                    return []

            # Calculate simple similarity scores
            scored_chunks = []
            for chunk in chunks_to_search:
                chunk_words = set(chunk.text.lower().split())

                # Multiple similarity metrics
                # 1. Jaccard similarity
                intersection = len(query_words.intersection(chunk_words))
                union = len(query_words.union(chunk_words))
                jaccard_sim = intersection / union if union > 0 else 0

                # 2. Simple word overlap score
                overlap_score = intersection / len(query_words) if len(query_words) > 0 else 0

                # 3. Check for exact phrase matches
                phrase_bonus = 0
                if len(query.strip()) > 3 and query.lower() in chunk.text.lower():
                    phrase_bonus = 0.5

                # Combined similarity score
                similarity = (jaccard_sim * 0.4) + (overlap_score * 0.4) + phrase_bonus

                if similarity > 0.01:  # Lower threshold to include more results
                    scored_chunks.append((chunk, similarity))

            # Sort by similarity score
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            # Return top k chunks
            result_chunks = [chunk for chunk, _ in scored_chunks[:k]]

            logger.info(f"✅ Local search found {len(result_chunks)} relevant chunks out of {len(scored_chunks)} candidates")

            # Log some debug info about the best matches
            if scored_chunks:
                best_score = scored_chunks[0][1]
                logger.info(f"📊 Best match score: {best_score:.3f}")

            return result_chunks

        except Exception as e:
            logger.error(f"❌ Error in local similarity search: {e}")
            import traceback
            traceback.print_exc()
            return []