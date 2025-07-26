import re
import time
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional

import google.generativeai as genai
import numpy as np
import faiss

from models import DocumentChunk
from config import config

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class GeminiEmbeddingSearchEngine:
    """Enhanced search engine using Gemini embeddings with fallback to sentence-transformers"""
    
    def __init__(self, use_gemini_primary: bool = True):
        """
        Initialize with Gemini as primary embedding model
        
        Args:
            use_gemini_primary: Whether to use Gemini as primary embedding model
        """
        self.use_gemini_primary = use_gemini_primary
        
        # Gemini embedding configuration
        self.gemini_model_name = "models/text-embedding-004"  # Latest Gemini embedding model
        self.gemini_dimension = 768  # Gemini embedding dimension
        
        # Fallback to sentence transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                self.sentence_transformer_dimension = 384
                logger.info("Sentence transformer loaded as fallback")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.sentence_transformer = None
        else:
            self.sentence_transformer = None
        
        # Initialize FAISS index based on primary model
        if self.use_gemini_primary:
            self.dimension = self.gemini_dimension
            logger.info("Using Gemini embeddings as primary")
        else:
            self.dimension = self.sentence_transformer_dimension
            logger.info("Using Sentence Transformer embeddings as primary")
            
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.chunks: List[DocumentChunk] = []
        
        # Rate limiting for Gemini API
        self.last_gemini_call = 0
        self.min_call_interval = 1.0  # Minimum 1 second between calls
        
    async def _get_gemini_embeddings(self, texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
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
                
                # Get embedding from Gemini
                result = await asyncio.to_thread(
                    genai.embed_content,
                    model=self.gemini_model_name,
                    content=text,
                    task_type=task_type,
                    title=f"Document chunk" if task_type == "retrieval_document" else "Query"
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
                        padded = np.zeros(self.gemini_dimension)
                        padded[:len(fallback_embedding)] = fallback_embedding
                        embeddings.append(padded.tolist())
                    else:
                        # Truncate
                        embeddings.append(fallback_embedding[:self.gemini_dimension].tolist())
                else:
                    # Return zero embedding as last resort
                    embeddings.append([0.0] * self.gemini_dimension)
        
        return np.array(embeddings)
    
    async def _get_sentence_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from sentence transformer"""
        if self.sentence_transformer is None:
            raise ValueError("Sentence transformer not available")
        
        embeddings = await asyncio.to_thread(self.sentence_transformer.encode, texts)
        return embeddings
    
    async def add_document(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Process document and add to search index with Gemini embeddings"""
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
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
            self.chunks.append(chunk)
        
        processing_time = time.time() - start_time
        logger.info(f"Added {len(chunks)} chunks with embeddings in {processing_time:.2f}s")
        
        return chunks
    
    async def search_similar_chunks(self, query: str, k: int = None) -> List[DocumentChunk]:
        """Find most similar chunks to query using Gemini embeddings"""
        if k is None:
            k = config.TOP_K_CHUNKS
        
        if not self.chunks:
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
        
        # Normalize query embedding
        query_embedding = query_embeddings[0:1]
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return matching chunks with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and score > 0.3:  # Minimum similarity threshold
                chunk = self.chunks[idx]
                # Add similarity score to metadata
                chunk.metadata['similarity_score'] = float(score)
                results.append(chunk)
        
        search_time = time.time() - start_time
        logger.info(f"Found {len(results)} relevant chunks in {search_time:.2f}s")
        
        return results
    
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
        # More sophisticated sentence splitting
        # Handle common abbreviations and edge cases
        abbreviations = r'(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Inc|Ltd|Corp|Co)'
        
        # Split on sentence endings, but not after abbreviations
        sentences = re.split(r'(?<!' + abbreviations + r')(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+', text)
        
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