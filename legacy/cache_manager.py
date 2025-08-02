import json
import hashlib
import logging
from typing import Optional, List, Dict
from config import config

# Enhanced features
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis-based caching for embeddings and responses"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or config.REDIS_URL
        self.enabled = False
        self.redis_client = None
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                self.redis_client.ping()
                self.enabled = True
                logger.info("Redis cache enabled")
            except Exception as e:
                logger.warning(f"Redis not available: {e}, caching disabled")
        else:
            logger.warning("Redis package not installed, caching disabled")
    
    def _get_cache_key(self, prefix: str, content: str) -> str:
        """Generate cache key from content hash"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{prefix}:{content_hash}"
    
    async def get_document_embeddings(self, document_text: str) -> Optional[List[Dict]]:
        """Get cached document embeddings"""
        if not self.enabled:
            return None
        
        key = self._get_cache_key("embeddings", document_text)
        try:
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    async def set_document_embeddings(self, document_text: str, embeddings: List[Dict], 
                                    ttl: int = None):
        """Cache document embeddings"""
        if not self.enabled:
            return
        
        ttl = ttl or config.CACHE_TTL_EMBEDDINGS
        key = self._get_cache_key("embeddings", document_text)
        try:
            self.redis_client.setex(key, ttl, json.dumps(embeddings))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def get_query_response(self, query: str, document_hash: str) -> Optional[Dict]:
        """Get cached query response"""
        if not self.enabled:
            return None
        
        key = self._get_cache_key("response", f"{query}:{document_hash}")
        try:
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    async def set_query_response(self, query: str, document_hash: str, response: Dict, 
                               ttl: int = None):
        """Cache query response"""
        if not self.enabled:
            return
        
        ttl = ttl or config.CACHE_TTL_RESPONSES
        key = self._get_cache_key("response", f"{query}:{document_hash}")
        try:
            self.redis_client.setex(key, ttl, json.dumps(response))
        except Exception as e:
            logger.error(f"Cache set error: {e}")