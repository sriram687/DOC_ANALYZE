import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_CHUNKS: int = 10
    CONFIDENCE_THRESHOLD: float = 0.7
    MAX_CONCURRENT_BATCH: int = 5
    CACHE_TTL_EMBEDDINGS: int = 3600  # 1 hour
    CACHE_TTL_RESPONSES: int = 1800   # 30 minutes
    
    # Gemini-specific configurations
    USE_GEMINI_EMBEDDINGS: bool = True
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"
    GEMINI_RATE_LIMIT_DELAY: float = 1.0  # Seconds between calls
    ENABLE_HYBRID_SEARCH: bool = True  # Use both Gemini and sentence transformer

config = Config()