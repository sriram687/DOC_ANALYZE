import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "52428800"))  # 50MB default
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "10"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    MAX_CONCURRENT_BATCH: int = int(os.getenv("MAX_CONCURRENT_BATCH", "5"))
    CACHE_TTL_EMBEDDINGS: int = int(os.getenv("CACHE_TTL_EMBEDDINGS", "3600"))  # 1 hour
    CACHE_TTL_RESPONSES: int = int(os.getenv("CACHE_TTL_RESPONSES", "1800"))   # 30 minutes

    # Gemini-specific configurations
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT")
    USE_GEMINI_EMBEDDINGS: bool = os.getenv("USE_GEMINI_EMBEDDINGS", "true").lower() == "true"
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-pro")
    GEMINI_RATE_LIMIT_DELAY: float = float(os.getenv("GEMINI_RATE_LIMIT_DELAY", "1.0"))
    ENABLE_HYBRID_SEARCH: bool = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"

    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_DIMENSION: int = int(os.getenv("PINECONE_DIMENSION", "1024"))  # Match your index dimension
    PINECONE_METRIC: str = os.getenv("PINECONE_METRIC", "cosine")  # Match your index metric

    # PostgreSQL Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))

config = Config()