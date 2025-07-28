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
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "10"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    MAX_CONCURRENT_BATCH: int = int(os.getenv("MAX_CONCURRENT_BATCH", "5"))
    CACHE_TTL_EMBEDDINGS: int = int(os.getenv("CACHE_TTL_EMBEDDINGS", "3600"))  # 1 hour
    CACHE_TTL_RESPONSES: int = int(os.getenv("CACHE_TTL_RESPONSES", "1800"))   # 30 minutes

    # Gemini-specific configurations
    USE_GEMINI_EMBEDDINGS: bool = True
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"
    GEMINI_RATE_LIMIT_DELAY: float = 1.0  # Seconds between calls
    ENABLE_HYBRID_SEARCH: bool = True  # Use both Gemini and sentence transformer

    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "pcsk_6VE3cD_Ca6NPMnRZtB9ZWQAD7ErCyjUxyvvspdDv967ST8QBFQS8w3ZvQv8HEV2Sdjggqg")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "developer-quickstart-py")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

    # PostgreSQL Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_xb5wPCsYMQz7@ep-little-smoke-ae82akon-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))

config = Config()