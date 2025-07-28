import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import uuid

from sqlalchemy import create_engine, Column, String, Text, DateTime, Float, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import UUID

from config import config

logger = logging.getLogger(__name__)

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    content_hash = Column(String(64), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    doc_metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    chunk_metadata = Column(JSON, default={})
    embedding_id = Column(String(255))  # Pinecone vector ID
    created_at = Column(DateTime, default=datetime.utcnow)

class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    document_id = Column(UUID(as_uuid=True))
    response = Column(JSON)
    confidence_score = Column(Float)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """PostgreSQL database manager for document storage and retrieval"""
    
    def __init__(self):
        self.database_url = config.DATABASE_URL
        # Prepare URLs for different drivers
        if self.database_url.startswith("postgresql://"):
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(self.database_url)

            # Create asyncpg URL (remove sslmode, use ssl)
            asyncpg_query = "ssl=require" if "sslmode=require" in (parsed.query or "") else ""
            asyncpg_parsed = parsed._replace(
                scheme="postgresql+asyncpg",
                query=asyncpg_query
            )
            self.async_database_url = urlunparse(asyncpg_parsed)

            # Create psycopg URL (keep original format)
            psycopg_parsed = parsed._replace(scheme="postgresql+psycopg")
            self.psycopg_url = urlunparse(psycopg_parsed)
        else:
            self.async_database_url = self.database_url
            self.psycopg_url = self.database_url
            
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        
    async def initialize(self):
        """Initialize database connections and create tables"""
        import asyncio
        import sys

        # Set correct event loop policy for Windows + psycopg BEFORE creating engine
        if sys.platform == "win32":
            try:
                # Get current event loop policy
                current_policy = asyncio.get_event_loop_policy()
                if not isinstance(current_policy, asyncio.WindowsSelectorEventLoopPolicy):
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                    logger.info("🔧 Set Windows selector event loop policy for psycopg compatibility")
            except Exception as e:
                logger.warning(f"Could not set event loop policy: {e}")

        # Try psycopg first (better Windows compatibility), then asyncpg
        drivers_to_try = [
            ("psycopg", self.psycopg_url),
            ("asyncpg", self.async_database_url)
        ]

        for driver_name, url in drivers_to_try:
            try:
                logger.info(f"🔄 Trying PostgreSQL connection with {driver_name}...")

                # Create async engine
                self.async_engine = create_async_engine(
                    url,
                    pool_size=3,
                    max_overflow=5,
                    pool_timeout=20,
                    pool_recycle=1800,
                    echo=False,
                    connect_args={
                        "server_settings": {
                            "application_name": "bajaj_finserv_api",
                        }
                    } if driver_name == "asyncpg" else {}
                )

                # Create async session factory
                self.AsyncSessionLocal = async_sessionmaker(
                    self.async_engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )

                # Test connection with timeout
                logger.info(f"📡 Testing {driver_name} connection...")
                async with self.async_engine.begin() as conn:
                    await asyncio.wait_for(
                        conn.run_sync(Base.metadata.create_all),
                        timeout=20.0
                    )

                logger.info(f"✅ PostgreSQL database initialized successfully with {driver_name}")
                return

            except Exception as e:
                logger.warning(f"❌ {driver_name} connection failed: {e}")
                if self.async_engine:
                    await self.async_engine.dispose()
                    self.async_engine = None

                # If this was the last driver, re-raise the exception
                if driver_name == "psycopg2":
                    raise

                # Otherwise, try the next driver
                continue

        raise Exception("All database drivers failed")
    
    async def close(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()

    async def test_connection(self):
        """Test database connection without creating tables"""
        import asyncio
        try:
            logger.info("🔍 Testing database connection...")
            async with self.async_engine.connect() as conn:
                result = await asyncio.wait_for(
                    conn.execute("SELECT 1"),
                    timeout=10.0
                )
                await result.fetchone()
            logger.info("✅ Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    async def store_document(self, filename: str, content: str, content_hash: str, metadata: Dict = None) -> str:
        """Store a document in the database"""
        async with self.AsyncSessionLocal() as session:
            try:
                document = Document(
                    filename=filename,
                    content=content,
                    content_hash=content_hash,
                    doc_metadata=metadata or {}
                )
                session.add(document)
                await session.commit()
                await session.refresh(document)

                logger.info(f"Document stored with ID: {document.id}")
                return str(document.id)

            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to store document: {e}")
                raise
    
    async def get_document(self, document_id: str) -> Optional[Dict]:
        """Retrieve a document by ID"""
        async with self.AsyncSessionLocal() as session:
            try:
                from sqlalchemy import select
                result = await session.execute(
                    select(Document).where(Document.id == document_id)
                )
                document = result.scalar_one_or_none()
                
                if document:
                    return {
                        "id": str(document.id),
                        "filename": document.filename,
                        "content": document.content,
                        "content_hash": document.content_hash,
                        "metadata": document.doc_metadata,
                        "created_at": document.created_at,
                        "updated_at": document.updated_at
                    }
                return None
                
            except Exception as e:
                logger.error(f"Failed to retrieve document: {e}")
                return None
    
    async def store_document_chunks(self, document_id: str, chunks: List[Dict]) -> List[str]:
        """Store document chunks in the database"""
        async with self.AsyncSessionLocal() as session:
            try:
                chunk_ids = []
                for i, chunk_data in enumerate(chunks):
                    chunk = DocumentChunk(
                        document_id=document_id,
                        chunk_index=i,
                        content=chunk_data.get("content", ""),
                        chunk_metadata=chunk_data.get("metadata", {}),
                        embedding_id=chunk_data.get("embedding_id")
                    )
                    session.add(chunk)
                    chunk_ids.append(str(chunk.id))
                
                await session.commit()
                logger.info(f"Stored {len(chunks)} chunks for document {document_id}")
                return chunk_ids
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to store document chunks: {e}")
                raise
    
    async def get_document_chunks(self, document_id: str) -> List[Dict]:
        """Retrieve all chunks for a document"""
        async with self.AsyncSessionLocal() as session:
            try:
                from sqlalchemy import select
                result = await session.execute(
                    select(DocumentChunk)
                    .where(DocumentChunk.document_id == document_id)
                    .order_by(DocumentChunk.chunk_index)
                )
                chunks = result.scalars().all()
                
                return [
                    {
                        "id": str(chunk.id),
                        "document_id": str(chunk.document_id),
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "metadata": chunk.chunk_metadata,
                        "embedding_id": chunk.embedding_id,
                        "created_at": chunk.created_at
                    }
                    for chunk in chunks
                ]
                
            except Exception as e:
                logger.error(f"Failed to retrieve document chunks: {e}")
                return []
    
    async def log_query(self, query: str, document_id: str = None, response: Dict = None, 
                       confidence_score: float = None, processing_time: float = None) -> str:
        """Log a query and its response"""
        async with self.AsyncSessionLocal() as session:
            try:
                query_log = QueryLog(
                    query=query,
                    document_id=document_id,
                    response=response,
                    confidence_score=confidence_score,
                    processing_time=processing_time
                )
                session.add(query_log)
                await session.commit()
                await session.refresh(query_log)
                
                return str(query_log.id)
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to log query: {e}")
                raise
    
    async def get_document_by_hash(self, content_hash: str) -> Optional[Dict]:
        """Check if a document with the same content hash already exists"""
        async with self.AsyncSessionLocal() as session:
            try:
                from sqlalchemy import select
                result = await session.execute(
                    select(Document).where(Document.content_hash == content_hash)
                )
                document = result.scalar_one_or_none()
                
                if document:
                    return {
                        "id": str(document.id),
                        "filename": document.filename,
                        "content": document.content,
                        "content_hash": document.content_hash,
                        "metadata": document.doc_metadata,
                        "created_at": document.created_at,
                        "updated_at": document.updated_at
                    }
                return None
                
            except Exception as e:
                logger.error(f"Failed to check document hash: {e}")
                return None
