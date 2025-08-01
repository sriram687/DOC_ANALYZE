from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    document_id: str = ""
    embedding: Optional[List[float]] = None

class ClauseMatch(BaseModel):
    clause_id: str
    text: str
    relevance_score: float
    metadata: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    query: str
    answer: str
    conditions: List[str]
    evidence: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    cache_hit: Optional[bool] = None

class DocumentAnalysis(BaseModel):
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    readability_score: float
    document_type: str
    key_topics: List[str]
    complexity_score: float
    chunk_count: int
    avg_chunk_length: float
    processing_timestamp: str
    document_id: str

class ComparisonResult(BaseModel):
    document_summaries: Dict[str, Any]
    aspect_comparisons: Dict[str, Any]
    similarities: Dict[str, Any]
    differences: Dict[str, Any]

class CleanQueryResponse(BaseModel):
    query: str
    answer: str
    conditions: List[str] = []
    confidence: float
    processing_time: float