import asyncio
import logging
from typing import List, Dict, Any, Tuple

from config import config

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Process multiple documents in batch"""
    
    def __init__(self, api_handler):
        self.api_handler = api_handler
        self.max_concurrent = config.MAX_CONCURRENT_BATCH
    
    async def process_batch(self, queries_and_files: List[tuple]) -> List[Dict]:
        """Process multiple query-file pairs concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_single(query, file):
            async with semaphore:
                try:
                    return await self.api_handler.process_query(query, file)
                except Exception as e:
                    return {
                        "query": query,
                        "error": str(e),
                        "status": "failed"
                    }
        
        tasks = [process_single(query, file) for query, file in queries_and_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            result if not isinstance(result, Exception) 
            else {"error": str(result), "status": "failed"}
            for result in results
        ]

class StreamingProcessor:
    """Process large documents in streaming fashion"""
    
    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size
        
    async def process_large_document(self, file_stream, callback):
        """Process document in chunks with progress callbacks"""
        chunks = []
        total_size = 0
        
        async for chunk in self._read_in_chunks(file_stream):
            chunks.append(chunk)
            total_size += len(chunk)
            
            # Progress callback
            if callback:
                await callback({
                    'status': 'reading',
                    'bytes_processed': total_size,
                    'chunks_count': len(chunks)
                })
        
        # Combine chunks and process
        full_content = b''.join(chunks)
        
        if callback:
            await callback({
                'status': 'processing',
                'total_size': total_size,
                'stage': 'text_extraction'
            })
        
        return full_content
    
    async def _read_in_chunks(self, file_stream):
        """Read file in chunks asynchronously"""
        while True:
            chunk = await file_stream.read(self.chunk_size)
            if not chunk:
                break
            yield chunk