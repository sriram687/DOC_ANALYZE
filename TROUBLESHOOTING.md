# 🔧 Troubleshooting Guide - Enhanced Document Query API

This guide helps you resolve common issues with the Enhanced Document Query API.

## 🚨 Quick Diagnosis

Run the diagnostic script to identify issues:

```bash
python diagnose_issues.py
```

## 🔍 Common Issues and Solutions

### 1. PostgreSQL SSL Connection Errors

**Error:** `SSL connection has been closed unexpectedly`

**Causes:**
- Network connectivity issues
- SSL certificate problems
- Connection timeout
- Database server overload

**Solutions:**

1. **Update your DATABASE_URL** with better connection parameters:
   ```bash
   # In your .env file
   DATABASE_URL=postgresql://user:pass@host:port/db?sslmode=require&connect_timeout=30&application_name=bajaj_finserv_api
   ```

2. **Test your database connection:**
   ```bash
   python -c "
   import asyncio
   from database_manager import DatabaseManager
   async def test():
       db = DatabaseManager()
       await db.initialize()
       success = await db.test_connection()
       print('✅ Success' if success else '❌ Failed')
   asyncio.run(test())
   "
   ```

3. **Alternative: Disable PostgreSQL** (system will use Pinecone only):
   ```bash
   # Comment out or remove DATABASE_URL from .env
   # DATABASE_URL=...
   ```

### 2. Gemini JSON Parsing Errors

**Error:** `Expecting value: line 1 column 1 (char 0)`

**Causes:**
- Empty responses from Gemini API
- Rate limiting
- API quota exceeded
- Network issues

**Solutions:**

1. **Check your API key and quota:**
   - Visit [Google AI Studio](https://makersuite.google.com/)
   - Verify your API key is active
   - Check your usage limits

2. **Test Gemini connection:**
   ```bash
   python -c "
   import google.generativeai as genai
   from config import config
   genai.configure(api_key=config.GEMINI_API_KEY)
   model = genai.GenerativeModel('gemini-2.5-pro')
   response = model.generate_content('Hello')
   print(response.text)
   "
   ```

3. **Increase rate limiting delay:**
   ```bash
   # In your .env file
   GEMINI_RATE_LIMIT_DELAY=2.0
   ```

### 3. Pinecone Dimension Mismatch

**Error:** Dimension mismatch between Gemini (768) and Pinecone index

**Solution:**

1. **Run the fix script:**
   ```bash
   python fix_pinecone_dimensions.py
   ```

2. **Or manually create a new index:**
   ```python
   from pinecone import Pinecone, ServerlessSpec
   pc = Pinecone(api_key="your_api_key")
   pc.create_index(
       name="bajaj-finserv-gemini",
       dimension=768,  # Correct for Gemini
       metric="cosine",
       spec=ServerlessSpec(cloud="aws", region="us-east-1")
   )
   ```

3. **Update your .env file:**
   ```bash
   PINECONE_INDEX_NAME=bajaj-finserv-gemini
   PINECONE_DIMENSION=768
   ```

### 4. Memory Issues

**Error:** Out of memory during processing

**Solutions:**

1. **Reduce chunk size:**
   ```bash
   # In your .env file
   CHUNK_SIZE=500
   MAX_CONCURRENT_BATCH=2
   ```

2. **Process smaller files:**
   ```bash
   MAX_FILE_SIZE=10485760  # 10MB instead of 50MB
   ```

### 5. Redis Connection Issues

**Error:** Redis connection failed

**Solutions:**

1. **Install and start Redis:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   sudo systemctl start redis-server
   
   # macOS
   brew install redis
   brew services start redis
   
   # Windows (use Docker)
   docker run -d -p 6379:6379 redis:alpine
   ```

2. **Or disable Redis caching:**
   ```bash
   # Comment out REDIS_URL in .env
   # REDIS_URL=redis://localhost:6379
   ```

## 🔧 Step-by-Step Recovery Process

### If the API is completely broken:

1. **Run diagnostics:**
   ```bash
   python diagnose_issues.py
   ```

2. **Fix critical issues first:**
   - Ensure GEMINI_API_KEY is valid
   - Ensure PINECONE_API_KEY is valid
   - Fix dimension mismatch if present

3. **Test individual components:**
   ```bash
   # Test Gemini
   python -c "from gemini_parser import GeminiParser; import asyncio; asyncio.run(GeminiParser().parse_query_intent('test'))"
   
   # Test Pinecone
   python -c "from embedding_search import GeminiEmbeddingSearchEngine; engine = GeminiEmbeddingSearchEngine()"
   ```

4. **Restart the application:**
   ```bash
   python main.py
   ```

### If only database issues:

1. **The system can work without PostgreSQL** - it will use Pinecone for storage
2. **Comment out DATABASE_URL** in your .env file
3. **Restart the application**

## 📊 Performance Optimization

### For better performance:

1. **Enable Redis caching:**
   ```bash
   REDIS_URL=redis://localhost:6379
   CACHE_TTL_EMBEDDINGS=3600
   CACHE_TTL_RESPONSES=1800
   ```

2. **Optimize chunk settings:**
   ```bash
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   TOP_K_CHUNKS=5
   ```

3. **Reduce concurrent processing:**
   ```bash
   MAX_CONCURRENT_BATCH=3
   ```

## 🆘 Getting Help

### Log Analysis

1. **Enable debug logging:**
   ```bash
   LOG_LEVEL=DEBUG
   ```

2. **Check specific error patterns:**
   - `SSL connection has been closed` → Database connection issue
   - `Expecting value: line 1 column 1` → Gemini API issue
   - `Failed to upsert vectors` → Pinecone issue
   - `dimension mismatch` → Configuration issue

### Contact Information

If you're still having issues:

1. **Check the logs** for specific error messages
2. **Run the diagnostic script** and share the output
3. **Verify your API keys** are valid and have sufficient quota
4. **Check your network connectivity** to external services

## 🔄 Recovery Commands

Quick commands to reset and restart:

```bash
# Full diagnostic
python diagnose_issues.py

# Fix Pinecone dimensions
python fix_pinecone_dimensions.py

# Test individual components
python -c "import asyncio; from database_manager import DatabaseManager; asyncio.run(DatabaseManager().test_connection())"

# Restart with minimal config (no database, no Redis)
# Comment out DATABASE_URL and REDIS_URL in .env, then:
python main.py
```

## ✅ Health Check Endpoints

Once the API is running, test these endpoints:

```bash
# Basic health check
curl http://localhost:3000/health

# Test document processing
curl -X POST "http://localhost:3000/ask-document" \
     -F "query=What is this document about?" \
     -F "file=@test_document.pdf"
```
