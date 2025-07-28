# 🚀 Enhanced Document Query API

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated AI-powered document analysis system that leverages **Google Gemini AI**, **Pinecone vector database**, and **PostgreSQL** for intelligent document processing, semantic search, and real-time querying with advanced caching and analytics.

## 🏗️ Technology Stack

- **🤖 AI & ML**: Google Gemini AI, Sentence Transformers
- **🗄️ Databases**: Pinecone (Vector), PostgreSQL (Relational)
- **🌐 Web Framework**: FastAPI, Uvicorn
- **📄 Document Processing**: PyMuPDF, python-docx
- **⚡ Caching**: Redis (optional)
- **🔧 Infrastructure**: Cloud-deployable, Production-ready

## ✨ Features

- **Multi-format Document Processing**: PDF, DOCX, EML, TXT, and image files (with OCR)
- **Gemini Integration**: Advanced query parsing and answer generation using Google Gemini 2.0 Flash
- **Hybrid Embedding Search**: Combines Gemini embeddings with sentence transformers for optimal results
- **Intelligent Caching**: Redis-based caching for improved performance
- **Document Analytics**: Comprehensive document analysis including readability, complexity, and topic extraction
- **Batch Processing**: Process multiple documents concurrently
- **Real-time Updates**: WebSocket support for live processing updates
- **Document Comparison**: Compare multiple documents across specified aspects

## 🏗️ Architecture

The system is modularized into the following components:

```
├── config.py              # Configuration management
├── models.py               # Pydantic data models
├── cache_manager.py        # Redis caching system
├── document_processor.py   # Document text extraction
├── query_optimizer.py      # Query optimization and intent classification
├── document_analytics.py   # Document analysis and insights
├── gemini_parser.py        # Gemini API integration
├── embedding_search.py     # Enhanced embedding search engine
├── batch_processor.py      # Batch and streaming processing
├── api_handler.py          # Main API orchestration
├── main.py                 # FastAPI application
└── requirements.txt        # Dependencies
```

## 🚀 Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd BAJAJ_FINSERV
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
# Update .env file with your API keys:
# - GEMINI_API_KEY (Google AI)
# - PINECONE_API_KEY (Pinecone)
# - DATABASE_URL (PostgreSQL/Neon)
```

### 5. Run the application
```bash
python main.py
```

### 6. Access the API
- **Server**: http://localhost:3000
- **API Documentation**: http://localhost:3000/docs
- **Health Check**: http://localhost:3000/health



### 4. Install optional dependencies

**For OCR support:**
```bash
# Install Tesseract OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

**For Redis caching:**
```bash
# Install Redis server
# Ubuntu/Debian:
sudo apt-get install redis-server

# macOS:
brew install redis

# Windows: Use Docker or WSL
```

## ⚙️ Configuration

### Required Environment Variables
- `GEMINI_API_KEY`: Your Google AI API key (required)
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID (optional)

### Optional Configuration
- `REDIS_URL`: Redis connection URL for caching
- `USE_GEMINI_EMBEDDINGS`: Enable Gemini embeddings (default: true)
- `ENABLE_HYBRID_SEARCH`: Enable hybrid search combining multiple models
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: 50MB)
- `CHUNK_SIZE`: Text chunk size for processing (default: 1000)

## 🔧 Usage

### Starting the API
```bash
python main.py
```
The API will be available at `http://localhost:8000`

### API Documentation
Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📚 Key Endpoints

### 1. Basic Document Query
```bash
curl -X POST "http://localhost:8000/ask-document" \
     -F "query=What does this policy cover?" \
     -F "file=@document.pdf"
```

### 2. Gemini Enhanced Query (with hybrid search)
```bash
curl -X POST "http://localhost:8000/ask-document-gemini" \
     -F "query=What are the eligibility requirements?" \
     -F "file=@document.pdf"
```

### 3. Document Analysis
```bash
curl -X POST "http://localhost:8000/analyze-document" \
     -F "file=@document.pdf"
```

### 4. Query Suggestions
```bash
curl -X POST "http://localhost:8000/suggest-queries" \
     -F "file=@document.pdf" \
     -F "num_suggestions=5"
```

### 5. Document Comparison
```bash
curl -X POST "http://localhost:8000/compare-documents" \
     -F "files=@doc1.pdf" \
     -F "files=@doc2.pdf" \
     -F "comparison_aspects=coverage" \
     -F "comparison_aspects=conditions"
```

## 📋 Response Format

### Query Response
```json
{
  "query": "What does this policy cover?",
  "answer": "This policy covers...",
  "conditions": ["Must be enrolled", "Valid ID required"],
  "evidence": [
    {
      "clause_id": "section_1",
      "text": "Relevant text from document",
      "relevance": "Why this clause is relevant"
    }
  ],
  "confidence": 0.85,
  "processing_time": 2.34
}
```

### Document Analysis Response
```json
{
  "word_count": 1500,
  "sentence_count": 75,
  "avg_sentence_length": 20.0,
  "readability_score": 45.2,
  "document_type": "insurance_policy",
  "key_topics": ["coverage", "benefits", "claims"],
  "complexity_score": 65.8,
  "chunk_count": 12,
  "avg_chunk_length": 125.0,
  "processing_timestamp": "2024-01-15T10:30:00Z",
  "document_id": "doc_123"
}
```

## 🚀 Advanced Features

### Caching
The system supports Redis-based caching for:
- Document embeddings (1 hour TTL)
- Query responses (30 minutes TTL)

Enable caching by setting up Redis and configuring `REDIS_URL`.

### Hybrid Search
Combines Gemini embeddings with sentence transformers for optimal search results:
- **Primary**: Gemini embeddings (768 dimensions)
- **Fallback**: Sentence transformers (384 dimensions)
- **Weighted scoring**: 70% Gemini + 30% Sentence transformer

### Batch Processing
Process multiple documents concurrently:
```bash
curl -X POST "http://localhost:8000/batch-process" \
     -F "queries=Query 1" \
     -F "queries=Query 2" \
     -F "files=@doc1.pdf" \
     -F "files=@doc2.pdf"
```

### WebSocket Support
Real-time processing updates via WebSocket:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/process-document');
ws.send(JSON.stringify({query: 'What does this cover?'}));
```

## ⚡ Performance Optimization

### Embedding Caching
- Document embeddings are cached to avoid recomputation
- Query embeddings use rate limiting for Gemini API

### Concurrent Processing
- Configurable concurrency limits for batch processing
- Async processing throughout the system

### Memory Management
- Streaming document processing for large files
- Efficient FAISS indexing for similarity search

## 🛠️ Error Handling

The system includes comprehensive error handling:
- Graceful fallback from Gemini to sentence transformers
- File format validation and size limits
- API rate limiting compliance
- Detailed error messages and logging

## 👩‍💻 Development

### Running Tests
```bash
pytest tests/
```

### Code Structure
Each module has a specific responsibility:
- `config.py`: Centralized configuration
- `models.py`: Type-safe data models
- `document_processor.py`: File processing and OCR
- `gemini_parser.py`: AI model interactions
- `embedding_search.py`: Advanced search capabilities

### Adding New Features
1. Add configuration to `config.py`
2. Define models in `models.py`
3. Implement logic in appropriate module
4. Add endpoint to `main.py`
5. Update documentation

## 🔧 Troubleshooting

### Common Issues

#### Gemini API Errors
- Verify API key is correct
- Check rate limits and quotas
- Ensure model names are up to date

#### OCR Not Working
- Install Tesseract OCR system package
- Verify pytesseract can find Tesseract executable
- Check image quality and format

#### Redis Connection Issues
- Verify Redis server is running
- Check Redis URL configuration
- System will work without Redis (caching disabled)

#### Memory Issues
- Reduce `CHUNK_SIZE` for large documents
- Lower `MAX_CONCURRENT_BATCH` for batch processing
- Monitor system resources

### Logging
Set log level in environment:
```bash
export LOG_LEVEL=DEBUG
```

Logs include:
- Processing times
- Cache hit/miss rates
- Error details and stack traces
- Performance metrics

## 📊 Project Status

![GitHub last commit](https://img.shields.io/github/last-commit/username/enhanced-document-query-api)
![GitHub issues](https://img.shields.io/github/issues/username/enhanced-document-query-api)
![GitHub pull requests](https://img.shields.io/github/issues-pr/username/enhanced-document-query-api)
![GitHub stars](https://img.shields.io/github/stars/username/enhanced-document-query-api)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contribution Steps
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Documentation

- **[API Documentation](http://localhost:3000/docs)** - Interactive API docs
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Changelog](CHANGELOG.md)** - Version history
- **[License](LICENSE)** - MIT License

## 🙏 Acknowledgments

- **Google AI** for the Gemini API
- **Pinecone** for vector database services
- **FastAPI** for the excellent web framework
- **The open-source community** for amazing tools and libraries

---

**⭐ Star this repository if you find it helpful!**

**Built with ❤️ using Google Gemini AI, Pinecone, and PostgreSQL**