# 🚀 AI-Powered Document Analyzer - Full-Stack Application

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://langchain.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue.svg)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3+-blue.svg)](https://tailwindcss.com/)

A complete full-stack AI document analysis platform with modern React frontend and powerful FastAPI backend. Upload documents, ask questions in natural language, and get instant AI-powered answers using advanced LangChain RAG pipeline.

## 🌟 **Live Demo**

- **🎨 Frontend**: [Deploy on Vercel](https://vercel.com/new/clone?repository-url=https://github.com/sriram687/DOC_ANALYZE)
- **⚡ Backend API**: [Deploy on Render](https://render.com/deploy?repo=https://github.com/sriram687/DOC_ANALYZE)
- **📚 API Docs**: Interactive documentation available after deployment

## ✨ **Features**

### 🎨 **Modern Frontend (React + TypeScript)**
- **Beautiful UI** with Tailwind CSS and Lucide icons
- **Drag & Drop Upload** with file validation and preview
- **Real-time Processing** with animated loading states
- **Voice Input Support** for natural language queries
- **Professional Results** with copy/download functionality
- **Mobile Responsive** design for all devices

### ⚡ **Powerful Backend (FastAPI + LangChain)**
- **LangChain RAG Pipeline** for advanced document analysis
- **Multi-format Support** (PDF, DOCX, images with OCR)
- **Gemini 2.5 Pro Integration** for AI responses
- **Pinecone Vector Store** for semantic search
- **Clean Response Format** optimized for users
- **Production Ready** with comprehensive error handling

### 🤖 **AI Capabilities**
- **Natural Language Processing** - Ask questions in plain English
- **Document Understanding** - Extracts meaning from complex documents
- **Contextual Answers** - Provides relevant, accurate responses
- **Confidence Scoring** - Shows reliability of answers
- **Multi-language Support** - Works with various languages

## 🏗️ **Architecture**

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   React Frontend    │    │   FastAPI Backend   │    │   AI Services       │
│   (Vercel)          │◄──►│   (Render)          │◄──►│   (Gemini/Pinecone) │
│                     │    │                     │    │                     │
│ • Modern UI/UX      │    │ • Document Process  │    │ • Vector Search     │
│ • File Upload       │    │ • LangChain RAG     │    │ • AI Generation     │
│ • Real-time Results │    │ • Clean Responses   │    │ • Embeddings        │
│ • Mobile Responsive │    │ • Production Ready  │    │ • Semantic Search   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 📁 **Project Structure**

```
DOC_ANALYZE/
├── 🔧 backend/                       # FastAPI + LangChain
│   ├── main.py                       # Application entry point
│   ├── src/
│   │   ├── models/                   # Data models
│   │   └── services/                 # Business logic
│   ├── config/                       # Configuration
│   └── requirements.txt              # Dependencies
│
├── 🎨 frontend/                      # React + TypeScript
│   ├── src/
│   │   ├── components/               # React components
│   │   ├── services/                 # API integration
│   │   └── types/                    # TypeScript types
│   └── package.json                  # Dependencies
│
├── 📖 docs/                          # Documentation
├── 🗂️ legacy/                       # Archived code
└── 📄 README.md                     # This file
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed organization.

## 🚀 **Quick Start**

### **Option 1: Run Locally (5 minutes)**

```bash
# 1. Clone the repository
git clone https://github.com/sriram687/DOC_ANALYZE.git
cd DOC_ANALYZE

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Start Backend (Terminal 1)
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py

# 4. Start Frontend (Terminal 2)
cd frontend
npm install
npm start

# 5. Open your browser
# Frontend: http://localhost:3000
# Backend API: http://127.0.0.1:3000/docs
```

### **Option 2: Deploy to Production (10 minutes)**

1. **Deploy Backend to Render**:
   - Connect GitHub repository
   - Set environment variables (API keys)
   - Deploy automatically

2. **Deploy Frontend to Vercel**:
   - Import GitHub repository
   - Set `REACT_APP_API_URL` environment variable
   - Deploy with one click

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed instructions.

## 🔑 **Required API Keys**

### **Gemini API Key**
1. Visit: https://makersuite.google.com/app/apikey
2. Create API key
3. Copy key (starts with `AIza...`)

### **Pinecone Setup**
1. Visit: https://app.pinecone.io/
2. Create account and new index:
   - **Name**: `doc-analyze`
   - **Dimension**: `768`
   - **Metric**: `cosine`
3. Copy API key from Settings

## 🎯 **Use Cases**

- **📄 Document Analysis** - Extract insights from PDFs, contracts, reports
- **🏥 Insurance Claims** - Analyze policy documents and coverage details
- **📋 Legal Documents** - Review contracts, agreements, and legal texts
- **📊 Research Papers** - Summarize and query academic documents
- **📝 Business Reports** - Extract key information from business documents
- **🎓 Educational Content** - Analyze textbooks, study materials

## 🛠️ **Tech Stack**

### **Frontend**
- **React 18** with TypeScript for type safety
- **Tailwind CSS** for modern, responsive styling
- **Lucide React** for beautiful icons
- **Axios** for API communication
- **React Dropzone** for file uploads

### **Backend**
- **FastAPI** for high-performance API
- **LangChain** for RAG pipeline
- **Google Gemini 2.5 Pro** for AI responses
- **Pinecone** for vector storage
- **Python 3.13** compatible

### **Deployment**
- **Frontend**: Vercel (automatic deployments)
- **Backend**: Render (containerized deployment)
- **Database**: Pinecone (managed vector database)

## 📊 **Performance**

- **⚡ Fast Response Times**: 5-15 seconds for document analysis
- **📱 Mobile Optimized**: Perfect performance on all devices
- **🔄 Real-time Updates**: Live progress indicators
- **📈 Scalable**: Handles multiple concurrent users
- **🛡️ Reliable**: Production-grade error handling

## 📖 **Documentation**

- **[Quick Deploy Guide](docs/QUICK_DEPLOY.md)** - 10-minute setup
- **[Full Deployment Guide](docs/DEPLOYMENT.md)** - Complete instructions
- **[Project Structure](PROJECT_STRUCTURE.md)** - Folder organization
- **[Frontend Setup](docs/FRONTEND_SETUP.md)** - React app details

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **LangChain** for the powerful RAG framework
- **Google Gemini** for advanced AI capabilities
- **Pinecone** for vector database infrastructure
- **Vercel & Render** for seamless deployment platforms

---

**🌟 Star this repository if you find it helpful!**

**Built with ❤️ using React, FastAPI, LangChain, and AI**
