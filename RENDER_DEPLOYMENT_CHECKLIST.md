# 🚀 Render Deployment Checklist

## ✅ Pre-Deployment Setup Complete

### 📁 Repository Status
- ✅ Code pushed to GitHub: https://github.com/sriram687/DOC_ANALYZE.git
- ✅ Render configuration files created
- ✅ Dependencies optimized for production
- ✅ Unnecessary files removed

### 📋 Required API Keys & Setup

Before deploying on Render, ensure you have:

#### 1. Google Gemini API Key
- ✅ Get from: https://makersuite.google.com/app/apikey
- ✅ Copy the API key (starts with `AIza...`)

#### 2. Pinecone Setup
- ✅ Create account at: https://app.pinecone.io/
- ✅ Create a new index with these settings:
  - **Name**: `doc-analyze` (or your preferred name)
  - **Dimension**: `768`
  - **Metric**: `cosine`
  - **Environment**: `us-east-1`
- ✅ Copy your Pinecone API key

## 🌐 Deploy on Render

### Step 1: Create Web Service
1. Go to https://dashboard.render.com/
2. Click "New +" → "Web Service"
3. Connect GitHub repository: `https://github.com/sriram687/DOC_ANALYZE.git`

### Step 2: Configure Service
**Basic Settings:**
- **Name**: `doc-analyze-api`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements_render.txt`
- **Start Command**: `bash start.sh`

### Step 3: Environment Variables
Add these in Render's Environment section:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here
PINECONE_API_KEY=your_actual_pinecone_api_key_here
PINECONE_INDEX_NAME=doc-analyze
PINECONE_ENVIRONMENT=us-east-1
PINECONE_DIMENSION=768
PINECONE_METRIC=cosine
GEMINI_CHAT_MODEL=gemini-2.5-pro
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_CHUNKS=10
MAX_FILE_SIZE=52428800
```

### Step 4: Deploy
1. Click "Create Web Service"
2. Wait for deployment (5-10 minutes)
3. Your API will be live at: `https://your-service-name.onrender.com`

## 🧪 Post-Deployment Testing

### 1. Health Check
Visit: `https://your-service-name.onrender.com/health`

Expected response:
```json
{
  "status": "healthy",
  "model": "gemini-2.5-pro",
  "timestamp": "2025-01-01T12:00:00"
}
```

### 2. API Documentation
Visit: `https://your-service-name.onrender.com/docs`

### 3. Test Document Query
```bash
curl -X POST "https://your-service-name.onrender.com/ask-document" \
     -F "query=What is this document about?" \
     -F "file=@test_document.pdf"
```

## 🎯 Success Indicators

✅ **Deployment Status**: "Live" in Render dashboard
✅ **Health Check**: Returns 200 OK
✅ **API Docs**: Loads successfully
✅ **Document Upload**: Accepts files without errors
✅ **Query Processing**: Returns structured responses
✅ **No Error Logs**: Clean startup in Render logs

## 🚨 Troubleshooting

### Common Issues:

1. **Build Fails**
   - Check requirements_render.txt for dependency conflicts
   - Verify Python version compatibility

2. **App Won't Start**
   - Verify all environment variables are set
   - Check Pinecone index exists and is accessible
   - Validate Gemini API key

3. **API Errors**
   - Check Render logs for detailed error messages
   - Verify API keys have proper permissions
   - Ensure Pinecone index dimension matches (768)

### Debug Steps:
1. Check Render logs: Dashboard → Your Service → Logs
2. Verify environment variables are set correctly
3. Test API keys independently before deployment

## 🎉 You're Ready!

Once deployed successfully, your Enhanced Document Query API will be:
- 🌐 **Live**: Accessible worldwide
- 📚 **Documented**: Interactive API docs available
- 🔒 **Secure**: HTTPS enabled by default
- 📊 **Monitored**: Render provides basic monitoring
- 🚀 **Scalable**: Can handle multiple concurrent requests

**Your API URL**: `https://your-service-name.onrender.com`

Share this URL to start using your document analysis API! 🎯
