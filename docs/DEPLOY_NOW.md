# 🚀 Deploy Your API to Render NOW!

## ✅ **FIXED: Requirements Issue Resolved**

The Python compatibility issue has been fixed! Your repository is now ready for deployment.

## 🎯 **Quick Deploy Steps**

### 1. Go to Render Dashboard
👉 **Click here**: https://dashboard.render.com/

### 2. Create New Web Service
- Click **"New +"** → **"Web Service"**
- Choose **"Build and deploy from a Git repository"**
- Click **"Connect account"** (if not connected) or **"Configure account"**

### 3. Connect Repository
- Find and select: **`sriram687/DOC_ANALYZE`**
- Click **"Connect"**

### 4. Configure Service Settings
```
Name: doc-analyze-api
Environment: Python 3
Branch: main
Build Command: pip install -r requirements.txt
Start Command: bash start.sh
```

### 5. Add Environment Variables
**CRITICAL**: Add these environment variables in Render:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here
PINECONE_API_KEY=your_actual_pinecone_api_key_here
PINECONE_INDEX_NAME=your_pinecone_index_name
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

### 6. Deploy!
- Click **"Create Web Service"**
- Wait 5-10 minutes for deployment
- Your API will be live! 🎉

## 🔑 **Get Your API Keys First**

### Gemini API Key
1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key (starts with `AIza...`)

### Pinecone Setup
1. Go to: https://app.pinecone.io/
2. Create account/login
3. Create new index:
   - **Name**: `doc-analyze`
   - **Dimension**: `768`
   - **Metric**: `cosine`
4. Copy your API key from Settings

## 🧪 **Test After Deployment**

Once deployed, test these URLs:

1. **Health Check**: `https://your-app-name.onrender.com/health`
2. **API Docs**: `https://your-app-name.onrender.com/docs`
3. **Upload Test**: Use the docs page to test file upload

## 🎉 **Success!**

Your Enhanced Document Query API will be live and ready to:
- ✅ Process PDF, DOCX, and image documents
- ✅ Answer questions using AI
- ✅ Provide professional, clean responses
- ✅ Handle multiple file formats
- ✅ Scale automatically

**Deploy now and start using your AI-powered document analysis API!** 🚀
