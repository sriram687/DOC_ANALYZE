# 🚀 Deployment Guide - Enhanced Document Query API

## 📋 Prerequisites

Before deploying, ensure you have:

1. **API Keys**:
   - Google Gemini API Key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Pinecone API Key from [Pinecone Console](https://app.pinecone.io/)

2. **Pinecone Index**:
   - Create a Pinecone index with:
     - **Dimension**: 768 (for Gemini embeddings)
     - **Metric**: cosine
     - **Environment**: us-east-1 (or your preferred region)

## 🌐 Deploy on Render

### Step 1: Push to GitHub

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit - Enhanced Document Query API"

# Add remote repository
git remote add origin https://github.com/sriram687/DOC_ANALYZE.git

# Push to GitHub
git push -u origin main
```

### Step 2: Connect to Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository: `https://github.com/sriram687/DOC_ANALYZE.git`

### Step 3: Configure Render Settings

**Basic Settings:**
- **Name**: `doc-analyze-api`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements_render.txt`
- **Start Command**: `bash start.sh`

**Environment Variables:**
Add these in Render's Environment section:

```
GEMINI_API_KEY=your_actual_gemini_api_key
PINECONE_API_KEY=your_actual_pinecone_api_key
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

### Step 4: Deploy

1. Click "Create Web Service"
2. Render will automatically:
   - Clone your repository
   - Install dependencies
   - Start the application
3. Your API will be available at: `https://your-service-name.onrender.com`

## 🔧 Post-Deployment

### Health Check
Visit: `https://your-service-name.onrender.com/health`

Expected response:
```json
{
  "status": "healthy",
  "model": "gemini-2.5-pro",
  "timestamp": "2025-01-01T12:00:00"
}
```

### API Documentation
Visit: `https://your-service-name.onrender.com/docs`

### Test the API
```bash
curl -X POST "https://your-service-name.onrender.com/ask-document" \
     -F "query=What is this document about?" \
     -F "file=@your_document.pdf"
```

## 📊 Monitoring

### Render Logs
- View logs in Render Dashboard → Your Service → Logs
- Monitor for any startup errors or runtime issues

### Performance
- Free tier: Limited resources, may have cold starts
- Paid tiers: Better performance and uptime

## 🔒 Security Notes

1. **Environment Variables**: Never commit API keys to the repository
2. **HTTPS**: Render provides HTTPS by default
3. **Rate Limiting**: Consider implementing rate limiting for production use

## 🚨 Troubleshooting

### Common Issues

1. **Build Fails**:
   - Check `requirements_render.txt` for dependency conflicts
   - Verify Python version compatibility

2. **App Won't Start**:
   - Check environment variables are set correctly
   - Verify Pinecone index exists and is accessible

3. **API Errors**:
   - Check Gemini API key is valid and has quota
   - Verify Pinecone connection and index configuration

### Debug Commands

```bash
# Check environment variables
echo $GEMINI_API_KEY
echo $PINECONE_API_KEY

# Test Pinecone connection
python -c "from pinecone import Pinecone; pc = Pinecone(api_key='your_key'); print(pc.list_indexes())"

# Test Gemini connection
python -c "import google.generativeai as genai; genai.configure(api_key='your_key'); print('Connected')"
```

## 🎯 Production Recommendations

1. **Upgrade Render Plan**: For better performance and reliability
2. **Add Monitoring**: Set up health checks and alerts
3. **Implement Caching**: Add Redis for better performance
4. **Add Authentication**: Secure your API endpoints
5. **Rate Limiting**: Prevent abuse and manage costs

## 📞 Support

If you encounter issues:
1. Check Render logs for error messages
2. Verify all environment variables are set
3. Test API keys independently
4. Review the troubleshooting section above

Your Enhanced Document Query API is now ready for production! 🎉
