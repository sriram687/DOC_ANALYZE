# ⚡ Quick Deploy Guide - Get Live in 10 Minutes!

## 🎯 **What You'll Get**

A complete full-stack AI document analysis platform:
- **Frontend**: Modern React app on Vercel
- **Backend**: FastAPI service on Render  
- **Features**: Upload docs, ask questions, get AI answers

## 🚀 **Deploy Backend (5 minutes)**

### 1. Deploy to Render
👉 **Go to**: https://dashboard.render.com/

### 2. Create Web Service
- Click **"New +"** → **"Web Service"**
- Connect: `https://github.com/sriram687/DOC_ANALYZE`
- Settings:
  ```
  Name: doc-analyze-api
  Build Command: pip install -r requirements.txt
  Start Command: bash start.sh
  ```

### 3. Add Environment Variables
```
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=doc-analyze
```

### 4. Get API Keys
- **Gemini**: https://makersuite.google.com/app/apikey
- **Pinecone**: https://app.pinecone.io/ (create index: dimension=768, metric=cosine)

### 5. Deploy!
- Click **"Create Web Service"**
- Wait 5-10 minutes
- Note your URL: `https://your-service.onrender.com`

## 🎨 **Deploy Frontend (5 minutes)**

### 1. Create Frontend Repo
```bash
# Navigate to frontend folder
cd frontend

# Initialize git
git init
git add .
git commit -m "React frontend for document analyzer"

# Create new GitHub repository and push
# (Create repo at github.com first)
git remote add origin https://github.com/yourusername/doc-analyzer-frontend.git
git push -u origin main
```

### 2. Deploy to Vercel
👉 **Go to**: https://vercel.com/

- Click **"New Project"**
- Import your frontend repository
- Set environment variable:
  ```
  REACT_APP_API_URL=https://your-service.onrender.com
  ```
- Click **"Deploy"**

### 3. Your App is Live!
- Frontend: `https://your-app.vercel.app`
- Backend: `https://your-service.onrender.com`

## 🧪 **Test Your App**

1. Visit your frontend URL
2. Upload a PDF/DOCX document
3. Ask: "What is this document about?"
4. Get AI-powered answer!

## 🎉 **You're Done!**

Your complete AI document analysis platform is now live:

- ✅ **Modern React Frontend** - Beautiful, responsive UI
- ✅ **FastAPI Backend** - Powerful document processing
- ✅ **AI Integration** - Gemini + Pinecone powered
- ✅ **Production Ready** - Deployed on reliable platforms
- ✅ **Mobile Friendly** - Works on all devices

**Share your app and start analyzing documents with AI!** 🚀

---

### 🆘 **Need Help?**

- **Backend Issues**: Check Render logs
- **Frontend Issues**: Check Vercel deployment logs  
- **API Keys**: Verify they're set correctly
- **CORS Issues**: Ensure backend allows frontend domain

**Your AI-powered document analyzer is ready to use!** ✨
