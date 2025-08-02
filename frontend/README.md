# Document Analyzer Frontend

A modern React application for AI-powered document analysis. Upload documents and ask questions in natural language to get instant, accurate answers.

## 🚀 Features

- **Drag & Drop Upload** - Easy file upload with visual feedback
- **Multiple Formats** - Support for PDF, DOCX, TXT, EML, PNG, JPG
- **Natural Language Queries** - Ask questions in plain English
- **Real-time Analysis** - Get instant AI-powered responses
- **Beautiful UI** - Modern, responsive design with Tailwind CSS
- **Voice Input** - Speech-to-text support (where available)
- **Mobile Responsive** - Works perfectly on all devices

## 🛠️ Tech Stack

- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Lucide React** for icons
- **Axios** for API communication
- **React Dropzone** for file uploads

## 🏃‍♂️ Quick Start

### Prerequisites
- Node.js 16+ 
- npm or yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/sriram687/DOC_ANALYZE.git
cd DOC_ANALYZE/frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env.local

# Update .env.local with your API URL
REACT_APP_API_URL=https://your-api-name.onrender.com

# Start development server
npm start
```

The app will open at `http://localhost:3000`

## 🌐 Deployment

### Deploy to Vercel (Recommended)

1. **Push to GitHub** (if not already done)
2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Set environment variables:
     ```
     REACT_APP_API_URL=https://your-api-name.onrender.com
     ```
3. **Deploy** - Vercel will automatically build and deploy

### Deploy to Netlify

1. **Build the project**:
   ```bash
   npm run build
   ```
2. **Deploy to Netlify**:
   - Drag the `build` folder to [netlify.com/drop](https://app.netlify.com/drop)
   - Or connect your GitHub repository

## 📁 Project Structure

```
frontend/
├── public/
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── components/
│   │   ├── FileUpload.tsx      # File upload component
│   │   ├── QueryInput.tsx      # Query input with voice support
│   │   ├── ResultsDisplay.tsx  # Results display component
│   │   └── LoadingSpinner.tsx  # Loading states
│   ├── services/
│   │   └── api.ts             # API service layer
│   ├── types/
│   │   └── index.ts           # TypeScript type definitions
│   ├── App.tsx                # Main application component
│   ├── App.css                # Custom styles
│   ├── index.tsx              # React entry point
│   └── index.css              # Global styles with Tailwind
├── package.json
├── tailwind.config.js
└── README.md
```

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file:

```env
# Backend API URL
REACT_APP_API_URL=https://your-api-name.onrender.com

# For local development
# REACT_APP_API_URL=http://localhost:3001
```

### API Integration

The frontend communicates with the FastAPI backend through:

- **Health Check**: `GET /health`
- **Document Query**: `POST /ask-document-clean`
- **File Upload**: Multipart form data

## 🎨 Customization

### Styling
- Modify `tailwind.config.js` for theme customization
- Update `src/index.css` for global styles
- Component-specific styles in individual files

### Features
- Add new components in `src/components/`
- Extend API service in `src/services/api.ts`
- Update types in `src/types/index.ts`

## 🧪 Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage
```

## 📦 Building

```bash
# Create production build
npm run build

# Serve build locally
npx serve -s build
```

## 🚀 Performance

- **Code Splitting** - Automatic with React
- **Lazy Loading** - Components loaded on demand
- **Optimized Images** - Automatic optimization
- **CDN Delivery** - Via Vercel/Netlify

## 🔒 Security

- **Environment Variables** - Sensitive data in env files
- **HTTPS Only** - Secure communication
- **Input Validation** - Client-side validation
- **CORS Handling** - Proper cross-origin setup

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check if backend is deployed and running
   - Verify `REACT_APP_API_URL` is correct
   - Check CORS settings on backend

2. **File Upload Issues**
   - Ensure file size is under 50MB
   - Check supported file formats
   - Verify network connection

3. **Build Errors**
   - Clear node_modules: `rm -rf node_modules && npm install`
   - Check Node.js version compatibility
   - Update dependencies: `npm update`

## 📞 Support

- **GitHub Issues**: [Report bugs](https://github.com/sriram687/DOC_ANALYZE/issues)
- **Documentation**: Check README files
- **API Status**: Monitor backend health endpoint

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

Built with ❤️ using React, TypeScript, and Tailwind CSS
