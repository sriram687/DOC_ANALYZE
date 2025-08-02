import React, { useState, useEffect } from 'react';
import { FileText, Zap, Shield, Globe, Github, ExternalLink } from 'lucide-react';
import FileUpload from './components/FileUpload.tsx';
import QueryInput from './components/QueryInput.tsx';
import ResultsDisplay from './components/ResultsDisplay.tsx';
import LoadingSpinner from './components/LoadingSpinner.tsx';
import { apiService } from './services/api.ts';
import { QueryResponse, CleanQueryResponse, LoadingState } from './types/index.ts';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<QueryResponse | CleanQueryResponse | null>(null);
  const [loading, setLoading] = useState<LoadingState>({ isLoading: false });
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  // Check API status on mount
  useEffect(() => {
    checkApiStatus();
  }, []);

  const checkApiStatus = async () => {
    try {
      setApiStatus('checking');
      const isOnline = await apiService.testConnection();
      setApiStatus(isOnline ? 'online' : 'offline');
    } catch (error) {
      setApiStatus('offline');
    }
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
  };

  const handleQuerySubmit = async () => {
    if (!selectedFile || !query.trim()) return;

    setLoading({ isLoading: true, message: 'Processing your request...', progress: 0 });
    setError(null);
    setResult(null);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setLoading(prev => ({
          ...prev,
          progress: Math.min((prev.progress || 0) + Math.random() * 20, 90)
        }));
      }, 500);

      // Call API
      const response = await apiService.queryDocumentClean(selectedFile, query);
      
      clearInterval(progressInterval);
      setLoading({ isLoading: true, progress: 100 });
      
      // Small delay to show 100% completion
      setTimeout(() => {
        setResult(response);
        setLoading({ isLoading: false });
      }, 500);

    } catch (err: any) {
      setLoading({ isLoading: false });
      setError(err.detail || 'An error occurred while processing your request.');
    }
  };

  const resetForm = () => {
    setSelectedFile(null);
    setQuery('');
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-primary-600 rounded-lg">
                <FileText className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">
                  Document Analyzer
                </h1>
                <p className="text-sm text-gray-500">
                  AI-powered document analysis
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* API Status */}
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  apiStatus === 'online' ? 'bg-green-500' : 
                  apiStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
                }`} />
                <span className="text-sm text-gray-600">
                  {apiStatus === 'online' ? 'Online' : 
                   apiStatus === 'offline' ? 'Offline' : 'Checking...'}
                </span>
              </div>
              
              {/* GitHub Link */}
              <a
                href="https://github.com/sriram687/DOC_ANALYZE"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 text-gray-500 hover:text-gray-700 transition-colors"
                title="View on GitHub"
              >
                <Github className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        {!selectedFile && !result && (
          <div className="text-center mb-12 animate-fade-in">
            <div className="mb-6">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-primary-600 to-primary-700 rounded-full mb-4">
                <Zap className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 mb-4">
                Analyze Any Document with AI
              </h2>
              <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                Upload your documents and ask questions in natural language. 
                Get instant, accurate answers powered by advanced AI technology.
              </p>
            </div>
            
            {/* Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-200">
                <FileText className="w-8 h-8 text-primary-600 mx-auto mb-3" />
                <h3 className="font-semibold text-gray-900 mb-2">Multiple Formats</h3>
                <p className="text-sm text-gray-600">
                  PDF, DOCX, TXT, EML, and image files supported
                </p>
              </div>
              <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-200">
                <Shield className="w-8 h-8 text-primary-600 mx-auto mb-3" />
                <h3 className="font-semibold text-gray-900 mb-2">Secure & Private</h3>
                <p className="text-sm text-gray-600">
                  Your documents are processed securely and not stored
                </p>
              </div>
              <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-200">
                <Globe className="w-8 h-8 text-primary-600 mx-auto mb-3" />
                <h3 className="font-semibold text-gray-900 mb-2">Fast & Accurate</h3>
                <p className="text-sm text-gray-600">
                  Get precise answers in seconds using advanced AI
                </p>
              </div>
            </div>
          </div>
        )}

        {/* File Upload */}
        <div className="mb-8">
          <FileUpload
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            isLoading={loading.isLoading}
          />
        </div>

        {/* Query Input */}
        {selectedFile && (
          <div className="mb-8 animate-slide-up">
            <QueryInput
              query={query}
              onQueryChange={setQuery}
              onSubmit={handleQuerySubmit}
              isLoading={loading.isLoading}
              disabled={apiStatus !== 'online'}
            />
          </div>
        )}

        {/* Loading State */}
        {loading.isLoading && (
          <div className="mb-8">
            <LoadingSpinner
              message={loading.message}
              progress={loading.progress}
              stage="analyzing"
            />
          </div>
        )}

        {/* Results */}
        <ResultsDisplay
          result={result}
          isLoading={loading.isLoading}
          error={error}
        />

        {/* Reset Button */}
        {(result || error) && (
          <div className="text-center mt-8">
            <button
              onClick={resetForm}
              className="btn-secondary"
            >
              Analyze Another Document
            </button>
          </div>
        )}

        {/* API Offline Message */}
        {apiStatus === 'offline' && (
          <div className="card border-yellow-200 bg-yellow-50 mt-8">
            <div className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
              <div>
                <h3 className="font-semibold text-yellow-800">API Currently Offline</h3>
                <p className="text-yellow-700 text-sm mt-1">
                  The backend service is currently unavailable. Please try again later or check the 
                  <a 
                    href="https://github.com/sriram687/DOC_ANALYZE" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="inline-flex items-center ml-1 text-yellow-800 hover:text-yellow-900 underline"
                  >
                    deployment status
                    <ExternalLink className="w-3 h-3 ml-1" />
                  </a>
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-sm text-gray-500">
            <p>
              Built with ❤️ using React, FastAPI, and AI • 
              <a 
                href="https://github.com/sriram687/DOC_ANALYZE" 
                target="_blank" 
                rel="noopener noreferrer"
                className="ml-1 text-primary-600 hover:text-primary-700 underline"
              >
                View Source Code
              </a>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
