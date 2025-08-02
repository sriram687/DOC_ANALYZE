import axios, { AxiosResponse } from 'axios';
import { QueryResponse, CleanQueryResponse, HealthCheck, ApiError } from '../types';

// API Configuration with fallbacks
const API_BASE_URL = process.env.REACT_APP_API_URL ||
                     (process.env.NODE_ENV === 'production'
                       ? 'https://doc-analyze-api.onrender.com'
                       : 'http://localhost:8000');

console.log('🔧 API Configuration:', {
  baseURL: API_BASE_URL,
  environment: process.env.NODE_ENV,
  version: process.env.REACT_APP_VERSION || '1.0.0',
  REACT_APP_API_URL: process.env.REACT_APP_API_URL,
  fullHealthURL: `${API_BASE_URL}/health`
});

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes timeout for document processing
  // Add retry configuration for production
  ...(process.env.NODE_ENV === 'production' && {
    retry: 3,
    retryDelay: 1000,
  }),
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    if (process.env.NODE_ENV === 'development' || process.env.REACT_APP_ENABLE_DEBUG === 'true') {
      console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    }
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    if (process.env.NODE_ENV === 'development' || process.env.REACT_APP_ENABLE_DEBUG === 'true') {
      console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    }
    return response;
  },
  (error) => {
    // Enhanced error logging for production debugging
    const errorInfo = {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      message: error.response?.data?.detail || error.message,
      timestamp: new Date().toISOString()
    };

    if (process.env.NODE_ENV === 'development' || process.env.REACT_APP_ENABLE_DEBUG === 'true') {
      console.error('❌ API Response Error:', errorInfo);
    }

    // Transform error for consistent handling
    const apiError: ApiError = {
      detail: error.response?.data?.detail || error.message || 'An unexpected error occurred',
      status_code: error.response?.status,
    };

    return Promise.reject(apiError);
  }
);

export const apiService = {
  // Health check
  async healthCheck(): Promise<HealthCheck> {
    console.log('🏥 Making health check request to:', `${API_BASE_URL}/health`);
    const response: AxiosResponse<HealthCheck> = await api.get('/health');
    console.log('✅ Health check successful:', response.data);
    return response.data;
  },

  // Main document query endpoint
  async queryDocument(file: File, query: string): Promise<QueryResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('query', query);

    const response: AxiosResponse<QueryResponse> = await api.post('/ask-document', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Clean format document query
  async queryDocumentClean(file: File, query: string): Promise<CleanQueryResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('query', query);

    const response: AxiosResponse<CleanQueryResponse> = await api.post('/ask-document-clean', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Test connection
  async testConnection(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch (error) {
      console.error('Connection test failed:', error);
      return false;
    }
  },
};

export default apiService;
