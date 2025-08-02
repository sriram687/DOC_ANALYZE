import axios, { AxiosResponse } from 'axios';
import { QueryResponse, CleanQueryResponse, HealthCheck, ApiError } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes timeout for document processing
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
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
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    
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
    const response: AxiosResponse<HealthCheck> = await api.get('/health');
    return response.data;
  },

  // Main document query endpoint
  async queryDocument(file: File, query: string): Promise<QueryResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('query', query);

    const response: AxiosResponse<QueryResponse> = await api.post('/ask-document', formData);
    return response.data;
  },

  // Clean format document query
  async queryDocumentClean(file: File, query: string): Promise<CleanQueryResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('query', query);

    const response: AxiosResponse<CleanQueryResponse> = await api.post('/ask-document-clean', formData);
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
