export interface QueryResponse {
  query: string;
  answer: string;
  conditions: string[];
  evidence?: Evidence[];
  confidence: number;
  processing_time: number;
}

export interface CleanQueryResponse {
  query: string;
  answer: string;
  conditions: string[];
  confidence: number;
  processing_time: number;
}

export interface Evidence {
  clause_id: string;
  text: string;
  relevance: string;
  metadata: Record<string, any>;
}

export interface UploadedFile {
  file: File;
  preview?: string;
  status: 'pending' | 'uploading' | 'success' | 'error';
  error?: string;
}

export interface ApiError {
  detail: string;
  status_code?: number;
}

export interface HealthCheck {
  status: string;
  model: string;
  timestamp: string;
}

export interface LoadingState {
  isLoading: boolean;
  message?: string;
  progress?: number;
}
