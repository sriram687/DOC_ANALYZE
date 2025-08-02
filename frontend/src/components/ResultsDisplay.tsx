import React from 'react';
import { CheckCircle, AlertTriangle, Clock, Zap, Copy, Download } from 'lucide-react';
import { QueryResponse, CleanQueryResponse } from '../types';

interface ResultsDisplayProps {
  result: QueryResponse | CleanQueryResponse | null;
  isLoading: boolean;
  error: string | null;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result, isLoading, error }) => {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // You could add a toast notification here
  };

  const downloadResult = () => {
    if (!result) return;
    
    const content = `
Query: ${result.query}

Answer: ${result.answer}

${result.conditions.length > 0 ? `Conditions:\n${result.conditions.map(c => `• ${c}`).join('\n')}` : ''}

Confidence: ${(result.confidence * 100).toFixed(1)}%
Processing Time: ${result.processing_time.toFixed(2)}s
Generated: ${new Date().toLocaleString()}
    `.trim();
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `document-analysis-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-100';
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High Confidence';
    if (confidence >= 0.6) return 'Medium Confidence';
    return 'Low Confidence';
  };

  if (isLoading) {
    return (
      <div className="card animate-pulse">
        <div className="flex items-center justify-center space-x-3 py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
          <div className="text-lg font-medium text-gray-700">
            <span className="loading-dots">Analyzing your document</span>
          </div>
        </div>
        <div className="space-y-3 mt-6">
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          <div className="h-4 bg-gray-200 rounded w-5/6"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card border-red-200 bg-red-50 animate-slide-up">
        <div className="flex items-start space-x-3">
          <AlertTriangle className="w-6 h-6 text-red-500 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-red-800 mb-2">
              Analysis Failed
            </h3>
            <p className="text-red-700 mb-4">{error}</p>
            <div className="text-sm text-red-600">
              <p className="font-medium mb-1">Possible solutions:</p>
              <ul className="list-disc list-inside space-y-1">
                <li>Check if your document is readable and not corrupted</li>
                <li>Ensure the file size is under 50MB</li>
                <li>Try uploading a different document format</li>
                <li>Check your internet connection</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Header with Actions */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-gray-900">Analysis Results</h2>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => copyToClipboard(result.answer)}
            className="btn-secondary text-sm"
            title="Copy answer"
          >
            <Copy className="w-4 h-4 mr-1" />
            Copy
          </button>
          <button
            onClick={downloadResult}
            className="btn-secondary text-sm"
            title="Download results"
          >
            <Download className="w-4 h-4 mr-1" />
            Download
          </button>
        </div>
      </div>

      {/* Query */}
      <div className="card bg-gray-50 border-gray-200">
        <div className="flex items-start space-x-3">
          <div className="p-2 bg-primary-100 rounded-lg">
            <Zap className="w-5 h-5 text-primary-600" />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-gray-900 mb-2">Your Question</h3>
            <p className="text-gray-700 italic">"{result.query}"</p>
          </div>
        </div>
      </div>

      {/* Answer */}
      <div className="card">
        <div className="flex items-start space-x-3">
          <div className="p-2 bg-green-100 rounded-lg">
            <CheckCircle className="w-5 h-5 text-green-600" />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-gray-900 mb-3">Answer</h3>
            <div className="prose prose-sm max-w-none">
              <div 
                className="text-gray-700 leading-relaxed whitespace-pre-wrap"
                dangerouslySetInnerHTML={{ 
                  __html: result.answer.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') 
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Conditions */}
      {result.conditions && result.conditions.length > 0 && (
        <div className="card border-yellow-200 bg-yellow-50">
          <div className="flex items-start space-x-3">
            <div className="p-2 bg-yellow-100 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-yellow-600" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-yellow-800 mb-3">Important Conditions</h3>
              <ul className="space-y-2">
                {result.conditions.map((condition, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <span className="w-1.5 h-1.5 bg-yellow-500 rounded-full mt-2 flex-shrink-0"></span>
                    <span className="text-yellow-700">{condition}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Evidence (if available) */}
      {'evidence' in result && result.evidence && result.evidence.length > 0 && (
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-4">Supporting Evidence</h3>
          <div className="space-y-3">
            {result.evidence.slice(0, 3).map((evidence, index) => (
              <div key={index} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-600">
                    Source {index + 1}
                  </span>
                  <span className="text-xs text-gray-500">{evidence.relevance}</span>
                </div>
                <p className="text-sm text-gray-700 line-clamp-3">{evidence.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Metadata */}
      <div className="card bg-gray-50 border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(result.confidence)}`}>
                {getConfidenceLabel(result.confidence)}
              </div>
              <span className="text-sm text-gray-600">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <Clock className="w-4 h-4" />
              <span>{result.processing_time.toFixed(2)}s</span>
            </div>
          </div>
          <div className="text-xs text-gray-500">
            Generated at {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;
