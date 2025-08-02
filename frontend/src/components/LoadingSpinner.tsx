import React from 'react';
import { Loader2, FileText, Brain, Sparkles } from 'lucide-react';

interface LoadingSpinnerProps {
  message?: string;
  progress?: number;
  stage?: 'uploading' | 'processing' | 'analyzing' | 'generating';
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  message = 'Processing...', 
  progress,
  stage = 'processing'
}) => {
  const getStageInfo = (currentStage: string) => {
    switch (currentStage) {
      case 'uploading':
        return {
          icon: FileText,
          color: 'text-blue-600',
          bgColor: 'bg-blue-100',
          message: 'Uploading document...',
        };
      case 'processing':
        return {
          icon: Loader2,
          color: 'text-primary-600',
          bgColor: 'bg-primary-100',
          message: 'Processing document...',
        };
      case 'analyzing':
        return {
          icon: Brain,
          color: 'text-purple-600',
          bgColor: 'bg-purple-100',
          message: 'Analyzing content...',
        };
      case 'generating':
        return {
          icon: Sparkles,
          color: 'text-green-600',
          bgColor: 'bg-green-100',
          message: 'Generating response...',
        };
      default:
        return {
          icon: Loader2,
          color: 'text-primary-600',
          bgColor: 'bg-primary-100',
          message: 'Processing...',
        };
    }
  };

  const stageInfo = getStageInfo(stage);
  const Icon = stageInfo.icon;

  return (
    <div className="flex flex-col items-center justify-center py-12 space-y-6">
      {/* Animated Icon */}
      <div className={`relative p-6 ${stageInfo.bgColor} rounded-full`}>
        <Icon 
          className={`w-12 h-12 ${stageInfo.color} ${
            stage === 'processing' || stage === 'uploading' ? 'animate-spin' : 'animate-pulse'
          }`} 
        />
        
        {/* Progress Ring */}
        {progress !== undefined && (
          <div className="absolute inset-0 flex items-center justify-center">
            <svg className="w-20 h-20 transform -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="40"
                stroke="currentColor"
                strokeWidth="4"
                fill="transparent"
                className="text-gray-200"
              />
              <circle
                cx="50"
                cy="50"
                r="40"
                stroke="currentColor"
                strokeWidth="4"
                fill="transparent"
                strokeDasharray={`${2 * Math.PI * 40}`}
                strokeDashoffset={`${2 * Math.PI * 40 * (1 - progress / 100)}`}
                className={stageInfo.color}
                style={{
                  transition: 'stroke-dashoffset 0.5s ease-in-out',
                }}
              />
            </svg>
          </div>
        )}
      </div>

      {/* Message */}
      <div className="text-center space-y-2">
        <h3 className="text-lg font-semibold text-gray-900">
          {message || stageInfo.message}
        </h3>
        
        {progress !== undefined && (
          <div className="flex items-center space-x-2">
            <div className="w-32 bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-500 ease-out ${
                  stageInfo.color.replace('text-', 'bg-')
                }`}
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="text-sm text-gray-600 font-medium">
              {progress}%
            </span>
          </div>
        )}
        
        <p className="text-sm text-gray-500 max-w-md">
          {stage === 'uploading' && 'Securely uploading your document to our servers...'}
          {stage === 'processing' && 'Extracting text and preparing for analysis...'}
          {stage === 'analyzing' && 'Using AI to understand your document content...'}
          {stage === 'generating' && 'Crafting a comprehensive response to your query...'}
        </p>
      </div>

      {/* Processing Steps */}
      <div className="flex items-center space-x-4 text-xs text-gray-400">
        <div className={`flex items-center space-x-1 ${
          ['uploading', 'processing', 'analyzing', 'generating'].includes(stage) ? 'text-primary-600' : ''
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            ['uploading', 'processing', 'analyzing', 'generating'].includes(stage) ? 'bg-primary-600' : 'bg-gray-300'
          }`} />
          <span>Upload</span>
        </div>
        
        <div className="w-4 h-px bg-gray-300" />
        
        <div className={`flex items-center space-x-1 ${
          ['processing', 'analyzing', 'generating'].includes(stage) ? 'text-primary-600' : ''
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            ['processing', 'analyzing', 'generating'].includes(stage) ? 'bg-primary-600' : 'bg-gray-300'
          }`} />
          <span>Process</span>
        </div>
        
        <div className="w-4 h-px bg-gray-300" />
        
        <div className={`flex items-center space-x-1 ${
          ['analyzing', 'generating'].includes(stage) ? 'text-primary-600' : ''
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            ['analyzing', 'generating'].includes(stage) ? 'bg-primary-600' : 'bg-gray-300'
          }`} />
          <span>Analyze</span>
        </div>
        
        <div className="w-4 h-px bg-gray-300" />
        
        <div className={`flex items-center space-x-1 ${
          stage === 'generating' ? 'text-primary-600' : ''
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            stage === 'generating' ? 'bg-primary-600' : 'bg-gray-300'
          }`} />
          <span>Generate</span>
        </div>
      </div>

      {/* Fun Facts */}
      <div className="text-center text-xs text-gray-400 max-w-md">
        <p className="italic">
          💡 Did you know? Our AI can process documents in multiple languages and understand complex formatting!
        </p>
      </div>
    </div>
  );
};

export default LoadingSpinner;
