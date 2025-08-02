import React, { useState, useRef, useEffect } from 'react';
import { Send, Mic, MicOff, Sparkles } from 'lucide-react';

interface QueryInputProps {
  query: string;
  onQueryChange: (query: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  disabled: boolean;
}

const QueryInput: React.FC<QueryInputProps> = ({
  query,
  onQueryChange,
  onSubmit,
  isLoading,
  disabled
}) => {
  const [isListening, setIsListening] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Sample questions for inspiration
  const sampleQuestions = [
    "What is this document about?",
    "What are the key terms and conditions?",
    "What are the coverage details?",
    "What are the exclusions mentioned?",
    "What is the claim process?",
    "What are the premium details?",
  ];

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [query]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading && !disabled) {
      onSubmit();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleSampleQuestion = (question: string) => {
    onQueryChange(question);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  // Voice input (if supported)
  const toggleVoiceInput = () => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      if (!isListening) {
        setIsListening(true);
        recognition.start();

        recognition.onresult = (event: any) => {
          const transcript = event.results[0][0].transcript;
          onQueryChange(query + ' ' + transcript);
          setIsListening(false);
        };

        recognition.onerror = () => {
          setIsListening(false);
        };

        recognition.onend = () => {
          setIsListening(false);
        };
      } else {
        recognition.stop();
        setIsListening(false);
      }
    }
  };

  const isVoiceSupported = 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;

  return (
    <div className="w-full space-y-4">
      {/* Sample Questions */}
      {!query && (
        <div className="animate-fade-in">
          <div className="flex items-center space-x-2 mb-3">
            <Sparkles className="w-4 h-4 text-primary-500" />
            <span className="text-sm font-medium text-gray-700">Try asking:</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {sampleQuestions.map((question, index) => (
              <button
                key={index}
                onClick={() => handleSampleQuestion(question)}
                disabled={disabled || isLoading}
                className="text-left p-3 text-sm text-gray-600 bg-gray-50 hover:bg-primary-50 hover:text-primary-700 rounded-lg border border-gray-200 hover:border-primary-200 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                "{question}"
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Query Input Form */}
      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask any question about your document..."
            disabled={disabled || isLoading}
            className="input-field resize-none min-h-[60px] max-h-[200px] pr-20"
            rows={1}
          />
          
          {/* Voice Input Button */}
          {isVoiceSupported && (
            <button
              type="button"
              onClick={toggleVoiceInput}
              disabled={disabled || isLoading}
              className={`absolute right-12 top-1/2 transform -translate-y-1/2 p-2 rounded-full transition-all duration-200 ${
                isListening
                  ? 'bg-red-100 text-red-600 hover:bg-red-200'
                  : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
              title={isListening ? 'Stop listening' : 'Voice input'}
            >
              {isListening ? (
                <MicOff className="w-4 h-4" />
              ) : (
                <Mic className="w-4 h-4" />
              )}
            </button>
          )}
          
          {/* Submit Button */}
          <button
            type="submit"
            disabled={!query.trim() || disabled || isLoading}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 p-2 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-300 text-white rounded-full transition-all duration-200 disabled:cursor-not-allowed"
            title="Send query"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        
        {/* Character Count */}
        <div className="flex justify-between items-center text-xs text-gray-500">
          <span>
            {isListening && (
              <span className="flex items-center space-x-1 text-red-600">
                <div className="w-2 h-2 bg-red-600 rounded-full animate-pulse"></div>
                <span>Listening...</span>
              </span>
            )}
          </span>
          <span>{query.length}/1000</span>
        </div>
      </form>
    </div>
  );
};

export default QueryInput;
