import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X, CheckCircle, AlertCircle } from 'lucide-react';
import { UploadedFile } from '../types';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  isLoading: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, selectedFile, isLoading }) => {
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const uploadFile: UploadedFile = {
        file,
        status: 'pending',
        preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined,
      };
      
      setUploadedFile(uploadFile);
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt'],
      'message/rfc822': ['.eml'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
    },
    multiple: false,
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  const removeFile = () => {
    setUploadedFile(null);
    onFileSelect(null as any);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (file: File) => {
    if (file.type.includes('pdf')) return '📄';
    if (file.type.includes('word') || file.name.endsWith('.docx')) return '📝';
    if (file.type.includes('image')) return '🖼️';
    if (file.type.includes('text')) return '📃';
    return '📁';
  };

  return (
    <div className="w-full">
      {!uploadedFile ? (
        <div
          {...getRootProps()}
          className={`upload-zone ${isDragActive ? 'active' : ''} ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <input {...getInputProps()} disabled={isLoading} />
          <div className="flex flex-col items-center space-y-4">
            <div className="p-4 bg-primary-100 rounded-full">
              <Upload className="w-8 h-8 text-primary-600" />
            </div>
            <div className="text-center">
              <p className="text-lg font-medium text-gray-700 mb-2">
                {isDragActive ? 'Drop your document here' : 'Upload your document'}
              </p>
              <p className="text-sm text-gray-500 mb-4">
                Drag and drop or click to browse
              </p>
              <div className="flex flex-wrap justify-center gap-2 text-xs text-gray-400">
                <span className="bg-gray-100 px-2 py-1 rounded">PDF</span>
                <span className="bg-gray-100 px-2 py-1 rounded">DOCX</span>
                <span className="bg-gray-100 px-2 py-1 rounded">TXT</span>
                <span className="bg-gray-100 px-2 py-1 rounded">EML</span>
                <span className="bg-gray-100 px-2 py-1 rounded">PNG</span>
                <span className="bg-gray-100 px-2 py-1 rounded">JPG</span>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="card">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="text-2xl">{getFileIcon(uploadedFile.file)}</div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">
                  {uploadedFile.file.name}
                </p>
                <p className="text-sm text-gray-500">
                  {formatFileSize(uploadedFile.file.size)}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              {uploadedFile.status === 'success' && (
                <CheckCircle className="w-5 h-5 text-green-500" />
              )}
              {uploadedFile.status === 'error' && (
                <AlertCircle className="w-5 h-5 text-red-500" />
              )}
              {!isLoading && (
                <button
                  onClick={removeFile}
                  className="p-1 hover:bg-gray-100 rounded-full transition-colors"
                  title="Remove file"
                >
                  <X className="w-4 h-4 text-gray-400 hover:text-gray-600" />
                </button>
              )}
            </div>
          </div>
          
          {uploadedFile.preview && (
            <div className="mt-4">
              <img
                src={uploadedFile.preview}
                alt="Preview"
                className="max-w-full h-32 object-cover rounded-lg border"
              />
            </div>
          )}
          
          {uploadedFile.error && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-600">{uploadedFile.error}</p>
            </div>
          )}
        </div>
      )}
      
      <div className="mt-4 text-center">
        <p className="text-xs text-gray-500">
          Maximum file size: 50MB • Supported formats: PDF, DOCX, TXT, EML, PNG, JPG
        </p>
      </div>
    </div>
  );
};

export default FileUpload;
