'use client';

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, Database } from 'lucide-react';

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
  onDemoSelect: () => void;
  isLoading: boolean;
}

export function FileUploader({ onFileSelect, onDemoSelect, isLoading }: FileUploaderProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/octet-stream': ['.h5ad'],
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
    disabled: isLoading,
  });

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
          transition-colors duration-200
          ${isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'}
          ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        {isDragActive ? (
          <p className="text-primary-600 font-medium">Drop the file here...</p>
        ) : (
          <>
            <p className="text-gray-600 mb-2">
              Drag & drop your single-cell data file here
            </p>
            <p className="text-sm text-gray-400">
              Supports .h5ad (AnnData) and .csv files
            </p>
          </>
        )}
      </div>

      <div className="flex items-center gap-4">
        <div className="flex-1 h-px bg-gray-200" />
        <span className="text-sm text-gray-400">or</span>
        <div className="flex-1 h-px bg-gray-200" />
      </div>

      <button
        onClick={onDemoSelect}
        disabled={isLoading}
        className="w-full btn btn-secondary flex items-center justify-center gap-2"
      >
        <Database className="h-4 w-4" />
        Load Demo Dataset (Paul15)
      </button>
    </div>
  );
}
