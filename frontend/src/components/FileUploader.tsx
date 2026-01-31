'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, Database, GitBranch, Activity } from 'lucide-react';

type DemoDataset = 'paul15' | 'simulation_decision' | 'simulation_contribution';

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
  onDemoSelect: (dataset: DemoDataset) => void;
  isLoading: boolean;
}

const DEMO_DATASETS: { id: DemoDataset; label: string; desc: string; icon: typeof Database }[] = [
  { id: 'paul15', label: 'Paul15 (Hematopoiesis)', desc: 'Mouse bone marrow cells', icon: Database },
  { id: 'simulation_decision', label: 'Decision Simulation', desc: 'Simulated fate decision data', icon: GitBranch },
  { id: 'simulation_contribution', label: 'Contribution Simulation', desc: 'Simulated gene contribution data', icon: Activity },
];

export function FileUploader({ onFileSelect, onDemoSelect, isLoading }: FileUploaderProps) {
  const [loadingDemo, setLoadingDemo] = useState<DemoDataset | null>(null);
  
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

  const handleDemoSelect = async (dataset: DemoDataset) => {
    if (isLoading || loadingDemo) return;
    setLoadingDemo(dataset);
    try {
      await onDemoSelect(dataset);
    } finally {
      setLoadingDemo(null);
    }
  };

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
        <span className="text-sm text-gray-400">or load a demo dataset</span>
        <div className="flex-1 h-px bg-gray-200" />
      </div>

      <div className="space-y-2">
        <p className="text-sm font-medium text-gray-700">Demo Datasets</p>
        <div className="grid grid-cols-1 gap-2">
          {DEMO_DATASETS.map(({ id, label, desc, icon: Icon }) => (
            <button
              key={id}
              onClick={() => handleDemoSelect(id)}
              disabled={isLoading || loadingDemo !== null}
              className={`w-full p-3 border rounded-lg text-left transition-all hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-3
                ${loadingDemo === id ? 'border-primary-500 bg-primary-50' : 'border-gray-200 hover:border-primary-400'}`}
            >
              {loadingDemo === id ? (
                <div className="w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
              ) : (
                <Icon className="w-5 h-5 text-primary-500" />
              )}
              <div>
                <span className="text-sm font-medium text-gray-800">{label}</span>
                <p className="text-xs text-gray-500">{desc}</p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
