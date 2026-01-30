'use client';

import React, { useState, useEffect } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { FileUploader } from '@/components/FileUploader';
import { ScatterPlot } from '@/components/ScatterPlot';
import { PipelineSteps } from '@/components/PipelineSteps';
import { dataApi, analysisApi, trainingApi, resultsApi, SessionStatus, EmbeddingData, GridData } from '@/lib/api';
import { Loader2, Play, Settings, BarChart3, GitBranch, Dna, AlertCircle, CheckCircle } from 'lucide-react';

const PIPELINE_STEPS = [
  { id: 'upload', name: 'Upload Data', description: 'Upload single-cell data file' },
  { id: 'preprocess', name: 'Preprocess', description: 'Normalize, transform, and cluster' },
  { id: 'grid', name: 'Generate Grid', description: 'Create grid embedding' },
  { id: 'pseudotime', name: 'Align Pseudotime', description: 'Calculate developmental time' },
  { id: 'rewards', name: 'Generate Rewards', description: 'Set up RL reward system' },
  { id: 'train', name: 'Train Model', description: 'Train the RL agent' },
];

export default function HomePage() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState('upload');
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Configuration state
  const [config, setConfig] = useState({
    embedding_method: 'tsne' as 'tsne' | 'umap',
    n_pcs: 50,
    clustering_resolution: 1.0,
    grid_n: 50,
    grid_j: 3,
    start_cluster: '',
    reward_type: 'discrete' as 'discrete' | 'continuous',
    reward_mode: 'Decision' as 'Decision' | 'Contribution',
    starts: [] as string[],
    ends: [] as string[],
    reward_genes: [] as string[],
    num_episodes: 5000,
    gamma: 0.9,
  });
  
  // Data state
  const [dataInfo, setDataInfo] = useState<{ n_cells: number; n_genes: number } | null>(null);
  const [clusters, setClusters] = useState<string[]>([]);
  const [embeddingData, setEmbeddingData] = useState<EmbeddingData | null>(null);
  const [gridData, setGridData] = useState<GridData | null>(null);
  const [trainingResult, setTrainingResult] = useState<any>(null);
  const [colorBy, setColorBy] = useState<'cluster' | 'pseudotime' | 'value'>('cluster');

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: (file: File) => dataApi.uploadFile(file),
    onSuccess: (data) => {
      setSessionId(data.session_id);
      setDataInfo({ n_cells: data.n_cells, n_genes: data.n_genes });
      completeStep('upload');
      setError(null);
    },
    onError: (err: any) => setError(err.message || 'Upload failed'),
  });

  // Demo data mutation
  const demoMutation = useMutation({
    mutationFn: () => dataApi.loadDemo('paul15'),
    onSuccess: (data) => {
      setSessionId(data.session_id);
      setDataInfo({ n_cells: data.n_cells, n_genes: data.n_genes });
      completeStep('upload');
      setError(null);
    },
    onError: (err: any) => setError(err.message || 'Failed to load demo'),
  });

  // Preprocess mutation
  const preprocessMutation = useMutation({
    mutationFn: () => dataApi.preprocess(sessionId!, {
      normalize: true,
      log_transform: true,
      scale: true,
      n_pcs: config.n_pcs,
      embedding_method: config.embedding_method,
      clustering_resolution: config.clustering_resolution,
    }),
    onSuccess: async () => {
      completeStep('preprocess');
      // Get clusters
      const { clusters: clusterList } = await dataApi.getClusters(sessionId!);
      setClusters(clusterList);
      if (clusterList.length > 0) {
        setConfig(prev => ({ ...prev, start_cluster: clusterList[0] }));
      }
      // Get embedding
      const embedding = await resultsApi.getEmbedding(sessionId!);
      setEmbeddingData(embedding);
      setError(null);
    },
    onError: (err: any) => setError(err.message || 'Preprocessing failed'),
  });

  // Grid generation mutation
  const gridMutation = useMutation({
    mutationFn: () => analysisApi.generateGrid(sessionId!, {
      n: config.grid_n,
      j: config.grid_j,
      n_jobs: 8,
    }),
    onSuccess: async () => {
      completeStep('grid');
      const grid = await resultsApi.getGrid(sessionId!);
      setGridData(grid);
      setError(null);
    },
    onError: (err: any) => setError(err.message || 'Grid generation failed'),
  });

  // Pseudotime mutation
  const pseudotimeMutation = useMutation({
    mutationFn: () => analysisApi.alignPseudotime(sessionId!, {
      start_cluster: config.start_cluster,
      n_sample_cells: 10,
      boundary: true,
    }),
    onSuccess: async () => {
      completeStep('pseudotime');
      const embedding = await resultsApi.getEmbedding(sessionId!);
      setEmbeddingData(embedding);
      setError(null);
    },
    onError: (err: any) => setError(err.message || 'Pseudotime alignment failed'),
  });

  // Rewards mutation
  const rewardsMutation = useMutation({
    mutationFn: async () => {
      if (config.reward_type === 'discrete') {
        return analysisApi.generateDiscreteRewards(sessionId!, {
          starts: config.starts,
          ends: config.ends,
          beta: 1.0,
          mode: config.reward_mode,
        });
      } else {
        return analysisApi.generateContinuousRewards(sessionId!, {
          reward_keys: config.reward_genes,
          starts: config.starts.length > 0 ? config.starts : undefined,
          beta: 1.0,
          mode: config.reward_mode,
        });
      }
    },
    onSuccess: () => {
      completeStep('rewards');
      setError(null);
    },
    onError: (err: any) => setError(err.message || 'Reward generation failed'),
  });

  // Training mutation
  const trainingMutation = useMutation({
    mutationFn: () => trainingApi.train(sessionId!, {
      algo: 'ActorCritic',
      reward_type: config.reward_type === 'discrete' ? 'd' : 'c',
      reward_mode: config.reward_mode,
      num_episodes: config.num_episodes,
      max_step: 50,
      gamma: config.gamma,
      hidden_dim: 128,
    }),
    onSuccess: async (data) => {
      completeStep('train');
      setTrainingResult(data);
      // Get updated embedding with state values
      const embedding = await resultsApi.getEmbedding(sessionId!);
      setEmbeddingData(embedding);
      setColorBy('value');
      setError(null);
    },
    onError: (err: any) => setError(err.message || 'Training failed'),
  });

  const completeStep = (stepId: string) => {
    setCompletedSteps(prev => [...prev, stepId]);
    const stepIndex = PIPELINE_STEPS.findIndex(s => s.id === stepId);
    if (stepIndex < PIPELINE_STEPS.length - 1) {
      setCurrentStep(PIPELINE_STEPS[stepIndex + 1].id);
    }
  };

  const isLoading = uploadMutation.isPending || demoMutation.isPending || 
    preprocessMutation.isPending || gridMutation.isPending || 
    pseudotimeMutation.isPending || rewardsMutation.isPending || 
    trainingMutation.isPending;

  const renderStepContent = () => {
    switch (currentStep) {
      case 'upload':
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">Step 1: Upload Your Data</h2>
            <p className="text-gray-600">
              Upload your single-cell RNA sequencing data file or use a demo dataset.
            </p>
            <FileUploader
              onFileSelect={(file) => uploadMutation.mutate(file)}
              onDemoSelect={() => demoMutation.mutate()}
              isLoading={isLoading}
            />
            {dataInfo && (
              <div className="p-4 bg-green-50 rounded-lg">
                <p className="text-green-800">
                  ✓ Loaded {dataInfo.n_cells.toLocaleString()} cells × {dataInfo.n_genes.toLocaleString()} genes
                </p>
              </div>
            )}
          </div>
        );
      
      case 'preprocess':
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">Step 2: Preprocessing</h2>
            <p className="text-gray-600">
              Configure preprocessing parameters for your data.
            </p>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">Embedding Method</label>
                <select
                  className="input"
                  value={config.embedding_method}
                  onChange={(e) => setConfig(prev => ({ ...prev, embedding_method: e.target.value as 'tsne' | 'umap' }))}
                >
                  <option value="tsne">t-SNE</option>
                  <option value="umap">UMAP</option>
                </select>
              </div>
              <div>
                <label className="label">Number of PCs</label>
                <input
                  type="number"
                  className="input"
                  value={config.n_pcs}
                  onChange={(e) => setConfig(prev => ({ ...prev, n_pcs: parseInt(e.target.value) }))}
                  min={10}
                  max={100}
                />
              </div>
              <div>
                <label className="label">Clustering Resolution</label>
                <input
                  type="number"
                  className="input"
                  value={config.clustering_resolution}
                  onChange={(e) => setConfig(prev => ({ ...prev, clustering_resolution: parseFloat(e.target.value) }))}
                  min={0.1}
                  max={3.0}
                  step={0.1}
                />
              </div>
            </div>
            <button
              onClick={() => preprocessMutation.mutate()}
              disabled={isLoading}
              className="btn btn-primary w-full"
            >
              {preprocessMutation.isPending ? <Loader2 className="animate-spin mr-2" /> : <Play className="mr-2 h-4 w-4" />}
              Run Preprocessing
            </button>
          </div>
        );
      
      case 'grid':
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">Step 3: Generate Grid Embedding</h2>
            <p className="text-gray-600">
              Transform the 2D embedding into a structured grid for RL navigation.
            </p>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">Grid Resolution (n)</label>
                <input
                  type="number"
                  className="input"
                  value={config.grid_n}
                  onChange={(e) => setConfig(prev => ({ ...prev, grid_n: parseInt(e.target.value) }))}
                  min={20}
                  max={100}
                />
              </div>
              <div>
                <label className="label">Observer Number (j)</label>
                <input
                  type="number"
                  className="input"
                  value={config.grid_j}
                  onChange={(e) => setConfig(prev => ({ ...prev, grid_j: parseInt(e.target.value) }))}
                  min={1}
                  max={10}
                />
              </div>
            </div>
            <button
              onClick={() => gridMutation.mutate()}
              disabled={isLoading}
              className="btn btn-primary w-full"
            >
              {gridMutation.isPending ? <Loader2 className="animate-spin mr-2" /> : <GitBranch className="mr-2 h-4 w-4" />}
              Generate Grid
            </button>
          </div>
        );
      
      case 'pseudotime':
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">Step 4: Align Pseudotime</h2>
            <p className="text-gray-600">
              Select the starting cluster (typically stem/progenitor cells) for pseudotime calculation.
            </p>
            <div>
              <label className="label">Starting Cluster</label>
              <select
                className="input"
                value={config.start_cluster}
                onChange={(e) => setConfig(prev => ({ ...prev, start_cluster: e.target.value }))}
              >
                {clusters.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
            <button
              onClick={() => pseudotimeMutation.mutate()}
              disabled={isLoading}
              className="btn btn-primary w-full"
            >
              {pseudotimeMutation.isPending ? <Loader2 className="animate-spin mr-2" /> : <BarChart3 className="mr-2 h-4 w-4" />}
              Align Pseudotime
            </button>
          </div>
        );
      
      case 'rewards':
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">Step 5: Configure Rewards</h2>
            <p className="text-gray-600">
              Set up the reward system for RL training.
            </p>
            <div className="space-y-4">
              <div>
                <label className="label">Reward Type</label>
                <select
                  className="input"
                  value={config.reward_type}
                  onChange={(e) => setConfig(prev => ({ ...prev, reward_type: e.target.value as 'discrete' | 'continuous' }))}
                >
                  <option value="discrete">Discrete (Lineage-based)</option>
                  <option value="continuous">Continuous (Gene-based)</option>
                </select>
              </div>
              <div>
                <label className="label">Reward Mode</label>
                <select
                  className="input"
                  value={config.reward_mode}
                  onChange={(e) => setConfig(prev => ({ ...prev, reward_mode: e.target.value as 'Decision' | 'Contribution' }))}
                >
                  <option value="Decision">Decision (Early fate points)</option>
                  <option value="Contribution">Contribution (Late fate points)</option>
                </select>
              </div>
              {config.reward_type === 'discrete' && (
                <>
                  <div>
                    <label className="label">Starting Clusters</label>
                    <select
                      className="input"
                      multiple
                      value={config.starts}
                      onChange={(e) => {
                        const values = Array.from(e.target.selectedOptions, option => option.value);
                        setConfig(prev => ({ ...prev, starts: values }));
                      }}
                    >
                      {clusters.map(c => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="label">Target Clusters</label>
                    <select
                      className="input"
                      multiple
                      value={config.ends}
                      onChange={(e) => {
                        const values = Array.from(e.target.selectedOptions, option => option.value);
                        setConfig(prev => ({ ...prev, ends: values }));
                      }}
                    >
                      {clusters.map(c => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                </>
              )}
            </div>
            <button
              onClick={() => rewardsMutation.mutate()}
              disabled={isLoading || (config.reward_type === 'discrete' && (config.starts.length === 0 || config.ends.length === 0))}
              className="btn btn-primary w-full"
            >
              {rewardsMutation.isPending ? <Loader2 className="animate-spin mr-2" /> : <Dna className="mr-2 h-4 w-4" />}
              Generate Rewards
            </button>
          </div>
        );
      
      case 'train':
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">Step 6: Train RL Model</h2>
            <p className="text-gray-600">
              Train the Actor-Critic model to learn cellular fate decisions.
            </p>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">Number of Episodes</label>
                <input
                  type="number"
                  className="input"
                  value={config.num_episodes}
                  onChange={(e) => setConfig(prev => ({ ...prev, num_episodes: parseInt(e.target.value) }))}
                  min={100}
                  max={50000}
                  step={100}
                />
              </div>
              <div>
                <label className="label">Discount Factor (γ)</label>
                <input
                  type="number"
                  className="input"
                  value={config.gamma}
                  onChange={(e) => setConfig(prev => ({ ...prev, gamma: parseFloat(e.target.value) }))}
                  min={0.5}
                  max={0.99}
                  step={0.01}
                />
              </div>
            </div>
            <button
              onClick={() => trainingMutation.mutate()}
              disabled={isLoading}
              className="btn btn-primary w-full"
            >
              {trainingMutation.isPending ? <Loader2 className="animate-spin mr-2" /> : <Play className="mr-2 h-4 w-4" />}
              Start Training
            </button>
            {trainingResult && (
              <div className="p-4 bg-green-50 rounded-lg">
                <p className="text-green-800">
                  ✓ Training completed in {trainingResult.training_time.toFixed(1)}s
                </p>
                <p className="text-green-600 text-sm">
                  Final reward: {trainingResult.final_reward.toFixed(3)}
                </p>
              </div>
            )}
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-accent-500 rounded-lg flex items-center justify-center">
                <Dna className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">scRL-Web</h1>
                <p className="text-xs text-gray-500">Single-cell Reinforcement Learning</p>
              </div>
            </div>
            <div className="text-sm text-gray-500">
              {sessionId && <span>Session: {sessionId.slice(0, 8)}...</span>}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-12 gap-6">
          {/* Left Sidebar - Pipeline Steps */}
          <div className="col-span-3">
            <div className="card sticky top-8">
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-4">
                Analysis Pipeline
              </h3>
              <PipelineSteps
                steps={PIPELINE_STEPS}
                currentStep={currentStep}
                completedSteps={completedSteps}
                isProcessing={isLoading}
              />
            </div>
          </div>

          {/* Center - Step Content */}
          <div className="col-span-5">
            <div className="card">
              {error && (
                <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-red-800 font-medium">Error</p>
                    <p className="text-red-600 text-sm">{error}</p>
                  </div>
                </div>
              )}
              {renderStepContent()}
            </div>
          </div>

          {/* Right - Visualization */}
          <div className="col-span-4">
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider">
                  Visualization
                </h3>
                {embeddingData && (
                  <select
                    className="text-sm border rounded px-2 py-1"
                    value={colorBy}
                    onChange={(e) => setColorBy(e.target.value as 'cluster' | 'pseudotime' | 'value')}
                  >
                    <option value="cluster">Color by Cluster</option>
                    {embeddingData.pseudotime.length > 0 && (
                      <option value="pseudotime">Color by Pseudotime</option>
                    )}
                    {embeddingData.state_values && (
                      <option value="value">Color by Fate Value</option>
                    )}
                  </select>
                )}
              </div>
              
              {embeddingData ? (
                <ScatterPlot
                  data={{
                    x: embeddingData.embedding.map(p => p[0]),
                    y: embeddingData.embedding.map(p => p[1]),
                    colors: embeddingData.cluster_colors,
                    labels: embeddingData.clusters,
                    values: colorBy === 'pseudotime' 
                      ? embeddingData.pseudotime 
                      : colorBy === 'value' 
                        ? embeddingData.state_values || undefined
                        : undefined,
                  }}
                  colorBy={colorBy}
                  width={350}
                  height={350}
                  title={`Cell Embedding (${dataInfo?.n_cells.toLocaleString()} cells)`}
                />
              ) : (
                <div className="h-[350px] flex items-center justify-center bg-gray-50 rounded-lg">
                  <p className="text-gray-400">Upload data to see visualization</p>
                </div>
              )}

              {completedSteps.includes('train') && trainingResult && (
                <div className="mt-4 p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    <span className="font-medium text-green-800">Analysis Complete!</span>
                  </div>
                  <p className="text-sm text-green-700">
                    The model has identified fate decision strengths across {dataInfo?.n_cells.toLocaleString()} cells.
                    Higher values indicate stronger fate commitment points.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-white mt-8">
        <div className="max-w-7xl mx-auto px-4 py-4 text-center text-sm text-gray-500">
          <p>scRL-Web v0.1.0 | Based on scRL by Zeyu Fu | MIT License</p>
        </div>
      </footer>
    </div>
  );
}
