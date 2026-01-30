export const translations = {
  en: {
    // Header
    appName: 'scRL Web',
    appDescription: 'Single-cell Fate Decision Analysis Platform',
    session: 'Session',
    initializing: 'Initializing...',
    exportResults: 'Export Results',
    
    // Steps
    steps: {
      upload: 'Data Upload',
      preprocess: 'Preprocess',
      train: 'Train Model',
      results: 'Results',
    },
    
    // Upload
    upload: {
      title: 'Upload Single-cell Data',
      dragDrop: 'Drag and drop files here, or',
      clickSelect: 'click to select',
      supportedFormats: 'Supported formats: .h5ad, .csv',
      suggestion: 'Preprocessed AnnData files are recommended for optimal results.',
      uploadButton: 'Upload Data',
      uploading: 'Uploading...',
    },
    
    // Preprocess
    preprocess: {
      title: 'Preprocessing Parameters',
      minGenes: 'Min Genes per Cell',
      minCells: 'Min Cells per Gene',
      topGenes: 'Number of HVGs',
      nPcs: 'PCA Components',
      nNeighbors: 'Number of Neighbors',
      startButton: 'Run Preprocessing',
      processing: 'Processing...',
    },
    
    // Training
    train: {
      title: 'Training Parameters',
      hiddenDim: 'Hidden Dimension',
      nAgents: 'Number of Agents',
      maxSteps: 'Max Steps per Episode',
      numEpisodes: 'Training Episodes',
      learningRate: 'Learning Rate',
      gamma: 'Discount Factor',
      batchSize: 'Batch Size',
      memorySize: 'Memory Size',
      resolution: 'Cluster Resolution',
      runButton: 'Start Training',
      training: 'Training...',
    },
    
    // Data Info
    dataInfo: {
      title: 'Data Summary',
      cells: 'Cells',
      genes: 'Genes',
      hvg: 'HVGs',
      embeddings: 'Embeddings',
    },
    
    // Visualization
    visualization: {
      title: 'Embedding Visualization',
    },
    
    // Results
    results: {
      title: 'Analysis Results',
      summary: 'Results Summary',
      branches: 'Identified Branches',
      pseudotimeRange: 'Pseudotime Range',
      entropyRange: 'Entropy Range',
      analysisMode: 'Analysis Mode',
      actorCriticAlgorithm: 'Actor-Critic',
      simulatedMode: 'Simulation Mode',
      simulatedWarning: 'Running in simulation mode (scRL library not installed). Install scRL on the backend for actual analysis.',
      trainingProcess: 'Training Progress',
      trainingRewards: 'Training Rewards',
      trainingErrors: 'Training Errors',
    },
    
    // Welcome
    welcome: {
      title: 'scRL Analysis',
      description: 'Upload single-cell data to analyze cell fate decisions and compute pseudotime using the Actor-Critic reinforcement learning algorithm.',
      features: {
        actorCritic: 'Actor-Critic',
        multiAgent: 'Multi-Agent',
        noPrior: 'No Prior Required',
      },
    },
    
    // Status
    status: {
      starting: 'Starting preprocessing...',
      initializing: 'Initializing model...',
      complete: 'Complete',
      training: 'Training in progress...',
      episode: 'Episode',
    },
    
    // Language
    language: 'Language',
    chinese: '中文',
    english: 'English',
  },
  
  zh: {
    // Header
    appName: 'scRL Web',
    appDescription: '单细胞命运决策分析平台',
    session: '会话',
    initializing: '初始化中...',
    exportResults: '导出结果',
    
    // Steps
    steps: {
      upload: '数据上传',
      preprocess: '预处理',
      train: '训练模型',
      results: '结果',
    },
    
    // Upload
    upload: {
      title: '上传单细胞数据',
      dragDrop: '拖放文件到此处，或',
      clickSelect: '点击选择',
      supportedFormats: '支持格式: .h5ad, .csv',
      suggestion: '建议使用预处理后的AnnData文件以获得更好的分析效果。',
      uploadButton: '上传数据',
      uploading: '上传中...',
    },
    
    // Preprocess
    preprocess: {
      title: '预处理参数',
      minGenes: '每个细胞最小基因数',
      minCells: '每个基因最小细胞数',
      topGenes: '高变基因数量',
      nPcs: 'PCA主成分数',
      nNeighbors: '邻居数量',
      startButton: '运行预处理',
      processing: '处理中...',
    },
    
    // Training
    train: {
      title: '训练参数',
      hiddenDim: '隐藏层维度',
      nAgents: 'Agent数量',
      maxSteps: '每轮最大步数',
      numEpisodes: '训练轮数',
      learningRate: '学习率',
      gamma: '折扣因子',
      batchSize: '批次大小',
      memorySize: '记忆容量',
      resolution: '聚类分辨率',
      runButton: '开始训练',
      training: '训练中...',
    },
    
    // Data Info
    dataInfo: {
      title: '数据概要',
      cells: '细胞数',
      genes: '基因数',
      hvg: '高变基因',
      embeddings: 'Embeddings',
    },
    
    // Visualization
    visualization: {
      title: 'Embedding可视化',
    },
    
    // Results
    results: {
      title: '分析结果',
      summary: '结果概要',
      branches: '识别的分支数',
      pseudotimeRange: 'Pseudotime范围',
      entropyRange: 'Entropy范围',
      analysisMode: '分析模式',
      actorCriticAlgorithm: 'Actor-Critic',
      simulatedMode: '模拟模式',
      simulatedWarning: '当前为模拟模式（未安装scRL库）。实际分析需要在后端安装scRL。',
      trainingProcess: '训练进度',
      trainingRewards: '训练奖励',
      trainingErrors: '训练误差',
    },
    
    // Welcome
    welcome: {
      title: 'scRL分析',
      description: '上传单细胞数据，使用Actor-Critic强化学习算法分析细胞命运决策并计算pseudotime。',
      features: {
        actorCritic: 'Actor-Critic',
        multiAgent: '多智能体',
        noPrior: '无需先验',
      },
    },
    
    // Status
    status: {
      starting: '开始预处理...',
      initializing: '初始化模型...',
      complete: '完成',
      training: '训练中...',
      episode: '轮次',
    },
    
    // Language
    language: '语言',
    chinese: '中文',
    english: 'English',
  },
}

export type Language = 'en' | 'zh'
export type TranslationKey = keyof typeof translations.en
