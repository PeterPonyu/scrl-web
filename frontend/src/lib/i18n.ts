export const translations = {
  en: {
    // Header
    appName: 'scRL Web',
    appDescription: 'Single-cell Fate Decision Analysis Platform · Actor-Critic Algorithm',
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
      dragDrop: 'Drag and drop files here or',
      clickSelect: 'click to select',
      supportedFormats: 'Supported formats: .h5ad, .csv',
      suggestion: 'Suggestion: Use preprocessed AnnData files for better results',
      uploadButton: 'Upload and Load Data',
      uploading: 'Uploading...',
    },
    
    // Preprocess
    preprocess: {
      title: 'Preprocessing Parameters',
      minGenes: 'Min Genes',
      minCells: 'Min Cells',
      topGenes: 'HVG Count',
      nPcs: 'PCA Dimensions',
      nNeighbors: 'Neighbors',
      startButton: 'Start Preprocessing',
      processing: 'Processing...',
    },
    
    // Training
    train: {
      title: 'scRL Training Parameters',
      hiddenDim: 'Hidden Dimensions',
      nAgents: 'Agent Count',
      maxSteps: 'Max Steps per Episode',
      numEpisodes: 'Training Episodes',
      learningRate: 'Learning Rate',
      gamma: 'Gamma (Discount Factor)',
      batchSize: 'Batch Size',
      memorySize: 'Memory Size',
      resolution: 'Cluster Resolution',
      runButton: 'Start Training',
      training: 'Training...',
    },
    
    // Data Info
    dataInfo: {
      title: 'Data Info',
      cells: 'Cells',
      genes: 'Genes',
      hvg: 'HVG',
      embeddings: 'Embeddings',
    },
    
    // Visualization
    visualization: {
      title: 'UMAP Visualization',
    },
    
    // Results
    results: {
      title: 'Analysis Results Summary',
      summary: 'Analysis Results Summary',
      branches: 'Identified Branches',
      pseudotimeRange: 'Pseudotime Range',
      entropyRange: 'Entropy Range',
      analysisMode: 'Analysis Mode',
      actorCriticAlgorithm: 'Actor-Critic',
      simulatedMode: 'Simulated Mode',
      simulatedWarning: 'Currently using simulated data mode (scRL library not installed). Actual analysis requires scRL installation on the backend.',
      trainingProcess: 'Training Process',
      trainingRewards: 'Training Rewards',
      trainingErrors: 'Training Errors',
    },
    
    // Welcome
    welcome: {
      title: 'Start scRL Analysis',
      description: 'Upload your single-cell data, use Actor-Critic reinforcement learning algorithm to analyze cell fate decisions and calculate pseudotime',
      features: {
        actorCritic: 'Actor-Critic Algorithm',
        multiAgent: 'Multi-Agent Learning',
        noPrior: 'No Prior Knowledge Required',
      },
    },
    
    // Status
    status: {
      starting: 'Starting preprocessing...',
      initializing: 'Initializing scRL model...',
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
    appDescription: '单细胞命运决策分析平台 · Actor-Critic算法',
    session: '会话',
    initializing: '初始化中...',
    exportResults: '导出结果',
    
    // Steps
    steps: {
      upload: '数据上传',
      preprocess: '预处理',
      train: '训练模型',
      results: '结果展示',
    },
    
    // Upload
    upload: {
      title: '上传单细胞数据',
      dragDrop: '拖放文件到此处或',
      clickSelect: '点击选择',
      supportedFormats: '支持格式: .h5ad, .csv',
      suggestion: '建议: 使用预处理后的AnnData文件效果更佳',
      uploadButton: '上传并加载数据',
      uploading: '上传中...',
    },
    
    // Preprocess
    preprocess: {
      title: '预处理参数',
      minGenes: '最小基因数',
      minCells: '最小细胞数',
      topGenes: '高变基因数',
      nPcs: 'PCA维数',
      nNeighbors: '邻居数',
      startButton: '开始预处理',
      processing: '处理中...',
    },
    
    // Training
    train: {
      title: 'scRL训练参数',
      hiddenDim: '隐藏层维度',
      nAgents: 'Agent数量',
      maxSteps: '每轮最大步数',
      numEpisodes: '训练轮数',
      learningRate: '学习率',
      gamma: 'Gamma (折扣因子)',
      batchSize: '批次大小',
      memorySize: '记忆大小',
      resolution: '聚类分辨率',
      runButton: '开始训练',
      training: '训练中...',
    },
    
    // Data Info
    dataInfo: {
      title: '数据信息',
      cells: '细胞数',
      genes: '基因数',
      hvg: '高变基因',
      embeddings: 'Embeddings',
    },
    
    // Visualization
    visualization: {
      title: 'UMAP可视化',
    },
    
    // Results
    results: {
      title: '分析结果摘要',
      summary: '分析结果摘要',
      branches: '识别分支数',
      pseudotimeRange: 'Pseudotime范围',
      entropyRange: 'Entropy范围',
      analysisMode: '分析模式',
      actorCriticAlgorithm: 'Actor-Critic算法',
      simulatedMode: '模拟模式',
      simulatedWarning: '当前使用模拟数据模式（未安装scRL库）。实际分析需要在后端环境中安装scRL。',
      trainingProcess: '训练过程',
      trainingRewards: '训练奖励',
      trainingErrors: '训练误差',
    },
    
    // Welcome
    welcome: {
      title: '开始scRL分析',
      description: '上传您的单细胞数据，使用Actor-Critic强化学习算法分析细胞命运决策并计算pseudotime',
      features: {
        actorCritic: 'Actor-Critic算法',
        multiAgent: '多智能体学习',
        noPrior: '无需先验知识',
      },
    },
    
    // Status
    status: {
      starting: '开始预处理...',
      initializing: '初始化scRL模型...',
      complete: '完成',
      training: '训练进行中...',
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
