"""
Pydantic models for API request/response schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class RewardType(str, Enum):
    """Reward type for RL training"""
    DISCRETE = "d"
    CONTINUOUS = "c"


class RewardMode(str, Enum):
    """Reward mode for RL training"""
    DECISION = "Decision"
    CONTRIBUTION = "Contribution"


class Algorithm(str, Enum):
    """RL algorithm selection"""
    TABULAR_Q = "TabularQ"
    ACTOR_CRITIC = "ActorCritic"
    DDQN = "DDQN"


# Data Upload Schemas
class DataUploadResponse(BaseModel):
    """Response after data upload"""
    session_id: str
    filename: str
    n_cells: int
    n_genes: int
    obs_columns: List[str]
    message: str


class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing"""
    normalize: bool = True
    log_transform: bool = True
    scale: bool = True
    n_pcs: int = Field(default=50, ge=10, le=100)
    embedding_method: str = Field(default="tsne", pattern="^(tsne|umap)$")
    clustering_resolution: float = Field(default=1.0, ge=0.1, le=3.0)


class PreprocessingResponse(BaseModel):
    """Response after preprocessing"""
    session_id: str
    n_clusters: int
    embedding_shape: List[int]
    message: str


# Grid Embedding Schemas
class GridConfig(BaseModel):
    """Configuration for grid embedding generation"""
    n: int = Field(default=50, ge=20, le=100, description="Grid resolution")
    j: int = Field(default=3, ge=1, le=10, description="Observer number for mask generation")
    n_jobs: int = Field(default=8, ge=1, le=16, description="Number of parallel jobs")


class GridResponse(BaseModel):
    """Response after grid generation"""
    session_id: str
    n_grids: int
    n_mapped: int
    n_boundary: int
    message: str


# Pseudotime Schemas
class PseudotimeConfig(BaseModel):
    """Configuration for pseudotime alignment"""
    start_cluster: str = Field(..., description="Starting cluster name")
    n_sample_cells: int = Field(default=10, ge=1, le=50)
    boundary: bool = True


class PseudotimeResponse(BaseModel):
    """Response after pseudotime alignment"""
    session_id: str
    pseudotime_range: List[float]
    message: str


# Reward Configuration Schemas
class DiscreteRewardConfig(BaseModel):
    """Configuration for discrete (lineage) rewards"""
    starts: List[str] = Field(..., description="Starting cluster names")
    ends: List[str] = Field(..., description="Target cluster names")
    beta: float = Field(default=1.0, ge=0.1, le=10.0)
    mode: RewardMode = RewardMode.DECISION


class ContinuousRewardConfig(BaseModel):
    """Configuration for continuous (gene) rewards"""
    reward_keys: List[str] = Field(..., description="Gene names for rewards")
    starts: Optional[List[str]] = Field(default=None, description="Starting cluster names")
    starts_keys: Optional[List[str]] = Field(default=None, description="Starting by gene expression")
    punish_keys: Optional[List[str]] = Field(default=None, description="Punishment gene names")
    beta: float = Field(default=1.0, ge=0.1, le=10.0)
    mode: RewardMode = RewardMode.DECISION


class RewardResponse(BaseModel):
    """Response after reward generation"""
    session_id: str
    reward_type: str
    reward_mode: str
    message: str


# Training Schemas
class TrainingConfig(BaseModel):
    """Configuration for RL training"""
    algo: Algorithm = Algorithm.ACTOR_CRITIC
    reward_type: RewardType = RewardType.CONTINUOUS
    reward_mode: RewardMode = RewardMode.DECISION
    num_episodes: int = Field(default=5000, ge=100, le=50000)
    max_step: int = Field(default=50, ge=10, le=200)
    gamma: float = Field(default=0.9, ge=0.5, le=0.99)
    hidden_dim: int = Field(default=128, ge=32, le=512)


class TrainingStatus(BaseModel):
    """Training status response"""
    session_id: str
    status: str  # "running", "completed", "failed"
    progress: float  # 0.0 to 1.0
    current_episode: int
    total_episodes: int
    current_reward: Optional[float]
    message: str


class TrainingResult(BaseModel):
    """Training result response"""
    session_id: str
    final_reward: float
    training_time: float
    return_history: List[float]
    message: str


# Results Schemas
class EmbeddingResult(BaseModel):
    """Embedding data for visualization"""
    embedding: List[List[float]]  # [[x, y], ...]
    clusters: List[str]
    cluster_colors: List[str]
    pseudotime: List[float]
    state_values: Optional[List[float]]


class GridVisualization(BaseModel):
    """Grid visualization data"""
    grids: List[List[float]]
    mapped_grids: List[int]
    boundary_grids: List[int]
    grid_colors: List[str]
    grid_pseudotime: List[float]


class TrajectoryData(BaseModel):
    """Trajectory analysis data"""
    trajectories: List[List[int]]  # List of grid indices
    trajectory_rewards: List[float]


class GeneAnalysisResult(BaseModel):
    """Gene analysis result"""
    gene_name: str
    contribution_score: float
    decision_score: float
    expression_trend: List[float]


class AnalysisResults(BaseModel):
    """Complete analysis results"""
    session_id: str
    embedding: EmbeddingResult
    grids: GridVisualization
    state_values: Dict[str, List[float]]
    genes: Optional[List[GeneAnalysisResult]]


# Session Management
class SessionInfo(BaseModel):
    """Session information"""
    session_id: str
    created_at: str
    data_loaded: bool
    preprocessed: bool
    grid_generated: bool
    rewards_generated: bool
    trained: bool
    current_step: str


class SessionList(BaseModel):
    """List of sessions"""
    sessions: List[SessionInfo]
