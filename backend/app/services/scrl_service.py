"""
scRL Analysis Service
Encapsulates the scRL library functionality for web API usage
"""
import os
import uuid
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import scanpy as sc
import scRL
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class AnalysisSession:
    """
    Represents a single analysis session with all associated data and state
    """
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        
        # Data storage
        self.adata = None  # AnnData object
        self.gres = None   # Grid results
        self.trainer = None  # Trained model
        
        # State tracking
        self.data_loaded = False
        self.preprocessed = False
        self.grid_generated = False
        self.cluster_projected = False
        self.pseudotime_aligned = False
        self.rewards_generated = False
        self.trained = False
        
        # Results storage
        self.training_history = {
            "returns": [],
            "values": []
        }
        self.state_values = {}
        
    def get_status(self) -> Dict[str, Any]:
        """Get current session status"""
        steps = [
            ("data_loaded", "Data Loaded"),
            ("preprocessed", "Preprocessed"),
            ("grid_generated", "Grid Generated"),
            ("cluster_projected", "Clusters Projected"),
            ("pseudotime_aligned", "Pseudotime Aligned"),
            ("rewards_generated", "Rewards Generated"),
            ("trained", "Model Trained")
        ]
        
        current_step = "Not Started"
        for attr, name in steps:
            if getattr(self, attr):
                current_step = name
                
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "data_loaded": self.data_loaded,
            "preprocessed": self.preprocessed,
            "grid_generated": self.grid_generated,
            "rewards_generated": self.rewards_generated,
            "trained": self.trained,
            "current_step": current_step
        }
    
    def save(self, directory: str):
        """Save session to disk"""
        path = os.path.join(directory, f"{self.session_id}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(directory: str, session_id: str) -> 'AnalysisSession':
        """Load session from disk"""
        path = os.path.join(directory, f"{session_id}.pkl")
        with open(path, 'rb') as f:
            return pickle.load(f)


class ScRLService:
    """
    Service class for scRL analysis operations
    """
    
    def __init__(self, temp_dir: str = "temp", results_dir: str = "results"):
        self.temp_dir = temp_dir
        self.results_dir = results_dir
        self.sessions: Dict[str, AnalysisSession] = {}
        
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    
    def create_session(self) -> AnalysisSession:
        """Create a new analysis session"""
        session = AnalysisSession()
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> AnalysisSession:
        """Get an existing session"""
        if session_id not in self.sessions:
            # Try loading from disk
            try:
                session = AnalysisSession.load(self.results_dir, session_id)
                self.sessions[session_id] = session
            except FileNotFoundError:
                raise ValueError(f"Session {session_id} not found")
        return self.sessions[session_id]
    
    def load_data(self, session_id: str, file_path: str) -> Dict[str, Any]:
        """
        Load single-cell data from file
        Supports: .h5ad, .csv
        """
        session = self.get_session(session_id)
        
        if file_path.endswith('.h5ad'):
            session.adata = sc.read_h5ad(file_path)
        elif file_path.endswith('.csv'):
            # Assume cells x genes format
            df = pd.read_csv(file_path, index_col=0)
            session.adata = sc.AnnData(df)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        session.data_loaded = True
        
        return {
            "n_cells": session.adata.n_obs,
            "n_genes": session.adata.n_vars,
            "obs_columns": list(session.adata.obs.columns)
        }
    
    def load_demo_data(self, session_id: str, dataset: str = "paul15") -> Dict[str, Any]:
        """Load demo dataset for testing"""
        session = self.get_session(session_id)
        
        if dataset == "paul15":
            session.adata = sc.datasets.paul15()
        else:
            raise ValueError(f"Unknown demo dataset: {dataset}")
        
        session.data_loaded = True
        
        return {
            "n_cells": session.adata.n_obs,
            "n_genes": session.adata.n_vars,
            "obs_columns": list(session.adata.obs.columns)
        }
    
    def preprocess(
        self,
        session_id: str,
        normalize: bool = True,
        log_transform: bool = True,
        scale: bool = True,
        n_pcs: int = 50,
        embedding_method: str = "tsne",
        clustering_resolution: float = 1.0
    ) -> Dict[str, Any]:
        """
        Preprocess the loaded data
        """
        session = self.get_session(session_id)
        
        if not session.data_loaded:
            raise ValueError("No data loaded. Please load data first.")
        
        adata = session.adata
        
        # Standard preprocessing
        if normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)
        
        if log_transform:
            sc.pp.log1p(adata)
        
        if scale:
            sc.pp.scale(adata, max_value=10)
        
        # PCA
        sc.pp.pca(adata, n_comps=n_pcs)
        
        # Embedding
        if embedding_method == "tsne":
            sc.tl.tsne(adata, perplexity=50)
        elif embedding_method == "umap":
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        
        # Clustering
        if 'neighbors' not in adata.uns:
            sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=clustering_resolution)
        
        session.preprocessed = True
        
        return {
            "n_clusters": len(adata.obs['leiden'].unique()),
            "embedding_shape": list(adata.obsm[f'X_{embedding_method}'].shape)
        }
    
    def generate_grid(
        self,
        session_id: str,
        n: int = 50,
        j: int = 3,
        n_jobs: int = 8,
        embedding_key: str = None
    ) -> Dict[str, Any]:
        """
        Generate grid embedding from the data
        """
        session = self.get_session(session_id)
        
        if not session.preprocessed:
            raise ValueError("Data not preprocessed. Please preprocess first.")
        
        adata = session.adata
        
        # Determine embedding key
        if embedding_key is None:
            if 'X_tsne' in adata.obsm:
                embedding_key = 'X_tsne'
            elif 'X_umap' in adata.obsm:
                embedding_key = 'X_umap'
            else:
                raise ValueError("No embedding found in data")
        
        X = adata.obsm[embedding_key]
        
        # Generate grids using scRL
        session.gres = scRL.grids_from_embedding(X, n=n, j=j, n_jobs=n_jobs)
        session.grid_generated = True
        
        # Project clusters
        clusters = adata.obs['leiden']
        cluster_colors = adata.uns.get('leiden_colors', None)
        scRL.project_cluster(session.gres, clusters, cluster_colors)
        session.cluster_projected = True
        
        return {
            "n_grids": n * n,
            "n_mapped": len(session.gres.grids['mapped_grids']),
            "n_boundary": len(session.gres.grids['mapped_boundary'])
        }
    
    def align_pseudotime(
        self,
        session_id: str,
        start_cluster: str,
        n_sample_cells: int = 10,
        boundary: bool = True
    ) -> Dict[str, Any]:
        """
        Align pseudotime starting from specified cluster
        """
        session = self.get_session(session_id)
        
        if not session.grid_generated:
            raise ValueError("Grid not generated. Please generate grid first.")
        
        scRL.align_pseudotime(
            session.gres,
            start_cluster,
            n_sample_cells=n_sample_cells,
            boundary=boundary
        )
        
        scRL.project_back(session.gres, 'pseudotime')
        session.adata.obs['scRL_pseudotime'] = session.gres.embedding['pseudotime']
        session.pseudotime_aligned = True
        
        pseudotime = session.gres.grids['pseudotime']
        
        return {
            "pseudotime_range": [float(pseudotime.min()), float(pseudotime.max())]
        }
    
    def generate_discrete_rewards(
        self,
        session_id: str,
        starts: List[str],
        ends: List[str],
        beta: float = 1.0,
        mode: str = "Decision"
    ) -> Dict[str, Any]:
        """
        Generate discrete (lineage-based) rewards
        """
        session = self.get_session(session_id)
        
        if not session.pseudotime_aligned:
            raise ValueError("Pseudotime not aligned. Please align pseudotime first.")
        
        scRL.d_rewards(session.gres, starts=starts, ends=ends, beta=beta, mode=mode)
        session.rewards_generated = True
        session.reward_type = 'd'
        session.reward_mode = mode
        
        return {
            "reward_type": "discrete",
            "reward_mode": mode
        }
    
    def generate_continuous_rewards(
        self,
        session_id: str,
        reward_keys: List[str],
        starts: Optional[List[str]] = None,
        starts_keys: Optional[List[str]] = None,
        punish_keys: Optional[List[str]] = None,
        beta: float = 1.0,
        mode: str = "Decision"
    ) -> Dict[str, Any]:
        """
        Generate continuous (gene-based) rewards
        """
        session = self.get_session(session_id)
        
        if not session.pseudotime_aligned:
            raise ValueError("Pseudotime not aligned. Please align pseudotime first.")
        
        # Project gene expression to grids
        gene_exp = pd.DataFrame(
            session.adata[:, reward_keys].X,
            columns=reward_keys
        )
        scRL.project(session.gres, gene_exp)
        
        scRL.c_rewards(
            session.gres,
            reward_keys=reward_keys,
            starts=starts,
            starts_keys=starts_keys,
            punish_keys=punish_keys,
            beta=beta,
            mode=mode
        )
        session.rewards_generated = True
        session.reward_type = 'c'
        session.reward_mode = mode
        
        return {
            "reward_type": "continuous",
            "reward_mode": mode
        }
    
    def train(
        self,
        session_id: str,
        algo: str = "ActorCritic",
        num_episodes: int = 5000,
        max_step: int = 50,
        gamma: float = 0.9,
        hidden_dim: int = 128,
        callback: callable = None
    ) -> Dict[str, Any]:
        """
        Train the RL model
        """
        session = self.get_session(session_id)
        
        if not session.rewards_generated:
            raise ValueError("Rewards not generated. Please generate rewards first.")
        
        adata = session.adata
        gres = session.gres
        
        # Get latent space
        X_pca = adata.obsm['X_pca']
        
        # Create trainer
        session.trainer = scRL.trainer(
            algo=algo,
            gres=gres,
            reward_type=session.reward_type,
            reward_mode=session.reward_mode,
            X_latent=X_pca,
            num_episodes=num_episodes,
            max_step=max_step,
            gamma=gamma,
            hidden_dim=hidden_dim
        )
        
        # Train
        import time
        start_time = time.time()
        return_list, value_list = session.trainer.train()
        training_time = time.time() - start_time
        
        session.training_history["returns"] = [float(r) for r in return_list]
        session.training_history["values"] = [float(v) for v in value_list]
        session.trained = True
        
        # Extract state values
        key = f"fate_{session.reward_mode}"
        scRL.get_state_value(gres, session.trainer, key)
        scRL.project_back(gres, key)
        session.state_values[key] = gres.embedding[key].tolist()
        adata.obs[f'scRL_{key}'] = gres.embedding[key]
        
        # Save session
        session.save(self.results_dir)
        
        return {
            "final_reward": float(return_list[-1]) if return_list else 0.0,
            "training_time": training_time,
            "return_history": session.training_history["returns"]
        }
    
    def get_embedding_data(self, session_id: str) -> Dict[str, Any]:
        """
        Get embedding data for visualization
        """
        session = self.get_session(session_id)
        
        if not session.preprocessed:
            raise ValueError("Data not preprocessed")
        
        adata = session.adata
        
        # Determine embedding
        if 'X_tsne' in adata.obsm:
            embedding = adata.obsm['X_tsne'].tolist()
        elif 'X_umap' in adata.obsm:
            embedding = adata.obsm['X_umap'].tolist()
        else:
            raise ValueError("No embedding found")
        
        # Get clusters and colors
        clusters = adata.obs['leiden'].tolist()
        
        if 'leiden_colors' in adata.uns:
            color_map = dict(zip(
                adata.obs['leiden'].cat.categories,
                adata.uns['leiden_colors']
            ))
            cluster_colors = [color_map[c] for c in clusters]
        else:
            cluster_colors = clusters
        
        # Get pseudotime if available
        pseudotime = []
        if 'scRL_pseudotime' in adata.obs:
            pseudotime = adata.obs['scRL_pseudotime'].tolist()
        
        # Get state values if available
        state_values = None
        if session.state_values:
            state_values = list(session.state_values.values())[0]
        
        return {
            "embedding": embedding,
            "clusters": clusters,
            "cluster_colors": cluster_colors,
            "pseudotime": pseudotime,
            "state_values": state_values
        }
    
    def get_grid_data(self, session_id: str) -> Dict[str, Any]:
        """
        Get grid data for visualization
        """
        session = self.get_session(session_id)
        
        if not session.grid_generated:
            raise ValueError("Grid not generated")
        
        gres = session.gres
        
        grids = gres.grids['grids'].tolist()
        mapped_grids = gres.grids['mapped_grids'].tolist()
        boundary_grids = gres.grids['mapped_boundary'].tolist()
        
        # Get colors
        if 'mapped_grids_colors' in gres.grids:
            colors = [str(c) for c in gres.grids['mapped_grids_colors']]
        else:
            colors = ['#cccccc'] * len(mapped_grids)
        
        # Get pseudotime
        pseudotime = []
        if 'pseudotime' in gres.grids:
            pseudotime = [float(gres.grids['pseudotime'][i]) for i in mapped_grids]
        
        return {
            "grids": grids,
            "mapped_grids": mapped_grids,
            "boundary_grids": boundary_grids,
            "grid_colors": colors,
            "grid_pseudotime": pseudotime
        }
    
    def get_clusters(self, session_id: str) -> List[str]:
        """Get list of cluster names"""
        session = self.get_session(session_id)
        
        if not session.preprocessed:
            raise ValueError("Data not preprocessed")
        
        return list(session.adata.obs['leiden'].cat.categories)
    
    def get_genes(self, session_id: str, pattern: str = None) -> List[str]:
        """Get list of gene names, optionally filtered"""
        session = self.get_session(session_id)
        
        if not session.data_loaded:
            raise ValueError("Data not loaded")
        
        genes = list(session.adata.var_names)
        
        if pattern:
            import re
            genes = [g for g in genes if re.search(pattern, g, re.IGNORECASE)]
        
        return genes[:100]  # Limit to 100 genes


# Global service instance
scrl_service = ScRLService()
