"""
Analysis pipeline routes
Handles grid generation, pseudotime alignment, and reward generation
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional

from app.services import scrl_service
from app.models import (
    GridConfig, GridResponse,
    PseudotimeConfig, PseudotimeResponse,
    DiscreteRewardConfig, ContinuousRewardConfig, RewardResponse
)

router = APIRouter()


@router.post("/{session_id}/grid", response_model=GridResponse)
async def generate_grid(session_id: str, config: GridConfig):
    """
    Generate grid embedding from the preprocessed data
    
    The grid embedding transforms the 2D embedding into a structured grid
    that can be navigated by the RL agent.
    """
    try:
        result = scrl_service.generate_grid(
            session_id,
            n=config.n,
            j=config.j,
            n_jobs=config.n_jobs
        )
        
        return GridResponse(
            session_id=session_id,
            n_grids=result["n_grids"],
            n_mapped=result["n_mapped"],
            n_boundary=result["n_boundary"],
            message="Grid embedding generated successfully"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/pseudotime", response_model=PseudotimeResponse)
async def align_pseudotime(session_id: str, config: PseudotimeConfig):
    """
    Align pseudotime starting from the specified cluster
    
    Pseudotime represents the developmental progression of cells,
    calculated using Dijkstra's algorithm from the starting cluster.
    """
    try:
        result = scrl_service.align_pseudotime(
            session_id,
            start_cluster=config.start_cluster,
            n_sample_cells=config.n_sample_cells,
            boundary=config.boundary
        )
        
        return PseudotimeResponse(
            session_id=session_id,
            pseudotime_range=result["pseudotime_range"],
            message=f"Pseudotime aligned from cluster '{config.start_cluster}'"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/rewards/discrete", response_model=RewardResponse)
async def generate_discrete_rewards(session_id: str, config: DiscreteRewardConfig):
    """
    Generate discrete (lineage-based) rewards
    
    Use this when analyzing fate decisions based on cluster/lineage membership.
    The agent receives rewards for reaching target clusters from starting clusters.
    
    Modes:
    - Decision: Higher rewards at earlier pseudotime (identifies early fate decision points)
    - Contribution: Higher rewards at later pseudotime (identifies late fate contributions)
    """
    try:
        result = scrl_service.generate_discrete_rewards(
            session_id,
            starts=config.starts,
            ends=config.ends,
            beta=config.beta,
            mode=config.mode.value
        )
        
        return RewardResponse(
            session_id=session_id,
            reward_type=result["reward_type"],
            reward_mode=result["reward_mode"],
            message="Discrete rewards generated successfully"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/rewards/continuous", response_model=RewardResponse)
async def generate_continuous_rewards(session_id: str, config: ContinuousRewardConfig):
    """
    Generate continuous (gene-based) rewards
    
    Use this when analyzing fate decisions based on gene expression patterns.
    The agent receives rewards based on specified gene expression values.
    
    Modes:
    - Decision: Higher rewards at earlier pseudotime (identifies early fate decision points)
    - Contribution: Higher rewards at later pseudotime (identifies late fate contributions)
    """
    try:
        result = scrl_service.generate_continuous_rewards(
            session_id,
            reward_keys=config.reward_keys,
            starts=config.starts,
            starts_keys=config.starts_keys,
            punish_keys=config.punish_keys,
            beta=config.beta,
            mode=config.mode.value
        )
        
        return RewardResponse(
            session_id=session_id,
            reward_type=result["reward_type"],
            reward_mode=result["reward_mode"],
            message="Continuous rewards generated successfully"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/status")
async def get_analysis_status(session_id: str):
    """Get the current analysis pipeline status"""
    try:
        session = scrl_service.get_session(session_id)
        return session.get_status()
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
