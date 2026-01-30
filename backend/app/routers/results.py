"""
Results routes
Handles visualization data retrieval
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from app.services import scrl_service
from app.models import EmbeddingResult, GridVisualization

router = APIRouter()


@router.get("/{session_id}/embedding")
async def get_embedding(session_id: str):
    """
    Get embedding data for visualization
    
    Returns:
    - Cell coordinates (t-SNE or UMAP)
    - Cluster assignments
    - Cluster colors
    - Pseudotime values (if computed)
    - State values (if trained)
    """
    try:
        result = scrl_service.get_embedding_data(session_id)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/grid")
async def get_grid(session_id: str):
    """
    Get grid data for visualization
    
    Returns:
    - Grid point coordinates
    - Mapped grid indices
    - Boundary grid indices
    - Grid colors (by cluster)
    - Grid pseudotime values
    """
    try:
        result = scrl_service.get_grid_data(session_id)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/state-values")
async def get_state_values(session_id: str):
    """
    Get state values from trained model
    
    Returns the critic's evaluation of each state,
    representing the strength of fate decisions.
    """
    try:
        session = scrl_service.get_session(session_id)
        
        if not session.trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        return {
            "session_id": session_id,
            "state_values": session.state_values
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{session_id}/summary")
async def get_analysis_summary(session_id: str):
    """
    Get complete analysis summary
    
    Returns all analysis results in a single response
    for comprehensive visualization.
    """
    try:
        session = scrl_service.get_session(session_id)
        
        summary = {
            "session_id": session_id,
            "status": session.get_status()
        }
        
        # Add embedding if preprocessed
        if session.preprocessed:
            summary["embedding"] = scrl_service.get_embedding_data(session_id)
        
        # Add grid if generated
        if session.grid_generated:
            summary["grid"] = scrl_service.get_grid_data(session_id)
        
        # Add training results if trained
        if session.trained:
            summary["training"] = {
                "returns": session.training_history["returns"],
                "state_values": session.state_values
            }
        
        return summary
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
