"""
Training routes
Handles RL model training
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict

from app.services import scrl_service
from app.models import TrainingConfig, TrainingStatus, TrainingResult

router = APIRouter()

# Store training tasks status
training_tasks: Dict[str, TrainingStatus] = {}


@router.post("/{session_id}/train", response_model=TrainingResult)
async def train_model(session_id: str, config: TrainingConfig):
    """
    Train the reinforcement learning model
    
    This endpoint trains an Actor-Critic (or other algorithm) model
    to learn optimal policies for navigating the cellular fate landscape.
    
    The trained model can then be used to:
    - Identify fate decision strength at each state
    - Generate trajectories showing differentiation paths
    - Analyze gene/lineage contributions to fate decisions
    """
    try:
        result = scrl_service.train(
            session_id,
            algo=config.algo.value,
            num_episodes=config.num_episodes,
            max_step=config.max_step,
            gamma=config.gamma,
            hidden_dim=config.hidden_dim
        )
        
        return TrainingResult(
            session_id=session_id,
            final_reward=result["final_reward"],
            training_time=result["training_time"],
            return_history=result["return_history"],
            message="Training completed successfully"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/training-history")
async def get_training_history(session_id: str):
    """Get the training history (rewards over episodes)"""
    try:
        session = scrl_service.get_session(session_id)
        
        if not session.trained:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        return {
            "session_id": session_id,
            "returns": session.training_history["returns"],
            "values": session.training_history["values"]
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
