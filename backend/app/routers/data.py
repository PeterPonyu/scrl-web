"""
Data management routes
Handles file upload, demo data loading, and data info retrieval
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import os
import shutil
from typing import Optional

from app.services import scrl_service
from app.models import DataUploadResponse, PreprocessingConfig, PreprocessingResponse
from app.core.config import settings

router = APIRouter()


@router.post("/upload", response_model=DataUploadResponse)
async def upload_data(file: UploadFile = File(...)):
    """
    Upload single-cell data file
    
    Supports:
    - .h5ad (AnnData format)
    - .csv (cells x genes matrix)
    """
    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Create new session
    session = scrl_service.create_session()
    
    # Save uploaded file
    file_path = os.path.join(settings.TEMP_DIR, f"{session.session_id}{ext}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load data into session
        result = scrl_service.load_data(session.session_id, file_path)
        
        return DataUploadResponse(
            session_id=session.session_id,
            filename=file.filename,
            n_cells=result["n_cells"],
            n_genes=result["n_genes"],
            obs_columns=result["obs_columns"],
            message="Data uploaded and loaded successfully"
        )
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file after loading
        if os.path.exists(file_path):
            os.remove(file_path)


@router.post("/demo/{dataset}", response_model=DataUploadResponse)
async def load_demo_data(dataset: str = "paul15"):
    """
    Load a demo dataset for testing
    
    Available datasets:
    - paul15: Mouse hematopoiesis (Paul et al., 2015)
    """
    if dataset not in ["paul15"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset: {dataset}. Available: paul15"
        )
    
    session = scrl_service.create_session()
    
    try:
        result = scrl_service.load_demo_data(session.session_id, dataset)
        
        return DataUploadResponse(
            session_id=session.session_id,
            filename=f"demo_{dataset}",
            n_cells=result["n_cells"],
            n_genes=result["n_genes"],
            obs_columns=result["obs_columns"],
            message=f"Demo dataset '{dataset}' loaded successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/preprocess", response_model=PreprocessingResponse)
async def preprocess_data(
    session_id: str,
    config: PreprocessingConfig
):
    """
    Preprocess the loaded data
    
    Steps:
    1. Normalization (optional)
    2. Log transformation (optional)
    3. Scaling (optional)
    4. PCA
    5. Embedding (t-SNE or UMAP)
    6. Clustering (Leiden)
    """
    try:
        result = scrl_service.preprocess(
            session_id,
            normalize=config.normalize,
            log_transform=config.log_transform,
            scale=config.scale,
            n_pcs=config.n_pcs,
            embedding_method=config.embedding_method,
            clustering_resolution=config.clustering_resolution
        )
        
        return PreprocessingResponse(
            session_id=session_id,
            n_clusters=result["n_clusters"],
            embedding_shape=result["embedding_shape"],
            message="Preprocessing completed successfully"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/info")
async def get_data_info(session_id: str):
    """Get information about the loaded data"""
    try:
        session = scrl_service.get_session(session_id)
        
        if not session.data_loaded:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        adata = session.adata
        
        return {
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "obs_columns": list(adata.obs.columns),
            "var_columns": list(adata.var.columns),
            "obsm_keys": list(adata.obsm.keys()),
            "uns_keys": list(adata.uns.keys())
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{session_id}/clusters")
async def get_clusters(session_id: str):
    """Get list of cluster names"""
    try:
        clusters = scrl_service.get_clusters(session_id)
        return {"clusters": clusters}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{session_id}/genes")
async def get_genes(
    session_id: str,
    pattern: Optional[str] = Query(None, description="Filter genes by pattern")
):
    """Get list of gene names"""
    try:
        genes = scrl_service.get_genes(session_id, pattern)
        return {"genes": genes}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
