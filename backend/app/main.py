"""
scRL-Web Backend API
Single-cell Reinforcement Learning Web Interface

This FastAPI application provides RESTful APIs for:
1. Single-cell data upload and preprocessing
2. Grid embedding generation
3. Reinforcement learning training
4. Results visualization and export
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from contextlib import asynccontextmanager
import os

from app.routers import analysis, data, training, results
from app.core.config import settings

# Global state for storing analysis sessions
sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup: Initialize any required resources
    print("🚀 scRL-Web API Starting...")
    os.makedirs("temp", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    yield
    # Shutdown: Clean up resources
    print("🛑 scRL-Web API Shutting down...")
    # Clean up temp files
    import shutil
    if os.path.exists("temp"):
        shutil.rmtree("temp")

app = FastAPI(
    title="scRL-Web API",
    description="""
    ## Single-cell Reinforcement Learning Web API
    
    This API provides endpoints for analyzing single-cell RNA sequencing data
    using reinforcement learning to decode cellular fate decisions.
    
    ### Features:
    - **Data Upload**: Upload h5ad/csv files for analysis
    - **Grid Embedding**: Generate 2D grid representations
    - **RL Training**: Train Actor-Critic models for fate decision analysis
    - **Visualization**: Get embeddings, trajectories, and gene analysis results
    """,
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/api/data", tags=["Data Management"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis Pipeline"])
app.include_router(training.router, prefix="/api/training", tags=["RL Training"])
app.include_router(results.router, prefix="/api/results", tags=["Results & Visualization"])


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "scRL-Web API",
        "version": "0.1.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
