"""
Application configuration settings
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "scRL-Web"
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # CORS Settings - 允许所有来源以支持多种部署平台
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.vercel.app",
        "https://*.onrender.com",
        "https://scrl-web.vercel.app",
        "https://scrl-web.onrender.com",
    ]
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS: List[str] = [".h5ad", ".csv", ".tsv", ".txt"]
    
    # Analysis Settings
    DEFAULT_GRID_N: int = 50
    DEFAULT_GRID_J: int = 3
    DEFAULT_NUM_EPISODES: int = 5000
    DEFAULT_ALGO: str = "ActorCritic"
    
    # Storage
    TEMP_DIR: str = "temp"
    RESULTS_DIR: str = "results"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
