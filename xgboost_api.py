#!/usr/bin/env python3
"""
XGBoost Model API for QuantConnect Integration
Provides separate endpoints for spot and futures models
Returns data in base64 format for easy consumption by QuantConnect
"""

import os
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Database imports
from database_storage import DatabaseStorage
from dotenv import load_dotenv
from routes import MODEL_VERSIONS, register_model_routes

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="XGBoost Trading Model API",
    description="API for XGBoost trading models and dataset summaries for QuantConnect",
    version="2.0.0"
)

# Enable CORS - Load from environment
from_env_origins = os.getenv('ALLOWED_ORIGINS', '*').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=from_env_origins if from_env_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database storage
try:
    db_storage = DatabaseStorage()
    logger.info("✅ Database connected successfully")
except Exception as e:
    logger.error(f"❌ Failed to connect to database: {e}")
    db_storage = None

# Pydantic models for response
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database_connected: bool

# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        database_connected=db_storage is not None
    )

register_model_routes(app, db_storage=db_storage, logger=logger)

# =========================================================
# DATABASE MODEL REFERENCES
# =========================================================

def init_db_models():
    """Initialize database model references."""
    if db_storage:
        from database_storage import ModelStorage, DatasetSummary
        db_storage.db_model = ModelStorage
        db_storage.db_dataset_summary = DatasetSummary

# Initialize database models
init_db_models()

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv('API_PORT', 5000))
    host = os.getenv('API_HOST', '0.0.0.0')

    logger.info(f"🚀 Starting XGBoost FastAPI server on {host}:{port}")
    logger.info("📊 Available endpoints:")
    logger.info("   GET /health - Health check")
    for version in MODEL_VERSIONS:
        label = version.replace("_", " ").upper()
        prefix = f"/api/v1/{version}"
        logger.info("")
        logger.info(f"   {label} Endpoints:")
        logger.info(f"   GET {prefix}/latest/model - Get latest {version} model")
        logger.info(f"   GET {prefix}/latest/dataset-summary - Get latest {version} dataset summary")
        logger.info(f"   GET {prefix}/model/{{model_id}} - Get {version} model by ID")
        logger.info(f"   GET {prefix}/summary/{{summary_id}} - Get {version} dataset summary by ID")
        logger.info(f"   GET {prefix}/models - List all {version} models")
        logger.info(f"   POST {prefix}/model - Insert new {version} model")
    logger.info("")
    logger.info("📖 API docs available at: http://localhost:5000/docs")

    uvicorn.run(app, host=host, port=port)
