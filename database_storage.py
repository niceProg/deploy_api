#!/usr/bin/env python3
"""
Database storage manager for XGBoost training pipeline.
Stores all outputs (features, models, evaluation results) to database server.
This enables easy API access for QuantConnect integration.
"""

import os
import json
import pickle
import base64
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine, text, Column, String, DateTime, Integer, Float, Text, LargeBinary, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.mysql import MEDIUMBLOB, LONGTEXT, LONGBLOB
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
Base = declarative_base()

# DATABASE TABLES per client requirement
# Client hanya mau: xgboost_models dan xgboost_dataset_summary

class DatasetSummary(Base):
    __tablename__ = 'xgboost_dataset_summary'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), default='spot')  # 'spot' or 'futures'
    summary_file = Column(String(500))
    summary_data = Column(LONGBLOB)  # Store summary content as binary blob (like model_data)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Composite index for performance: filter by version, order by created_at DESC
    __table_args__ = (
        Index('idx_model_version_created', 'model_version', 'created_at'),
        {'mysql_engine': 'InnoDB'},
    )

class ModelStorage(Base):
    __tablename__ = 'xgboost_models'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(200), nullable=False)
    model_version = Column(String(50), default='spot')  # 'spot' or 'futures'
    model_file = Column(String(500))
    model_data = Column(LONGBLOB)  # pickled model
    created_at = Column(DateTime, default=datetime.utcnow)

    # Composite index for performance: filter by version, order by created_at DESC
    __table_args__ = (
        Index('idx_model_version_created', 'model_version', 'created_at'),
        {'mysql_engine': 'InnoDB'},
    )

  
class DatabaseStorage:
    """Manages database storage for XGBoost training pipeline outputs."""

    def __init__(self, db_config: Optional[Dict] = None, storage_path: str = './output_train'):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Database configuration
        if db_config is None:
            db_host = os.getenv('DB_HOST')
            if not db_host:
                raise ValueError("❌ DB_HOST environment variable not set! Required for database storage.")

            db_config = {
                'host': db_host,
                'port': int(os.getenv('DB_PORT', 3306)),
                'database': os.getenv('DB_NAME', 'xgboost_training'),
                'user': os.getenv('DB_USER', 'root'),
                'password': os.getenv('DB_PASSWORD', '')
            }

        self.db_config = db_config
        self.engine = None
        self.SessionLocal = None

        self._initialize_database()

    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            # Create connection string
            connection_string = (
                f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
                f"?charset=utf8mb4"
            )

            # Create engine
            self.engine = create_engine(
                connection_string,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False  # Set to True for SQL logging
            )

            # CREATE xgboost_models and xgboost_dataset_summary tables per client requirement
            # Client hanya mau: xgboost_models dan xgboost_dataset_summary
            DatasetSummary.__table__.create(bind=self.engine, checkfirst=True)
            ModelStorage.__table__.create(bind=self.engine, checkfirst=True)
            logger.info("✅ Created xgboost_dataset_summary and xgboost_models tables per client requirement")

            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    # DATABASE STORAGE DISABLED per client requirement
    # def _create_client_tables(self):
    #     """Create additional tables for client requirements."""
    #     pass

    def get_session(self) -> Session:
        """Get a database session."""
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()

    # METHOD DISABLED per client requirement
    # def create_training_session(self,
    #                          exchange_filter: Optional[List[str]] = None,
    #                          symbol_filter: Optional[List[str]] = None,
    #                          interval_filter: Optional[List[str]] = None,
    #                          time_range: Optional[tuple] = None,
    #                          days_filter: Optional[int] = None,
    #                          notes: Optional[str] = None) -> str:
    #     """Create a new training session record."""
    #     logger.info("❌ Training session creation disabled per client requirement")
    #     return self.session_id

    # METHOD DISABLED per client requirement
    # def store_file(self,
    #               file_path: Union[str, Path],
    #               table_name: str,
    #               feature_type: str,
    #               description: Optional[str] = None) -> str:
    #     """Store file information in database and return the stored path."""
    #     logger.info("❌ File storage disabled per client requirement")
    #     return str(file_path)

    def store_model(self,
                   model: Any,
                   model_name: str,
                   feature_names: List[str],
                   hyperparams: Dict,
                   train_score: float = None,
                   val_score: float = None,
                   cv_scores: Optional[List[float]] = None,
                   model_version: str = 'spot') -> int:
        """Store trained model in database. Returns model ID."""
        db = self.get_session()

        try:
            # Serialize model
            model_data = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Store model dengan hanya kolom yang client izinkan
            model_storage = ModelStorage(
                model_name=model_name,
                model_version=model_version,  # 'spot' or 'futures'
                model_file=f"{model_name}_{timestamp}.pkl",
                model_data=model_data
            )

            db.add(model_storage)
            db.commit()

            logger.info(f"✅ Stored model: {model_name} (version: {model_version}, ID: {model_storage.id})")
            return model_storage.id

        except Exception as e:
            db.rollback()
            logger.error(f"❌ Failed to store model: {e}")
            raise
        finally:
            db.close()

    def store_dataset_summary(self,
                            summary_file: str,
                            summary_data: Optional[bytes] = None,
                            model_version: str = 'spot') -> int:
        """Store dataset summary to xgboost_dataset_summary table. Returns summary ID."""
        db = self.get_session()

        try:
            # Create new dataset summary record
            dataset_summary = DatasetSummary(
                model_version=model_version,  # 'spot' or 'futures'
                summary_file=summary_file,
                summary_data=summary_data  # Binary blob data
            )
            db.add(dataset_summary)
            db.commit()

            logger.info(f"✅ Stored dataset summary (version: {model_version}, ID: {dataset_summary.id})")
            return dataset_summary.id

        except Exception as e:
            db.rollback()
            logger.error(f"❌ Failed to store dataset summary: {e}")
            raise
        finally:
            db.close()

    # METHOD DISABLED per client requirement
# def update_session_status(self,
#                         status: str,
#                         metrics: Optional[Dict] = None,
#                         model_version: Optional[str] = None,
#                         notes: Optional[str] = None):
#     """Update training session status and metrics."""
#     logger.info("❌ Session status update disabled per client requirement")

# METHOD DISABLED per client requirement
# def store_evaluation_results(self,
#                            eval_type: str,
#                            metrics: Dict,
#                            eval_params: Optional[Dict] = None,
#                            detailed_results: Optional[Dict] = None) -> str:
#     """Store evaluation results in database."""
#     logger.info("❌ Evaluation results storage disabled per client requirement")

    def load_latest_model(self, model_version: str = 'spot') -> tuple:
        """Load latest model from database using created_at ordering."""
        db = self.get_session()

        try:
            # Get latest model by version using created_at DESC
            model_record = db.query(ModelStorage).filter(
                ModelStorage.model_version == model_version
            ).order_by(
                ModelStorage.created_at.desc()
            ).first()

            if model_record:
                model = pickle.loads(model_record.model_data)
                # feature_names tidak disimpan per client requirement
                feature_names = []
                return model, feature_names, model_record.id
            else:
                raise ValueError("No model found in database")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        finally:
            db.close()

    # METHOD DISABLED per client requirement
# def get_session_history(self, limit: int = 10) -> List[Dict]:
#     """Get history of training sessions."""
#     logger.info("❌ Session history retrieval disabled per client requirement")
#     return []

# METHOD DISABLED per client requirement
# def store_feature_data(self,
#                       file_path: Optional[Union[str, Path]] = None,
#                       table_name: Optional[str] = None,
#                       feature_type: Optional[str] = None,
#                       session_id: Optional[str] = None,
#                       data: Optional[Union[pd.DataFrame, dict]] = None,
#                       description: Optional[str] = None) -> str:
#     """Store feature data in database and return the stored path."""
#     logger.info("❌ Feature data storage disabled per client requirement")
#     return ""

# METHOD DISABLED per client requirement
# def cleanup_old_data(self, days: int = 30):
#     """Clean up old data from database."""
#     logger.info("❌ Data cleanup disabled per client requirement")