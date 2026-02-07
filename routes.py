#!/usr/bin/env python3
"""
API routes and model/dataset handlers.
Separated from app initialization to keep xgboost_api.py lightweight.
"""

import base64
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, Optional, List

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from database_storage import DatabaseStorage

MODEL_VERSIONS = [
    "spot",
    "spot_btc_binance",
    "futures",
    "futures17",
    "futures_new_gen",
    "futures_new_gen_eth",
    "futures_new_gen_btc_binance",
    "futures_new_gen_btc_bybit",
    "futures_new_gen_eth_bybit",
]


class ModelResponse(BaseModel):
    success: bool
    id: int
    model_name: str
    model_version: str
    model_file: str
    created_at: Optional[str]
    model_data_base64: str
    feature_names: List[str]
    content_type: str
    file_extension: str


class DatasetSummaryResponse(BaseModel):
    success: bool
    id: int
    model_version: str
    summary_file: str
    created_at: str
    summary_data_base64: Optional[str]
    content_type: str
    file_extension: str


class ModelInfo(BaseModel):
    id: int
    model_name: str
    model_version: Optional[str]
    created_at: str
    has_dataset_summary: bool


class ModelsResponse(BaseModel):
    success: bool
    total_models: int
    models: List[ModelInfo]


class ErrorResponse(BaseModel):
    error: str
    message: str


class ModelInsertRequest(BaseModel):
    model_name: str
    model_data_base64: str
    feature_names: List[str] = Field(default_factory=list)
    hyperparams: Dict = Field(default_factory=dict)
    train_score: Optional[float] = None
    val_score: Optional[float] = None
    cv_scores: Optional[List[float]] = None


class ModelInsertResponse(BaseModel):
    success: bool
    message: str
    model_id: int


def encode_to_base64(data: bytes) -> str:
    """Encode bytes to base64 string."""
    return base64.b64encode(data).decode("utf-8")


def read_file_as_base64(file_path: str) -> Optional[str]:
    """Read file and return as base64 string."""
    try:
        with open(file_path, "rb") as f:
            return encode_to_base64(f.read())
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to read file {file_path}: {e}")
        return None


class ModelService:
    def __init__(self, db_storage: Optional[DatabaseStorage], logger: logging.Logger):
        self.db_storage = db_storage
        self.logger = logger

    def _ensure_db(self) -> None:
        if not self.db_storage:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database not connected",
            )

    def _resolve_summary_base64(self, summary_record) -> Optional[str]:
        if summary_record.summary_data:
            self.logger.info(
                f"✅ Retrieved {summary_record.model_version} dataset summary blob from database"
            )
            return encode_to_base64(summary_record.summary_data)

        summary_file_path = summary_record.summary_file
        possible_paths = [
            f"./output_train/datasets/summary/{summary_file_path}",
            f"./output_train/{summary_file_path}",
            summary_file_path,
        ]
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(
                    f"✅ Retrieved {summary_record.model_version} dataset summary from file: {path}"
                )
                return read_file_as_base64(path)
        return None

    def get_latest_model_by_version(self, model_version: str) -> ModelResponse:
        self._ensure_db()
        db = self.db_storage.get_session()
        try:
            model_record = (
                db.query(self.db_storage.db_model)
                .filter(self.db_storage.db_model.model_version == model_version)
                .order_by(self.db_storage.db_model.created_at.desc())
                .first()
            )

            if not model_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No {model_version} model found",
                )

            model = pickle.loads(model_record.model_data)
            model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            model_base64 = encode_to_base64(model_bytes)

            response = ModelResponse(
                success=True,
                id=model_record.id,
                model_name=model_record.model_name,
                model_version=model_record.model_version,
                model_file=model_record.model_file,
                created_at=model_record.created_at.isoformat(),
                model_data_base64=model_base64,
                feature_names=[],
                content_type="application/octet-stream",
                file_extension=".joblib",
            )

            self.logger.info(
                f"✅ Retrieved latest {model_version} model (ID: {model_record.id})"
            )
            return response
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to get latest {model_version} model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve model: {str(e)}",
            )
        finally:
            db.close()

    def get_dataset_summary_by_version(self, model_version: str) -> DatasetSummaryResponse:
        self._ensure_db()
        db = self.db_storage.get_session()
        try:
            summary_record = (
                db.query(self.db_storage.db_dataset_summary)
                .filter(self.db_storage.db_dataset_summary.model_version == model_version)
                .order_by(self.db_storage.db_dataset_summary.created_at.desc())
                .first()
            )

            if not summary_record:
                return DatasetSummaryResponse(
                    success=False,
                    id=0,
                    model_version=model_version,
                    summary_file="",
                    created_at=datetime.utcnow().isoformat(),
                    summary_data_base64=None,
                    content_type="text/plain",
                    file_extension=".txt",
                )

            summary_base64 = self._resolve_summary_base64(summary_record)

            response = DatasetSummaryResponse(
                success=True,
                id=summary_record.id,
                model_version=summary_record.model_version,
                summary_file=summary_record.summary_file,
                created_at=summary_record.created_at.isoformat(),
                summary_data_base64=summary_base64,
                content_type="text/plain",
                file_extension=".txt",
            )

            self.logger.info(
                f"✅ Retrieved {model_version} dataset summary (ID: {summary_record.id})"
            )
            return response
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to get {model_version} dataset summary: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve dataset summary: {str(e)}",
            )
        finally:
            db.close()

    def get_summary_by_id_version(
        self, summary_id: int, model_version: str
    ) -> DatasetSummaryResponse:
        self._ensure_db()
        db = self.db_storage.get_session()
        try:
            summary_record = (
                db.query(self.db_storage.db_dataset_summary)
                .filter(
                    self.db_storage.db_dataset_summary.id == summary_id,
                    self.db_storage.db_dataset_summary.model_version == model_version,
                )
                .first()
            )

            if not summary_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No {model_version} dataset summary found with id: {summary_id}",
                )

            summary_base64 = self._resolve_summary_base64(summary_record)

            response = DatasetSummaryResponse(
                success=True,
                id=summary_record.id,
                model_version=summary_record.model_version,
                summary_file=summary_record.summary_file,
                created_at=summary_record.created_at.isoformat(),
                summary_data_base64=summary_base64,
                content_type="text/plain",
                file_extension=".txt",
            )

            self.logger.info(
                f"✅ Retrieved {model_version} dataset summary (ID: {summary_id})"
            )
            return response
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to get {model_version} dataset summary: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve dataset summary: {str(e)}",
            )
        finally:
            db.close()

    def get_model_by_id_version(self, model_id: int, model_version: str) -> ModelResponse:
        self._ensure_db()
        db = self.db_storage.get_session()
        try:
            model_record = (
                db.query(self.db_storage.db_model)
                .filter(
                    self.db_storage.db_model.id == model_id,
                    self.db_storage.db_model.model_version == model_version,
                )
                .first()
            )

            if not model_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No {model_version} model found with id: {model_id}",
                )

            model = pickle.loads(model_record.model_data)
            model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            model_base64 = encode_to_base64(model_bytes)

            response = ModelResponse(
                success=True,
                id=model_id,
                model_name=model_record.model_name,
                model_version=model_record.model_version,
                model_file=model_record.model_file,
                created_at=model_record.created_at.isoformat(),
                model_data_base64=model_base64,
                feature_names=[],
                content_type="application/octet-stream",
                file_extension=".joblib",
            )

            self.logger.info(f"✅ Retrieved {model_version} model (ID: {model_id})")
            return response
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(
                f"❌ Failed to get {model_version} model for id {model_id}: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve model: {str(e)}",
            )
        finally:
            db.close()

    def list_models_by_version(self, model_version: str) -> ModelsResponse:
        self._ensure_db()
        db = self.db_storage.get_session()
        try:
            models = (
                db.query(self.db_storage.db_model)
                .filter(self.db_storage.db_model.model_version == model_version)
                .order_by(self.db_storage.db_model.created_at.desc())
                .all()
            )

            summary_exists = (
                db.query(self.db_storage.db_dataset_summary)
                .filter(self.db_storage.db_dataset_summary.model_version == model_version)
                .first()
                is not None
            )

            model_list = [
                ModelInfo(
                    id=model.id,
                    model_name=model.model_name,
                    model_version=model.model_version,
                    created_at=model.created_at.isoformat(),
                    has_dataset_summary=summary_exists,
                )
                for model in models
            ]

            return ModelsResponse(
                success=True, total_models=len(model_list), models=model_list
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to list {model_version} models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve models: {str(e)}",
            )
        finally:
            db.close()

    def insert_model_by_version(
        self, model_request: ModelInsertRequest, model_version: str
    ) -> ModelInsertResponse:
        self._ensure_db()
        try:
            model_bytes = base64.b64decode(model_request.model_data_base64)
            model = pickle.loads(model_bytes)

            model_id = self.db_storage.store_model(
                model=model,
                model_name=model_request.model_name,
                feature_names=model_request.feature_names,
                hyperparams=model_request.hyperparams,
                train_score=model_request.train_score,
                val_score=model_request.val_score,
                cv_scores=model_request.cv_scores,
                model_version=model_version,
            )

            self.logger.info(
                f"✅ Successfully inserted {model_version} model: {model_request.model_name} (ID: {model_id})"
            )

            return ModelInsertResponse(
                success=True,
                message=f"Model inserted successfully (version: {model_version})",
                model_id=model_id,
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to insert {model_version} model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to insert model: {str(e)}",
            )


def register_model_routes(
    app: FastAPI,
    db_storage: Optional[DatabaseStorage],
    logger: logging.Logger,
    model_versions: Optional[List[str]] = None,
) -> None:
    service = ModelService(db_storage=db_storage, logger=logger)
    versions = model_versions or MODEL_VERSIONS

    for version in versions:
        router = APIRouter(prefix=f"/api/v1/{version}", tags=[version])

        @router.get(
            "/latest/model",
            response_model=ModelResponse,
            operation_id=f"{version}_latest_model",
        )
        async def get_latest_model(model_version=version):
            return service.get_latest_model_by_version(model_version)

        @router.get(
            "/latest/dataset-summary",
            response_model=DatasetSummaryResponse,
            operation_id=f"{version}_latest_dataset_summary",
        )
        async def get_latest_dataset_summary(model_version=version):
            return service.get_dataset_summary_by_version(model_version)

        @router.get(
            "/summary/{summary_id}",
            response_model=DatasetSummaryResponse,
            operation_id=f"{version}_summary_by_id",
        )
        async def get_summary_by_id(summary_id: int, model_version=version):
            return service.get_summary_by_id_version(summary_id, model_version)

        @router.get(
            "/model/{model_id}",
            response_model=ModelResponse,
            operation_id=f"{version}_model_by_id",
        )
        async def get_model_by_id(model_id: int, model_version=version):
            return service.get_model_by_id_version(model_id, model_version)

        @router.get(
            "/models",
            response_model=ModelsResponse,
            operation_id=f"{version}_models_list",
        )
        async def list_models(model_version=version):
            return service.list_models_by_version(model_version)

        @router.post(
            "/model",
            response_model=ModelInsertResponse,
            operation_id=f"{version}_insert_model",
        )
        async def insert_model(model_request: ModelInsertRequest, model_version=version):
            return service.insert_model_by_version(model_request, model_version)

        app.include_router(router)
