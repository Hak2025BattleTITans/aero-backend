# Setup logger
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from logging.config import dictConfig
from pathlib import Path

import pandas as pd
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import (APIRouter, Depends, File, Header, HTTPException, Query,
                     Response, UploadFile)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis

import helpers
from api.auth import get_current_user
from api.data import _resolve_csv_path
from csv_reader import AsyncCSVReader
from database import (ensure_session_exists, get_full, get_redis,
                      get_redis_key, replace_unoptimized_schedule,
                      set_file_key, set_main_metrics)
from logging_config import LOGGING_CONFIG, ColoredFormatter
from models import Iframe as IFrameItem
from models import MainMetrics, MetricPair, ScheduleItem, SessionDoc
from olap import OLAPBuilder
from api.optimize import _make_outfile
from optimizer import Optimizer

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setFormatter(ColoredFormatter('%(levelname)s:     %(asctime)s %(name)s - %(message)s'))


router = APIRouter(
    prefix="/files",
    tags=["files"],
    dependencies=[Depends(get_current_user)]
    )

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OPTIMIZED_DIR = Path(os.environ.get("OPTIMIZED_DIR", "optimized")).resolve()
OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
CSV_CONTENT_TYPES = {"text/csv", "application/vnd.ms-excel"}  # browsers often use the latter

def _sanitize_filename(name: str) -> str:
    """
    Keep only safe chars and collapse spaces. Keep extension if present.
    """

    logger.debug(f"Sanitizing filename: {name}")
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    # disallow hidden files or empty names
    if not name or name.startswith("."):
        name = f"file.csv"
    # force .csv extension
    if not name.lower().endswith(".csv"):
        name += ".csv"

    logger.debug(f"Sanitized filename: {name}")
    return name

async def _read_csv(redis, x_session_id: str, file_path: UploadFile) -> bytes:
    logger.info(f"Reading CSV for session_id: {x_session_id}")


    # Читаем CSV
    reader = AsyncCSVReader(str(file_path), delimiter=";")
    try:
        items, passengers, income, avg_check = await reader.read()
    except Exception as e:
        logger.error("Error reading CSV", exc_info=True)
        raise HTTPException(status_code=500, detail=f"CSV read error: {e}")

    # Сохраняем в Redis как unoptimized_schedule
    logger.debug(f"Saving {len(items)} schedule items to Redis")
    try:
        await replace_unoptimized_schedule(redis, x_session_id, items)
    except Exception as e:
        logger.error("Error saving schedule to Redis", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Redis save error: {e}")

    # Обновляем основные метрики
    main_metrics = MainMetrics(
        passengers=MetricPair(value=passengers, optimized_value=0),
        income=MetricPair(value=income, optimized_value=0),
        avg_check=MetricPair(value=avg_check, optimized_value=0),
    )
    logger.debug(f"Updating main metrics: {main_metrics}")
    try:
        await set_main_metrics(redis, x_session_id, main_metrics)
    except Exception as e:
        logger.error("Error updating main metrics in Redis", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Redis metrics update error: {e}")

    return items

async def _build_olap(redis, x_session_id: str, items: List[ScheduleItem]):
    # builder = OLAPBuilder()

    # builder.create_olap_from_raw(x_session_id, items)
    pass


@router.post("/upload")
async def upload_csv_file(
    response: Response,
    file: UploadFile = File(..., description="CSV file to upload"),
    as_name: Optional[str] = Query(default=None, description="Name to save the file as"),
    session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    redis: Redis = Depends(get_redis),
):
    logger.info(f"Uploading file: {file.filename}, session_id: {session_id}")
    # 1) Ensure / create session id + session JSON doc
    if not session_id:
        logger.warning("No X-Session-Id header found, generating new session id")
        session_id = f"sess_{uuid.uuid4().hex}"
    response.headers["X-Session-Id"] = session_id

    # Make sure a SessionDoc exists so we can write .file_key into it
    await ensure_session_exists(redis, session_id)

    # 2) Validate file
    if file.content_type not in CSV_CONTENT_TYPES and not file.filename.lower().endswith(".csv"):
        logger.warning(f"Unsupported file type: {file.content_type}")
        raise HTTPException(status_code=415, detail="Only CSV files are allowed")

    original_name = file.filename or "upload.csv"
    safe_original = _sanitize_filename(original_name)
    target_display_name = _sanitize_filename(as_name) if as_name else safe_original

    # 3) Unique stored name + path
    stored_name = f"{uuid.uuid4().hex}_{target_display_name}"
    stored_path = UPLOAD_DIR / stored_name

    # 4) Stream upload with size cap
    total = 0
    try:
        async with aiofiles.open(stored_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB
                if not chunk:
                    logger.debug("File upload complete")
                    break
                total += len(chunk)
                if total > MAX_FILE_SIZE_BYTES:
                    logger.warning("File too large")
                    raise HTTPException(status_code=413, detail="File too large (limit 50 MB)")
                await out.write(chunk)
    except HTTPException:
        if stored_path.exists():
            try:
                stored_path.unlink()
            except Exception:
                logger.error("Failed to delete oversized file", exc_info=True)
                pass
        raise
    finally:
        logger.debug("Closing uploaded file")
        await file.close()

    # 5) Save stored file name INSIDE the session JSON (`.file_key`)
    #    This is the key part you asked for.


    # 5.1 Генерация номеров рейсов ("№") до чтения CSV
    stored_path = helpers.add_number_row(
        session_id=session_id,
        stored_path=stored_path,
        suffix=f"with_numbers_{uuid.uuid4().hex}"
    )

    out_over = _make_outfile(f"sess_{session_id}")
    logger.debug(f"Optimizing CSV: {stored_path} -> {out_over}")
    optimizer = Optimizer()
    try:
        raw_input_df = pd.read_csv(stored_path, delimiter=',')
        if raw_input_df.shape[1] == 1:
            raw_input_df = pd.read_csv(stored_path, delimiter=';')
        prepared_df = optimizer.universal_data_preparator(raw_input_df)
        if prepared_df is None:
            raise RuntimeError("Failed to prepare input data")
        prepared_df.to_csv(out_over, index=False, sep=';', encoding='utf-8-sig')
    except Exception as e:
        logger.error("Error optimizing CSV", exc_info=True)
        raise HTTPException(status_code=500, detail=f"CSV optimization error: {e}")


    stored_name = f"{out_over}"
    stored_name = stored_name.split("/")[-1]
    logger.debug(f"Saving file key: {stored_name}")
    await set_file_key(redis, session_id, stored_name)

    # Read CSV from file and store parsed data in Redis


    items = await _read_csv(redis, session_id, out_over)

    # Build olap data asynchronously (fire and forget)
    await _build_olap(redis, session_id, items)

    download_url = f"/api/v1/files/download?session_id={session_id}&stored_name={stored_name}"
    logger.info(f"File uploaded successfully: {stored_name} ({total} bytes)")
    return {
        "session_id": session_id,
        "file_name": safe_original,
        "stored_name": stored_name,
        "size_bytes": total,
        "download_url": download_url,
    }

@router.get("/download", summary="Скачать CSV, привязанный к сессии")
async def download_file(
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    redis: Redis = Depends(get_redis),
):
    """
    Возвращает файл, привязанный к сессии через `.file_key`.
    Путь к файлу резолвится через `_resolve_csv_path`, что предотвращает доступ вне `UPLOAD_DIR`.
    """
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id")

    logger.info(f"Downloading file for session_id: {x_session_id}")
    session = await get_full(redis, x_session_id)
    if not session or not session.file_key:
        logger.error("No file_key in session")
        raise HTTPException(status_code=400, detail="No file_key in session")

    csv_path = _resolve_csv_path(session.file_key)
    if not csv_path.exists() or not csv_path.is_file():
        logger.error(f"CSV file not found at path: {csv_path}")
        # Файл был привязан, но исчез с диска — корректнее 410 Gone
        raise HTTPException(status_code=410, detail="File no longer available")

    # Имя для скачивания: если file_key имеет шаблон "<stored>_<original>",
    # отдадим часть после первого "_", иначе используем имя файла.
    stored_name = csv_path.name
    download_name = stored_name.split("_", 1)[1] if "_" in stored_name else stored_name

    logger.info(f"Serving file {csv_path} as {download_name}")
    return FileResponse(path=str(csv_path), media_type="text/csv", filename=download_name)
