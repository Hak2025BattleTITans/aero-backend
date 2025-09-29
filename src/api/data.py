import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from logging.config import dictConfig
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import (APIRouter, Depends, File, Header, HTTPException, Query,
                     Response, UploadFile)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from api.auth import get_current_user
from csv_reader import AsyncCSVReader  # поправьте путь импорта под ваш проект
from database import (ensure_session_exists, get_full, get_redis,
                      get_redis_key, replace_unoptimized_schedule,
                      set_file_key, set_main_metrics)
from logging_config import LOGGING_CONFIG, ColoredFormatter
from models import Iframe as IFrameItem
from models import MainMetrics, MetricPair, ScheduleItem, SessionDoc
from api.optimize import _resolve_csv_path

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setFormatter(ColoredFormatter('%(levelname)s:     %(asctime)s %(name)s - %(message)s'))

router = APIRouter(
    prefix="/data",
    tags=["data"],
    dependencies=[Depends(get_current_user)]
    )

OPTIMIZED_DIR = Path(os.environ.get("OPTIMIZED_DIR", "optimized")).resolve()
OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

_raw_upload_dir = os.getenv("UPLOAD_DIR", "uploads").strip()

if Path(_raw_upload_dir).is_absolute():
    UPLOAD_DIR = Path(_raw_upload_dir)
else:
    # базируемся на текущем рабочем каталоге процесса (обычно WORKDIR в Docker = /app)
    app_root = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
    UPLOAD_DIR = (app_root / _raw_upload_dir).resolve()


UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/import-csv", summary="Прочитать CSV из file_key и сохранить в Redis")
async def import_csv_from_session_file(
    response: Response,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    redis: Redis = Depends(get_redis),
):
    """
    1) Берёт `file_key` из Redis-сессии (не спрашивает filename у клиента).
    2) Читает CSV через ваш `AsyncCSVReader`.
    3) Сохраняет распарсенный список `ScheduleItem` в `.unoptimized_schedule` сессии.
    4) Возвращает 200 без тела.
    """

    logger.info(f"Import CSV request for session_id: {x_session_id}")
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id")

    # Убедимся, что сессия существует (или создадим минимальную)
    session = await ensure_session_exists(redis, x_session_id)

    # Достанем file_key из сессии заново (на случай, если только что создали минимальную)
    session = await get_full(redis, x_session_id)
    if not session or not session.file_key:
        logger.error("No file_key in session")
        raise HTTPException(status_code=400, detail="No file_key in session")

    # Резолвим путь
    csv_path = _resolve_csv_path(session.file_key, OPTIMIZED_DIR)
    if not csv_path.exists() or not csv_path.is_file():
        logger.error(f"CSV file not found at path: {csv_path}")
        raise HTTPException(status_code=404, detail="CSV file not found")

    # Читаем CSV
    logger.debug(f"Reading CSV from path: {csv_path}")
    reader = AsyncCSVReader(str(csv_path), delimiter=";")
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

    response.status_code = 200
    return


