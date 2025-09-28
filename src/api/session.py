import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from logging.config import dictConfig
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from api.auth import get_current_user
from api.data import _resolve_csv_path
from csv_reader import AsyncCSVReader
from database import get_full, get_redis
from logging_config import LOGGING_CONFIG, ColoredFormatter
from models import Iframe as IFrameItem
from models import MainMetrics, ScheduleItem
from plotter import Plotter

env_path = find_dotenv(".env", usecwd=True)
load_dotenv(env_path, override=True, encoding="utf-8-sig")

router = APIRouter(
  prefix="/session",
  tags=["session"],
  dependencies=[Depends(get_current_user)]
  )

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setFormatter(ColoredFormatter('%(levelname)s:     %(asctime)s %(name)s - %(message)s'))


class SessionResponse(BaseModel):
    session_id: str

class SessionDataResponse(BaseModel):
    session_id: str = Field(..., alias="session_id")
    expires_at: Optional[str] = None
    main_metrics: MainMetrics
    unoptimized_schedule: List[ScheduleItem]
    optimized_schedule: List[ScheduleItem]
    plots: dict

def _expires_at_from_ttl(ttl_seconds: int) -> Optional[str]:
    """
    ttl_seconds:
      -2 -> ключ не существует
      -1 -> без истечения (None)
      >=0 -> вычислить ISO 8601 в UTC с суффиксом Z
    """

    if ttl_seconds < 0:
        return None if ttl_seconds == -1 else None
    dt = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    # ISO с 'Z' по примеру

    logger.debug(f"Computed expires_at: {dt.isoformat()} from TTL: {ttl_seconds}")
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

@router.get("/", response_model=SessionResponse)
async def get_session(
    response: Response,
    redis: Redis = Depends(get_redis),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    logger.info(f"Session request with X-Session-Id: {x_session_id}")
    # Если клиент не передал X-Session-Id — генерируем новый
    if not x_session_id:
        x_session_id = f"sess_{uuid.uuid4().hex}"

    # сохраняем в Redis (например с TTL 24 часа)
    await redis.setex(f"session:{x_session_id}", 60 * 60 * 24, "active")

    # Добавляем заголовок в ответ
    response.headers["X-Session-Id"] = x_session_id

    return SessionResponse(session_id=x_session_id)

async def generate_plots(redis: Redis, x_session_id: str) -> None:
    # Достанем file_key из сессии заново (на случай, если только что создали минимальную)
    session = await get_full(redis, x_session_id)
    if not session or not session.file_key:
        logger.error("No file_key in session")
        raise HTTPException(status_code=400, detail="No file_key in session")

    logger.debug(f"File key: {session.file_key}")

    csv_path = _resolve_csv_path(session.file_key)
    if not csv_path.exists() or not csv_path.is_file():
        logger.error(f"CSV file not found at path: {csv_path}")
        raise HTTPException(status_code=404, detail="CSV file not found")

    logger.debug(f"Reading CSV from path: {csv_path}")
    reader = AsyncCSVReader(str(csv_path), delimiter=";")
    try:
        items, passengers, income, avg_check = await reader.read()
    except Exception as e:
        logger.error("Error reading CSV", exc_info=True)
        raise HTTPException(status_code=500, detail=f"CSV read error: {e}")

    items = session.optimized_schedule

    raw_plotter = Plotter(csv_path)
    plots = [
        raw_plotter.avg_check(),
        raw_plotter.dyn_income(),
        raw_plotter.dyn_passenger()
    ]

    plotter = await Plotter.from_items(items, csv_path)
    optimized_plots = [
        plotter.avg_check(),
        plotter.dyn_income(),
        plotter.dyn_passenger()
    ]

    return {
        "plots": plots,
        "optimized_plots": optimized_plots
    }

@router.get("/data", response_model=SessionDataResponse, response_model_by_alias=True)
async def get_session_data(
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    redis: Redis = Depends(get_redis),
):
    logger.info(f"Session data request with X-Session-Id: {x_session_id}")
    # Проверяем наличие заголовка
    if not x_session_id:
        raise HTTPException(status_code=404, detail="Session not found")

    # Проверяем существование сессии и TTL основного ключа (создаётся в /session/me)
    session_key = f"session:{x_session_id}"
    ttl = await redis.ttl(session_key)
    if ttl == -2:
        # ключ не найден — считаем сессию отсутствующей
        logger.warning(f"Session key not found in Redis: {session_key}")
        raise HTTPException(status_code=404, detail="Session not found")

    # Достаём полноценные данные сессии из RedisJSON
    data_key = f"{session_key}:data"
    try:
        raw: Dict[str, Any] | None = await redis.json().get(data_key)  # type: ignore[attr-defined]
    except AttributeError:
        raw = None

    if not raw:
        # Нет объекта сессии с данными
        logger.warning(f"No session data found in Redis for key: {data_key}")
        raise HTTPException(status_code=404, detail="Session not found")

    # Формируем expires_at из TTL
    expires_at = _expires_at_from_ttl(ttl)
    logger.debug(f"Session {x_session_id} has TTL: {ttl}, expires_at: {expires_at}")

    # Склеиваем финальный ответ (не обрезая массивы)
    # Ожидается структура в RedisJSON:
    # {
    #   "main_metrics": {...},
    #   "unoptimized_schedule": [...],
    #   "optimized_schedule": [...],
    #   "iframes": [...]
    # }

    plots = await generate_plots(redis, x_session_id)


    payload = {
        "session_id": x_session_id,
        "expires_at": expires_at,
        "main_metrics": raw.get("main_metrics", {}),
        "unoptimized_schedule": raw.get("unoptimized_schedule", []),
        "optimized_schedule": raw.get("optimized_schedule", []),
        "plots": plots,
    }

    custom = {
        np.ndarray: lambda x: x.tolist(),
        np.generic: lambda x: x.item(),
        pd.Timestamp: lambda x: x.isoformat(),
    }

    content = jsonable_encoder(payload, custom_encoder=custom)

    # Валидируем и возвращаем по Pydantic-схеме
    return Response(content=json.dumps(content, ensure_ascii=False), media_type="application/json")
