import logging
import uuid
from datetime import datetime, timedelta, timezone
from logging.config import dictConfig
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from dotenv import load_dotenv, find_dotenv

from api.auth import get_current_user
from database import get_redis
from logging_config import LOGGING_CONFIG, ColoredFormatter
from models import Iframe as IFrameItem
from models import MainMetrics, ScheduleItem


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


TEST_JSON = {
  "main_metrics": {
    "income": { "value": 12000, "optimized_value": 13500 },
    "passengers": { "value": 240, "optimized_value": 270 },
    "avg_check": { "value": 50, "optimized_value": 55 }
  },
  "unoptimized_schedule": [
    {
      "date": "2025-04-18",
      "flight_number": "224",
      "dep_airport": "SVO",
      "arr_airport": "SYX",
      "dep_time": "10:35",
      "arr_time": "01:15",
      "flight_capacity": 28,
      "lf_cabin": 0.6786,
      "cabins_brones": 19,
      "flight_type": "359",
      "cabin_code": "C",
      "pass_income": 10048.02,
      "passengers": 19
    },
    {
      "date": "2025-04-19",
      "flight_number": "225",
      "dep_airport": "SVO",
      "arr_airport": "HKT",
      "dep_time": "12:00",
      "arr_time": "23:15",
      "flight_capacity": 30,
      "lf_cabin": 0.7333,
      "cabins_brones": 22,
      "flight_type": "321",
      "cabin_code": "Y",
      "pass_income": 15000.50,
      "passengers": 22
    }
  ],
  "optimized_schedule": [
    {
      "date": "2025-04-18",
      "flight_number": "224",
      "dep_airport": "SVO",
      "arr_airport": "SYX",
      "dep_time": "10:35",
      "arr_time": "01:15",
      "flight_capacity": 24,
      "lf_cabin": 0.25,
      "cabins_brones": 6,
      "flight_type": "359",
      "cabin_code": "W",
      "pass_income": 2013.02,
      "passengers": 6
    },
    {
      "date": "2025-04-19",
      "flight_number": "225",
      "dep_airport": "SVO",
      "arr_airport": "HKT",
      "dep_time": "12:00",
      "arr_time": "23:15",
      "flight_capacity": 28,
      "lf_cabin": 0.85,
      "cabins_brones": 24,
      "flight_type": "321",
      "cabin_code": "Y",
      "pass_income": 17000.00,
      "passengers": 24
    }
  ],
  "iframes": [
    {
      "id": "frame_reports",
      "title": "Отчёт по маршрутам",
      "src": "https://datalens.yandex/z2uxl5pbztkep?shopid_vj2j=sp-15&shopid_vj2j=sp-18&shopid_vj2j=sp-20&_embedded=1&_no_controls=1&_theme=light&_lang=ru"
    },
    {
      "id": "frame_charts",
      "title": "Графики доходности",
      "src": "https://datalens.yandex/z2uxl5pbztkep?shopid_vj2j=sp-15&shopid_vj2j=sp-18&shopid_vj2j=sp-20&_embedded=1&_no_controls=1&_theme=light&_lang=ru"
    }
  ]
}

class SessionResponse(BaseModel):
    session_id: str

class SessionDataResponse(BaseModel):
    session_id: str = Field(..., alias="session_id")
    expires_at: Optional[str] = None
    main_metrics: MainMetrics
    unoptimized_schedule: List[ScheduleItem]
    optimized_schedule: List[ScheduleItem]
    iframes: List[IFrameItem]

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
    raw = TEST_JSON
    # try:
    #     raw: Dict[str, Any] | None = await redis.json().get(data_key)  # type: ignore[attr-defined]
    #     raw = TEST_JSON if raw is None else raw
    # except AttributeError:
    #     raw = None

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
    payload = {
        "session_id": x_session_id,
        "expires_at": expires_at,
        "main_metrics": raw.get("main_metrics", {}),
        "unoptimized_schedule": raw.get("unoptimized_schedule", []),
        "optimized_schedule": raw.get("optimized_schedule", []),
        "iframes": raw.get("iframes", []),
    }

    # Валидируем и возвращаем по Pydantic-схеме
    logger.debug(f"Session data payload: {payload}")
    return SessionDataResponse(**payload)
