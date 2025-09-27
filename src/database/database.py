import calendar
import json
import os
import time
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone
from typing import Generator, AsyncGenerator

from dotenv import load_dotenv, find_dotenv
from redis.asyncio import Redis

from models import Iframe, MainMetrics, MetricPair, ScheduleItem, SessionDoc

env_path = find_dotenv(".env", usecwd=True)
load_dotenv(env_path, override=True, encoding="utf-8-sig")

redis_url = os.environ.get(
    "REDIS_URI",
    "redis://localhost:6379/0"
)

print("Using REDIS_URL:", redis_url)

async def get_redis() -> AsyncGenerator[Redis, None]:
    """
    FastAPI dependency to get Redis client.

    Automatically closes the client when the request is done.
    """
    client = Redis.from_url(redis_url, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()

def get_redis_key(session_id: str) -> str:
    """
    Generate Redis key for session data

    Args:
        session_id (str): X-Session-ID

    Returns:
        str: Redis key
    """

    return f"session:{session_id}:data"

def exat_from_iso8601(iso: str) -> int:
    """
    Get unix timestamp from ISO8601 string
    "2025-09-27T12:00:00Z" -> unix ts

    Args:
        iso (str): ISO8601 string

    Returns:
        int: unix timestamp
    """
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp())

async def put_session(client: Redis, session_id: str, doc: SessionDoc):
    """
    Get session document

    Args:
        client (Redis): Redis client
        session_id (str): X-Session-ID
        doc (SessionDoc): Session document to store
    """

    key = get_redis_key(session_id)
    await client.json().set(key, "$", doc.model_dump())
    await client.expireat(key, exat_from_iso8601(doc.expires_at))

async def set_main_metrics(client: Redis, session_id: str, main_metrics: MainMetrics):
    """
    Set main metrics for a session

    Args:
        client (Redis): Redis client
        session_id (str): X-Session-ID
        main_metrics (MainMetrics): Main metrics to store
    """
    key = get_redis_key(session_id)
    await client.json().set(key, ".main_metrics", main_metrics.model_dump())

async def update_income(client: Redis, session_id: str, value: int | None = None, optimized_value: int | None = None):
    """
    Update income for a session

    Args:
        client (Redis): Redis client
        session_id (str): X-Session-ID
        value (int|None): income value
        optimized_value (int|None): optimized income value

    Returns:
        None
    """
    key = get_redis_key(session_id)
    income = await client.json().get(key, ".main_metrics.income") or {}

    if value is None:
        value = income.get("value")
    if optimized_value is None:
        optimized_value = income.get("optimized_value")

    await client.json().set(key, ".main_metrics.income", {
        "value": value,
        "optimized_value": optimized_value
    })


async def update_passengers(client: Redis, session_id: str, value: int | None = None, optimized_value: int | None = None):
    """
    Update passengers count for a session

    Args:
        client (Redis): Redis client
        session_id (str): X-Session-ID
        value (int|None): passengers value
        optimized_value (int|None): optimized passengers value

    Returns:
        None
    """
    key = get_redis_key(session_id)
    passengers = await client.json().get(key, ".main_metrics.passengers") or {}

    if value is None:
        value = passengers.get("value")
    if optimized_value is None:
        optimized_value = passengers.get("optimized_value")

    await client.json().set(key, ".main_metrics.passengers", {
        "value": value,
        "optimized_value": optimized_value
    })


async def update_avg(client: Redis, session_id: str, value: int | None = None, optimized_value: int | None = None):
    """
    Update average check value for a session

    Args:
        client (Redis): Redis client
        session_id (str): X-Session-ID
        value (int|None): average check value
        optimized_value (int|None): optimized average check value

    Returns:
        None
    """
    key = get_redis_key(session_id)
    avg_check = await client.json().get(key, ".main_metrics.avg_check") or {}

    if value is None:
        value = avg_check.get("value")
    if optimized_value is None:
        optimized_value = avg_check.get("optimized_value")

    await client.json().set(key, ".main_metrics.avg_check", {
        "value": value,
        "optimized_value": optimized_value
    })

async def replace_unoptimized_schedule(client: Redis, session_id: str, items: list[ScheduleItem]):
    """
    Replace unoptimized schedule for a session.

    Args:
        client (Redis): Redis client
        session_id (str): X-Session-ID
        items (list[ScheduleItem]): List of schedule items
    """
    key = get_redis_key(session_id)
    await client.json().set(key, ".unoptimized_schedule", [i.model_dump() for i in items])


async def replace_optimized_schedule(client: Redis, session_id: str, items: list[ScheduleItem]):
    """
    Replace optimized schedule for a session.

    Args:
        client (Redis): Redis client
        session_id (str): X-Session-ID
        items (list[ScheduleItem]): List of schedule items
    """
    key = get_redis_key(session_id)
    await client.json().set(key, ".optimized_schedule", [i.model_dump() for i in items])


async def set_iframes(client: Redis, session_id: str, frames: list[Iframe]):
    """
    Set iframes for a session.

    Args:
        client (Redis): Redis client
        session_id (str): X-Session-ID
        frames (list[Iframe]): List of iframe objects
    """
    key = get_redis_key(session_id)
    await client.json().set(key, ".iframes", [f.model_dump() for f in frames])

async def get_full(client: Redis, session_id: str) -> SessionDoc | None:
    """
    Retrieve the full session document from Redis.

    Args:
        client (Redis): Redis client
        session_id (str): X-Session-ID

    Returns:
        SessionDoc | None: The full session document, or None if not found.
    """
    key = get_redis_key(session_id)
    raw = await client.json().get(key, "$")

    if not raw:
        return None

    # RedisJSON возвращает список, если используется путь "$"
    # Берем первый элемент
    data = raw[0] if isinstance(raw, list) else raw

    return SessionDoc.model_validate(data)


async def get_main_metrics_only(client: Redis, session_id: str) -> MainMetrics | None:
    """
    Retrieve only the main_metrics section from the session.

    Args:
        client (Redis): Redis client
        session_id (str): X-Session-ID

    Returns:
        MainMetrics | None: The main metrics data, or None if missing.
    """
    key = get_redis_key(session_id)
    raw = await client.json().get(key, ".main_metrics")

    if not raw:
        return None

    return MainMetrics.model_validate(raw)

async def sync_ttl_with_expires_at(client: Redis, session_id: str):
    """
    Sync Redis key TTL with the 'expires_at' field from the session document.

    Args:
        client (Redis): Redis client
        session_id (str): X-Session-ID
    """
    key = get_redis_key(session_id)

    # Получаем expires_at напрямую через RedisJSON
    expires_at = await client.json().get(key, ".expires_at")

    if expires_at:
        exat = exat_from_iso8601(expires_at)
        await client.expireat(key, exat)

