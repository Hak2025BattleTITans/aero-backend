import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import (APIRouter, Depends, File, Header, HTTPException, Query,
                     Response, UploadFile)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from api.auth import get_current_user
from database import get_redis_key, ensure_session_exists, set_file_key, get_redis
from models import Iframe as IFrameItem
from models import MainMetrics, ScheduleItem, SessionDoc, MetricPair

router = APIRouter(
    prefix="/files",
    tags=["files"],
    dependencies=[Depends(get_current_user)]
    )

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
CSV_CONTENT_TYPES = {"text/csv", "application/vnd.ms-excel"}  # browsers often use the latter

def _sanitize_filename(name: str) -> str:
    """
    Keep only safe chars and collapse spaces. Keep extension if present.
    """
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    # disallow hidden files or empty names
    if not name or name.startswith("."):
        name = f"file.csv"
    # force .csv extension
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return name

async def get_session_key(
    session_id: str = Header(..., alias="X-Session-Id"),
) -> str:
    """
    Возвращает готовый Redis ключ для данной сессии.
    """
    return get_redis_key(session_id)

@router.post("/upload")
async def upload_csv_file(
    response: Response,
    file: UploadFile = File(..., description="CSV file to upload"),
    as_name: Optional[str] = Query(default=None, description="Name to save the file as"),
    session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    redis: Redis = Depends(get_redis),
):
    # 1) Ensure / create session id + session JSON doc
    if not session_id:
        session_id = f"sess_{uuid.uuid4().hex}"
    response.headers["X-Session-Id"] = session_id

    # Make sure a SessionDoc exists so we can write .file_key into it
    await ensure_session_exists(redis, session_id)

    # 2) Validate file
    if file.content_type not in CSV_CONTENT_TYPES and not (file.filename or "").lower().endswith(".csv"):
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
                    break
                total += len(chunk)
                if total > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail="File too large (limit 50 MB)")
                await out.write(chunk)
    except HTTPException:
        if stored_path.exists():
            try:
                stored_path.unlink()
            except Exception:
                pass
        raise
    finally:
        await file.close()

    # 5) Save stored file name INSIDE the session JSON (`.file_key`)
    #    This is the key part you asked for.
    await set_file_key(redis, session_id, stored_name)

    download_url = f"/api/v1/files/download?session_id={session_id}&stored_name={stored_name}"
    return {
        "session_id": session_id,
        "file_name": safe_original,
        "stored_name": stored_name,
        "size_bytes": total,
        "download_url": download_url,
    }

@router.get("/download")
async def download_file(
    session_id: str = Query(...),
    stored_name: str = Query(...),
    redis: Redis = Depends(get_redis),
):
    """
    Simple download endpoint, checks that requested stored_name matches the one
    saved under the session's .file_key (prevents cross-session access).
    """
    # Verify the file is the one bound to this session
    from database import get_full  # local import to avoid circulars
    doc = await get_full(redis, session_id)
    if not doc or doc.file_key != stored_name:
        raise HTTPException(status_code=404, detail="File not found for this session")

    path = UPLOAD_DIR / stored_name
    if not path.exists():
        raise HTTPException(status_code=410, detail="File no longer available")

    # Use the original visible name from the stored_name suffix
    download_name = stored_name.split("_", 1)[1] if "_" in stored_name else "download.csv"
    return FileResponse(path=path, media_type="text/csv", filename=download_name)
