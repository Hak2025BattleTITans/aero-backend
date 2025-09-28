import logging
import os
from logging.config import dictConfig
from pathlib import Path

from fastapi import HTTPException

from logging_config import LOGGING_CONFIG, ColoredFormatter

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setFormatter(ColoredFormatter('%(levelname)s:     %(asctime)s %(name)s - %(message)s'))


_raw_upload_dir = os.getenv("UPLOAD_DIR", "uploads").strip()
base = Path(__file__).resolve().parents[0]
UPLOAD_DIR = (Path(_raw_upload_dir) if Path(_raw_upload_dir).is_absolute()
              else (base / _raw_upload_dir)).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def resolve_csv_path(file_key: str) -> Path:
    """
    Безопасно резолвит путь к CSV на основе file_key из сессии.
    Поддерживает как абсолютный путь, так и просто имя файла.
    """

    logger.debug(f"Resolving CSV path for file_key: {file_key}")

    # Если в Redis лежит абсолютный путь – используем его.
    candidate = Path(file_key)
    if not candidate.is_absolute():
        logger.debug(f"File_key is not absolute: {file_key}")
        candidate = (UPLOAD_DIR / file_key).resolve()

    # Блокируем выход из UPLOAD_DIR, если путь не абсолютный в file_key
    try:
        # Если путь нельзя выразить как относительный к UPLOAD_DIR — ValueError
        candidate.relative_to(UPLOAD_DIR)
    except ValueError:
        error_log = (
            f"\nCandidate: {candidate}"
            f"\nUpload dir: {UPLOAD_DIR}"
            f"\nParents: {list(candidate.parents)}"
            f"\nIs absolute: {candidate.is_absolute()}"
        )

        logger.error(f"Attempt to access file outside of upload directory: {error_log}")
        raise HTTPException(status_code=400, detail="Invalid CSV path")

    if not candidate.exists():
        logger.error(f"CSV not found: {candidate}")
        raise HTTPException(status_code=404, detail="CSV not found")

    if candidate.suffix.lower() != ".csv":
        logger.error(f"Not a CSV file: {candidate.name}")
        raise HTTPException(status_code=400, detail="Only .csv files are allowed")


    return candidate