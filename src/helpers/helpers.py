import logging
import os
from logging.config import dictConfig
from pathlib import Path
import uuid

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

OPTIMIZED_DIR = Path(os.environ.get("OPTIMIZED_DIR", "optimized")).resolve()
OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

def make_outfile(stem: str) -> Path:
    """Create a unique output file path under OPTIMIZED_DIR."""
    # Example: stem='sess_xxx__overbooking' -> 'sess_xxx__overbooking__1738139123.csv'
    from time import time

    outfile = OPTIMIZED_DIR / f"{stem}__{int(time())}.csv"
    # Defensive: ensure parent exists
    outfile.parent.mkdir(parents=True, exist_ok=True)
    return outfile

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

def add_number_row(session_id: str, stored_path: Path, suffix: str) -> None:
    logger.debug(f"Adding number row for {stored_path} in session {session_id}")
    try:
        import pandas as pd

        df = pd.read_csv(stored_path, delimiter=';')

        required = ['Номер рейса', 'Дата вылета', 'Время вылета', 'Код кабины']
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error(f"Input CSV is missing required columns: {stored_path}")
            raise HTTPException(
                status_code=400,
                detail=f"Input CSV is missing required columns: {missing}",
            )

        # Формируем стабильные группы (порядок появления сохраняется)
        df['__group'] = (
            df['Номер рейса'].astype(str)
            + '_'
            + df['Дата вылета'].astype(str)
            + '_'
            + df['Время вылета'].astype(str)
        )
        unique_groups = df['__group'].drop_duplicates().tolist()
        group_mapping = {group: i + 1 for i, group in enumerate(unique_groups)}

        # Конструируем "№" и переносим его в первый столбец
        df['№'] = df['__group'].map(group_mapping).astype(str) + '-' + df['Код кабины'].astype(str)
        df = df.drop(columns='__group')
        df = df[['№'] + [c for c in df.columns if c != '№']]

        # Сохраняем во временный файл и используем его как входной для оптимизаций
        input_with_numbers = make_outfile(f"{session_id}__{suffix}").with_suffix(".csv")
        df.to_csv(input_with_numbers, index=False, sep=';')
        stored_path = input_with_numbers

        logger.info(
            f"Flight numbers generated: {len(unique_groups)} unique groups. "
            f"Output CSV: {input_with_numbers}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate flight numbers", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Flight number generation error: {e}")

    return stored_path
