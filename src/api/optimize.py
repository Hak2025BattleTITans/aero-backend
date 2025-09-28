# api/v1/optimize.py
import logging
import os
from logging.config import dictConfig
from pathlib import Path
from typing import Literal, Optional

from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from api.auth import get_current_user
from csv_reader import AsyncCSVReader
from database import get_full  # session reader
from database import \
    replace_optimized_schedule  # write optimized list[ScheduleItem]
from database import set_main_metrics  # write MainMetrics
from database import get_redis
from logging_config import LOGGING_CONFIG, ColoredFormatter
from models import MainMetrics, MetricPair, ScheduleItem
from optimizer import Optimizer

# ------------------------------------------------------------------------------
# Environment & paths
# ------------------------------------------------------------------------------
env_path = find_dotenv(".env", usecwd=True)
load_dotenv(env_path, override=True, encoding="utf-8-sig")

OPTIMIZED_DIR = Path(os.environ.get("OPTIMIZED_DIR", "optimized")).resolve()
OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Router & logging
# ------------------------------------------------------------------------------
router = APIRouter(
    prefix="/api/v1/optimize",
    tags=["optimize"],
    dependencies=[Depends(get_current_user)],
)

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
for h in logging.getLogger().handlers:
    if isinstance(h, logging.StreamHandler):
        h.setFormatter(
            ColoredFormatter("%(levelname)s:     %(asctime)s %(name)s - %(message)s")
        )

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _resolve_csv_path(file_key: str) -> Path:
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


def _make_outfile(stem: str) -> Path:
    """Create a unique output file path under OPTIMIZED_DIR."""
    # Example: stem='sess_xxx__overbooking' -> 'sess_xxx__overbooking__1738139123.csv'
    from time import time

    outfile = OPTIMIZED_DIR / f"{stem}__{int(time())}.csv"
    # Defensive: ensure parent exists
    outfile.parent.mkdir(parents=True, exist_ok=True)
    return outfile


# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------
class OptimizeRequest(BaseModel):
    ranking: bool = Field(False, description="Включить оптимизацию слотов (ranking)")
    overbooking: bool = Field(False, description="Включить оптимизацию овербукинга")

class OptimizeResponse(BaseModel):
    session_id: str
    applied_steps: list[Literal["ranking", "overbooking"]]
    status: Literal["success"]
    optimized_rows: int
    metrics: dict


# ------------------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------------------
@router.post("", response_model=OptimizeResponse)
async def optimize_dataset(
    payload: OptimizeRequest,
    redis: Redis = Depends(get_redis),
    x_session_id: str = Header(..., alias="X-Session-Id"),
):
    """
    Запускает оптимизацию. Если заданы оба флага — порядок: overbooking -> ranking.
    Результат (items + метрики) сохраняется в Redis.
    """
    # 1) Проверка входных флагов
    if not payload.ranking and not payload.overbooking:
        raise HTTPException(
            status_code=400,
            detail="At least one optimization switch (ranking or overbooking) must be true",
        )

    # 2) Получаем сессию и базовые метрики
    session = await get_full(redis, x_session_id)
    if not session or not getattr(session, "file_key", None):
        logger.error("No file_key in session")
        raise HTTPException(status_code=400, detail="No file_key in session")

    # Старые (baseline) метрики: passengers, income, avg_check
    try:
        passengers_before: Optional[float] = session.main_metrics.passengers.value
        income_before: Optional[float] = session.main_metrics.income.value
        avg_check_before: Optional[float] = session.main_metrics.avg_check.value
    except Exception:
        # Если по каким-то причинам нет метрик в сессии — считаем нулями
        passengers_before = 0
        income_before = 0
        avg_check_before = 0

    # 3) Разрешаем путь к исходному CSV
    input_csv = _resolve_csv_path(session.file_key)

    # 3.1) Генерация номеров рейсов ("№") до оптимизации
    try:
        import pandas as pd

        df = pd.read_csv(input_csv, delimiter=';')

        required = ['Номер рейса', 'Дата вылета', 'Время вылета', 'Код кабины']
        missing = [c for c in required if c not in df.columns]
        if missing:
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
        input_with_numbers = _make_outfile(f"{x_session_id}__with_numbers").with_suffix(".csv")
        df.to_csv(input_with_numbers, index=False, sep=';')
        input_csv = input_with_numbers

        logger.info(
            f"Flight numbers generated: {len(unique_groups)} unique groups. "
            f"Output CSV: {input_with_numbers}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate flight numbers", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Flight number generation error: {e}")

    # 4) Готовим пути вывода
    # Порядок: overbooking -> (intermediate) -> ranking -> final
    applied_steps: list[str] = []
    optimizer = Optimizer()

    # Начинаем с входного файла, далее будем перезаписывать ссылку current_in
    current_in = input_csv
    final_out: Path | None = None

    try:
        if payload.overbooking:
            applied_steps.append("overbooking")
            out_over = _make_outfile(f"{x_session_id}__overbooking")
            logger.info(f"Running overbooking optimization: {current_in} -> {out_over}")
            optimizer.overbooking_optimization(str(current_in), str(out_over))
            current_in = out_over
            final_out = out_over

        if payload.ranking:
            applied_steps.append("ranking")
            out_rank = _make_outfile(f"{x_session_id}__ranking")
            logger.info(f"Running ranking optimization: {current_in} -> {out_rank}")
            optimizer.ranking_optimization(str(current_in), str(out_rank))
            current_in = out_rank
            final_out = out_rank

        # Если сработала только одна оптимизация — final_out уже на неё указывает.
        if final_out is None:
            # Теоретически недостижимо, т.к. проверили флаги выше
            raise RuntimeError("No optimization step produced an output file")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Optimization failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization error: {e}")

    # 5) Сформировать отчёт (необязательно, если требуется — раскомментируйте)
    try:
        report_name = _make_outfile(f"{x_session_id}__report").with_suffix(".html")
        # Тип может быть 'ranking', 'overbooking' или 'combo'
        opt_type = (
            "combo"
            if payload.ranking and payload.overbooking
            else "ranking" if payload.ranking
            else "overbooking"
        )
        optimizer.form_report(
            str(input_csv),
            str(final_out),
            str(report_name),
            optimization_type=opt_type,
        )
        logger.info(f"Report generated at: {report_name}")
    except Exception as e:
        # Отчёт не критичен — логируем, но не падаем
        logger.warning(f"Failed to generate report: {e}")

    # 6) Читаем результат, сохраняем расписание + метрики в Redis
    try:
        reader = AsyncCSVReader(str(final_out), delimiter=";")
        items: list[ScheduleItem]
        passengers_after: float
        income_after: float
        avg_check_after: float
        items, passengers_after, income_after, avg_check_after = await reader.read_after_ranking()
    except Exception as e:
        logger.error("Error reading optimized CSV", exc_info=True)
        raise HTTPException(status_code=500, detail=f"CSV read error: {e}")

    try:
        logger.debug(f"Saving {len(items)} schedule items to Redis (session={x_session_id})")
        await replace_optimized_schedule(redis, x_session_id, items)
    except Exception as e:
        logger.error("Error saving schedule to Redis", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Redis save error: {e}")

    # Обновляем основные метрики
    try:
        main_metrics = MainMetrics(
            passengers=MetricPair(
                value=passengers_before or 0, optimized_value=passengers_after
            ),
            income=MetricPair(
                value=income_before or 0, optimized_value=income_after
            ),
            avg_check=MetricPair(
                value=avg_check_before or 0, optimized_value=avg_check_after
            ),
        )
        logger.debug(f"Updating main metrics: {main_metrics}")
        await set_main_metrics(redis, x_session_id, main_metrics)
    except Exception as e:
        logger.error("Error updating main metrics in Redis", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Redis metrics update error: {e}")

    # 7) Ответ
    return OptimizeResponse(
        session_id=x_session_id,
        applied_steps=applied_steps,
        status="success",
        optimized_rows=len(items),
        metrics={
            "passengers_before": passengers_before,
            "passengers_after": passengers_after,
            "delta_passengers": (passengers_after - (passengers_before or 0)),
            "income_before": income_before,
            "income_after": income_after,
            "delta_income": (income_after - (income_before or 0)),
            "avg_check_before": avg_check_before,
            "avg_check_after": avg_check_after,
            "delta_avg_check": (avg_check_after - (avg_check_before or 0)),
        },
    )