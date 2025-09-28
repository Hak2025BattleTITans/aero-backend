import logging
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from logging.config import dictConfig
from typing import Iterable, Optional

import psycopg2
import psycopg2.extras
# -----------------------------------------------------------------------------
# ENV / DSN
# -----------------------------------------------------------------------------
from dotenv import find_dotenv, load_dotenv
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, make_dsn

from logging_config import LOGGING_CONFIG, ColoredFormatter
from models import ScheduleItem

env_path = find_dotenv(".env", usecwd=True)
load_dotenv(env_path, override=True, encoding="utf-8-sig")
POSTGRES_DSN = os.getenv("POSTGRES_DSN")
if not POSTGRES_DSN:
    raise RuntimeError("POSTGRES_DSN is not set in .env")

# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setFormatter(ColoredFormatter('%(levelname)s:     %(asctime)s %(name)s - %(message)s'))

# -----------------------------------------------------------------------------
# SQL: DDL
# -----------------------------------------------------------------------------
DDL_SQL = """
CREATE TABLE IF NOT EXISTS flight_numbers (
    id SERIAL PRIMARY KEY,
    flight_no VARCHAR(10) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS airports (
    id SERIAL PRIMARY KEY,
    airport_code VARCHAR(3) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS aircrafts (
    id SERIAL PRIMARY KEY,
    aircraft_type VARCHAR(50) NOT NULL,
    cabin_code VARCHAR(10) NOT NULL,
    cabin_capacity INTEGER NOT NULL CHECK (cabin_capacity > 0)
);

CREATE TABLE IF NOT EXISTS flight_loads (
    id SERIAL PRIMARY KEY,
    departure_date DATE NOT NULL,
    departure_time TIME NOT NULL,
    arrival_time TIME NOT NULL,
    cabin_load_factor NUMERIC(5, 2),
    bookings INTEGER NOT NULL,
    revenue NUMERIC(12, 2) NOT NULL,
    passengers INTEGER NOT NULL,
    flight_number_id INTEGER NOT NULL REFERENCES flight_numbers(id),
    departure_airport_id INTEGER NOT NULL REFERENCES airports(id),
    arrival_airport_id INTEGER NOT NULL REFERENCES airports(id),
    aircraft_id INTEGER NOT NULL REFERENCES aircrafts(id),
    CONSTRAINT chk_different_airports CHECK (departure_airport_id <> arrival_airport_id)
);

CREATE TABLE IF NOT EXISTS flight_loads_updated (
    id SERIAL PRIMARY KEY,
    departure_date DATE NOT NULL,
    departure_time TIME NOT NULL,
    arrival_time TIME NOT NULL,
    cabin_load_factor NUMERIC(5, 2),
    bookings INTEGER NOT NULL,
    revenue NUMERIC(12, 2) NOT NULL,
    passengers INTEGER NOT NULL,
    flight_number_id INTEGER NOT NULL REFERENCES flight_numbers(id),
    departure_airport_id INTEGER NOT NULL REFERENCES airports(id),
    arrival_airport_id INTEGER NOT NULL REFERENCES airports(id),
    aircraft_id INTEGER NOT NULL REFERENCES aircrafts(id),
    CONSTRAINT chk_different_airports_updated CHECK (departure_airport_id <> arrival_airport_id)
);

CREATE INDEX IF NOT EXISTS idx_flight_loads_flight_number ON flight_loads(flight_number_id);
CREATE INDEX IF NOT EXISTS idx_flight_loads_departure_airport ON flight_loads(departure_airport_id);
CREATE INDEX IF NOT EXISTS idx_flight_loads_arrival_airport ON flight_loads(arrival_airport_id);
CREATE INDEX IF NOT EXISTS idx_flight_loads_aircraft ON flight_loads(aircraft_id);

CREATE INDEX IF NOT EXISTS idx_flight_loads_updated_flight_number ON flight_loads_updated(flight_number_id);
CREATE INDEX IF NOT EXISTS idx_flight_loads_updated_departure_airport ON flight_loads_updated(departure_airport_id);
CREATE INDEX IF NOT EXISTS idx_flight_loads_updated_arrival_airport ON flight_loads_updated(arrival_airport_id);
CREATE INDEX IF NOT EXISTS idx_flight_loads_updated_aircraft ON flight_loads_updated(aircraft_id);
"""


# -----------------------------------------------------------------------------
# OLAP Helper
# -----------------------------------------------------------------------------
def _sanitize_db_name(x_session_id: str) -> str:
    """
    Безопасно формируем имя базы по X-Session-Id:
    - только латиница, цифры и нижние подчёркивания
    - префикс sess_
    """
    logger.debug(f"Sanitizing DB name from X-Session-Id: {x_session_id}")
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", x_session_id).strip("_").lower()
    if not base:
        base = uuid.uuid4().hex
    name = f"sess_{base}"
    # Ограничение длины имени базы в PostgreSQL: 63 байта
    logger.debug(f"Sanitized DB name: {name}")
    return name[:63]


def _as_admin_dsn(dsn: str, dbname: str = "postgres") -> str:
    """Меняем dbname в DSN на служебную postgres для операций CREATE/DROP DATABASE."""
    logger.debug(f"Converting DSN to admin DSN: {dsn}")
    return make_dsn(dsn, dbname=dbname)


@contextmanager
def _connect(dsn: str):
    logger.debug(f"Connecting to DB with DSN: {dsn}")
    conn = psycopg2.connect(dsn)
    try:
        yield conn
    finally:
        conn.close()


class OLAPBuilder:
    def __init__(self, base_dsn: str = POSTGRES_DSN):
        logger.info(f"OLAPBuilder initialized with base DSN: {base_dsn}")
        self.base_dsn = base_dsn

    # ---------------------------- DB lifecycle ---------------------------- #
    def _ensure_db(self, x_session_id: str) -> str:
        logger.debug(f"Ensuring DB for session: {x_session_id}")
        dbname = _sanitize_db_name(x_session_id)
        admin_dsn = _as_admin_dsn(self.base_dsn, "postgres")
        with _connect(admin_dsn) as conn:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,))
                exists = cur.fetchone() is not None
                if not exists:
                    cur.execute(f'CREATE DATABASE "{dbname}"')
        # Создать структуру, если новая / или убедиться, что она есть
        with _connect(_as_admin_dsn(self.base_dsn, dbname)) as conn:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(DDL_SQL)
            conn.commit()
        return dbname

    def _drop_db(self, dbname: str):
        admin_dsn = _as_admin_dsn(self.base_dsn, "postgres")
        with _connect(admin_dsn) as conn:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                # Завершаем активные коннекты к базе
                cur.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s
                      AND pid <> pg_backend_pid()
                    """,
                    (dbname,),
                )
                cur.execute(f'DROP DATABASE IF EXISTS "{dbname}"')

    def delete_session(self, x_session_id: str):
        """Удалить БД, связанную с сессией."""
        dbname = _sanitize_db_name(x_session_id)
        self._drop_db(dbname)

    # ---------------------------- Public API ----------------------------- #
    def create_olap_from_raw(self, x_session_id: str, raw_data: Iterable[ScheduleItem]) -> None:
        """
        Создаёт БД для сессии и загружает сырые данные в flight_loads.
        flight_loads_updated остаётся пустой.
        """
        dbname = self._ensure_db(x_session_id)
        dsn = _as_admin_dsn(self.base_dsn, dbname)
        self._load_schedule_items(dsn, raw_data, target="flight_loads")

    def create_olap(
        self,
        x_session_id: str,
        raw_data: Iterable[ScheduleItem],
        optimized_data: Iterable[ScheduleItem],
    ) -> None:
        """
        Создаёт БД и грузит:
        - raw_data -> flight_loads
        - optimized_data -> flight_loads_updated
        """
        dbname = self._ensure_db(x_session_id)
        dsn = _as_admin_dsn(self.base_dsn, dbname)
        self._load_schedule_items(dsn, raw_data, target="flight_loads")
        self._load_schedule_items(dsn, optimized_data, target="flight_loads_updated")

    def add_optimized_data(self, x_session_id: str, optimized_data: Iterable[ScheduleItem]) -> None:
        """Догружает оптимизированные данные в flight_loads_updated для существующей сессии."""
        dbname = _sanitize_db_name(x_session_id)
        dsn = _as_admin_dsn(self.base_dsn, dbname)
        self._load_schedule_items(dsn, optimized_data, target="flight_loads_updated")

    # ---------------------------- Query helpers -------------------------- #
    def get_summary(self, x_session_id: str) -> dict:
        """
        Пример простых агрегаций (по сырой витрине):
        - количество рейсов
        - суммарный доход
        - средний LF
        """
        dbname = _sanitize_db_name(x_session_id)
        dsn = _as_admin_dsn(self.base_dsn, dbname)
        with _connect(dsn) as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS flights,
                       COALESCE(SUM(revenue),0) AS total_revenue,
                       ROUND(AVG(cabin_load_factor)::numeric, 2) AS avg_lf
                FROM flight_loads
                """
            )
            row = cur.fetchone()
            return {
                "flights": int(row["flights"]),
                "total_revenue": float(row["total_revenue"]),
                "avg_lf": float(row["avg_lf"]) if row["avg_lf"] is not None else 0.0,
            }

    # ---------------------------- Internal loaders ----------------------- #
    def _load_schedule_items(self, dsn: str, items: Iterable[ScheduleItem], target: str) -> None:
        """
        Грузим пачкой:
        - референсы: flight_numbers, airports, aircrafts
        - факты: flight_loads / flight_loads_updated
        """
        # Временные кэши id для минимизации запросов
        cache_flight_no: dict[str, int] = {}
        cache_airport: dict[str, int] = {}
        cache_aircraft: dict[tuple[str, str, int], int] = {}

        with _connect(dsn) as conn:
            conn.autocommit = False
            with conn.cursor() as cur:
                for item in items:
                    # 1) Upsert flight_number
                    fn_id = cache_flight_no.get(item.flight_number)
                    if fn_id is None:
                        cur.execute(
                            "INSERT INTO flight_numbers (flight_no) VALUES (%s) ON CONFLICT (flight_no) DO UPDATE SET flight_no=EXCLUDED.flight_no RETURNING id",
                            (item.flight_number,),
                        )
                        fn_id = cur.fetchone()[0]
                        cache_flight_no[item.flight_number] = fn_id

                    # 2) Upsert airports (dep/arr)
                    dep_code = (item.dep_airport or "").upper()[:3]
                    arr_code = (item.arr_airport or "").upper()[:3]

                    dep_id = cache_airport.get(dep_code)
                    if dep_id is None:
                        cur.execute(
                            "INSERT INTO airports (airport_code) VALUES (%s) ON CONFLICT (airport_code) DO UPDATE SET airport_code=EXCLUDED.airport_code RETURNING id",
                            (dep_code,),
                        )
                        dep_id = cur.fetchone()[0]
                        cache_airport[dep_code] = dep_id

                    arr_id = cache_airport.get(arr_code)
                    if arr_id is None:
                        cur.execute(
                            "INSERT INTO airports (airport_code) VALUES (%s) ON CONFLICT (airport_code) DO UPDATE SET airport_code=EXCLUDED.airport_code RETURNING id",
                            (arr_code,),
                        )
                        arr_id = cur.fetchone()[0]
                        cache_airport[arr_code] = arr_id

                    # 3) Upsert aircrafts (нет уникального индекса в DDL -> делаем вручную)
                    ac_key = (item.flight_type, item.cabin_code, int(item.flight_capacity))
                    ac_id = cache_aircraft.get(ac_key)
                    if ac_id is None:
                        cur.execute(
                            """
                            SELECT id FROM aircrafts
                            WHERE aircraft_type=%s AND cabin_code=%s AND cabin_capacity=%s
                            """,
                            (item.flight_type, item.cabin_code, int(item.flight_capacity)),
                        )
                        row = cur.fetchone()
                        if row:
                            ac_id = row[0]
                        else:
                            cur.execute(
                                """
                                INSERT INTO aircrafts (aircraft_type, cabin_code, cabin_capacity)
                                VALUES (%s, %s, %s) RETURNING id
                                """,
                                (item.flight_type, item.cabin_code, int(item.flight_capacity)),
                            )
                            ac_id = cur.fetchone()[0]
                        cache_aircraft[ac_key] = ac_id

                    # 4) Insert fact row
                    dep_date = _parse_date(item.date)
                    dep_time = _parse_time(item.dep_time)
                    arr_time = _parse_time(item.arr_time)

                    cur.execute(
                        f"""
                        INSERT INTO {target} (
                            departure_date, departure_time, arrival_time,
                            cabin_load_factor, bookings, revenue, passengers,
                            flight_number_id, departure_airport_id, arrival_airport_id, aircraft_id
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            dep_date,
                            dep_time,
                            arr_time,
                            float(item.lf_cabin) if item.lf_cabin is not None else None,
                            int(item.cabins_brones),
                            float(item.pass_income),
                            int(item.passengers),
                            fn_id,
                            dep_id,
                            arr_id,
                            ac_id,
                        ),
                    )
            conn.commit()


# -----------------------------------------------------------------------------
# Parsing helpers (устойчивы к разным форматам)
# -----------------------------------------------------------------------------
def _parse_date(s: str) -> datetime.date:
    # поддержка форматов: 'YYYY-MM-DD', 'DD.MM.YYYY', 'YYYY/MM/DD'
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except Exception:
            pass
    # последний шанс: ISO-like
    return datetime.fromisoformat(s.strip()).date()


def _parse_time(s: str) -> datetime.time:
    # поддержка 'HH:MM', 'HH:MM:SS'
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt).time()
        except Exception:
            pass
    # последний шанс: ISO-like
    return datetime.fromisoformat(s.strip()).time()


# -----------------------------------------------------------------------------
# Пример использования (можно удалить в проде)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    olap = OLAPBuilder()

    demo_raw = [
        ScheduleItem(
            date="2025-09-01",
            flight_number="SU100",
            dep_airport="SVO",
            arr_airport="LED",
            dep_time="08:30",
            arr_time="09:45",
            flight_capacity=180,
            lf_cabin=82.5,
            cabins_brones=160,
            flight_type="A320",
            cabin_code="Y",
            pass_income=123456.78,
            passengers=148,
        ),
        ScheduleItem(
            date="2025-09-01",
            flight_number="SU101",
            dep_airport="LED",
            arr_airport="SVO",
            dep_time="11:00",
            arr_time="12:15",
            flight_capacity=180,
            lf_cabin=75.0,
            cabins_brones=150,
            flight_type="A320",
            cabin_code="Y",
            pass_income=98765.43,
            passengers=135,
        ),
    ]

    demo_opt = [
        ScheduleItem(
            date="2025-09-01",
            flight_number="SU100",
            dep_airport="SVO",
            arr_airport="LED",
            dep_time="08:30",
            arr_time="09:45",
            flight_capacity=180,
            lf_cabin=85.0,
            cabins_brones=162,
            flight_type="A320",
            cabin_code="Y",
            pass_income=128000.00,
            passengers=153,
        ),
    ]

    session_id = "sess_demo_123"
    olap.create_olap(session_id, demo_raw, demo_opt)
    print("Summary:", olap.get_summary(session_id))
    # olap.delete_session(session_id)
