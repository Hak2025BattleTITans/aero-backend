import csv
import logging
from logging.config import dictConfig
from typing import List

import aiofiles

from logging_config import LOGGING_CONFIG, ColoredFormatter
from models import ScheduleItem

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setFormatter(ColoredFormatter('%(levelname)s:     %(asctime)s %(name)s - %(message)s'))


class AsyncCSVReader:
    def __init__(self, path: str, delimiter: str = ";"):
        logger.debug(f"Initialized AsyncCSVReader with path: {path} and delimiter: '{delimiter}'")
        self.path = path
        self.delimiter = delimiter

    async def read(self) -> List[ScheduleItem]:
        items: List[ScheduleItem] = []

        async with aiofiles.open(self.path, mode="r", encoding="utf-8") as f:
            logger.debug(f"Reading CSV file from path: {self.path}")
            content = await f.read()

        reader = csv.DictReader(content.splitlines(), delimiter=self.delimiter)

        for row in reader:
            row = {k: v.replace(",", ".") if isinstance(v, str) else v for k, v in row.items()}

            item = ScheduleItem(
                date=row["Дата вылета"],
                flight_number=row["Номер рейса"],
                dep_airport=row["Аэропорт вылета"],
                arr_airport=row["Аэропорт прилета"],
                dep_time=row["Время вылета"],
                arr_time=row["Время прилета"],
                flight_capacity=int(row["Емкость кабины"]),
                lf_cabin=float(row["LF Кабина"]),
                cabins_brones=int(row["Бронирования по кабинам"]),
                flight_type=row["Тип ВС"],
                cabin_code=row["Код кабины"],
                pass_income=float(row["Доход пасс"]),
                passengers=int(row["Пассажиры"]),
            )
            items.append(item)

        logger.debug(f"Read {len(items)} items from CSV file")
        return items