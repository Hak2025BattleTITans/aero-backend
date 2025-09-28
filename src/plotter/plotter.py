import base64
import inspect
import json
import logging
import os
from logging.config import dictConfig
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder

from csv_reader import AsyncCSVReader
from logging_config import LOGGING_CONFIG, ColoredFormatter
from models import ScheduleItem

# Глобально запретить бинарные массивы в JSON
try:
    # В новых Plotly есть конфиг json-энджина
    pio.json.config.default_engine = "json"  # type: ignore[attr-defined]
except Exception:
    pass


dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
for h in logging.getLogger().handlers:
    if isinstance(h, logging.StreamHandler):
        h.setFormatter(
            ColoredFormatter("%(levelname)s:     %(asctime)s %(name)s - %(message)s")
        )

def _debug_trace_shapes(fig):
    spec = _fig_spec(fig)  # новый
    out = []
    for t in spec.get("data", []):
        ttype = t.get("type")
        if ttype == "pie":
            out.append({"type": ttype,
                        "labels": len(t.get("labels", [])),
                        "values": len(t.get("values", []))})
        else:
            out.append({"type": ttype,
                        "x": len(t.get("x", [])),
                        "y": len(t.get("y", []))})
    return out

def _unpack_bdata(obj):
    """Рекурсивно заменяет {"dtype": "...", "bdata": "..."} на обычные list"""
    if isinstance(obj, dict):
        # точное совпадение по ключам
        if set(obj.keys()) == {"dtype", "bdata"}:
            arr = np.frombuffer(base64.b64decode(obj["bdata"]), dtype=np.dtype(obj["dtype"]))
            return arr.tolist()
        return {k: _unpack_bdata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack_bdata(v) for v in obj]
    return obj

def _fig_spec(fig) -> dict:
    # 1) пробуем «правильный» движок, который обычно НЕ делает bdata
    s = pio.to_json(fig, engine="json", validate=True)
    spec = json.loads(s)
    # 2) на всякий случай распаковываем, если где-то всё же проскочило
    spec = _unpack_bdata(spec)
    return spec


def _trace_shapes(spec: dict):
    out = []
    for t in spec.get("data", []):
        ttype = t.get("type")
        if ttype == "pie":
            out.append({"type": ttype, "labels": len(t.get("labels", [])), "values": len(t.get("values", []))})
        else:
            out.append({"type": ttype, "x": len(t.get("x", [])), "y": len(t.get("y", []))})
    return out

class Plotter:
    def __init__(self, csv_path: str):
        if inspect.iscoroutine(csv_path):
            raise TypeError("csv_path is coroutine; did you forget to await AsyncCSVReader.write()?")

        self.csv_path = csv_path

        # 1) Читаем с учётом возможного BOM
        self.df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")

        # 2) Чистим названия колонок (BOM, пробелы)
        self.df.rename(columns=lambda s: str(s).replace("\ufeff", "").strip(), inplace=True)

        # 3) Нормализация чисел: убираем обычные и неразрывные пробелы, меняем запятые на точки
        def _to_num(s: pd.Series) -> pd.Series:
            return pd.to_numeric(
                s.astype(str)
                 .str.replace(r"[\s\u00A0]", "", regex=True)   # пробел + NBSP
                 .str.replace(",", ".", regex=False),
                errors="coerce"
            )

        for col in ["Доход пасс", "Пассажиры"]:
            if col in self.df.columns:
                self.df[col] = _to_num(self.df[col])
            else:
                raise KeyError(f"В CSV нет обязательной колонки: {col!r}. Нашлись: {list(self.df.columns)}")

        # 4) Даты: учитываем российский формат
        if "Дата вылета" not in self.df.columns:
            raise KeyError("Нет колонки 'Дата вылета'")

        # Сначала чистим строку от пробелов/NBSP, потом парсим с dayfirst.
        self.df["Дата вылета"] = pd.to_datetime(
            self.df["Дата вылета"].astype(str).str.strip().str.replace("\u00A0", ""),
            dayfirst=True, errors="coerce", infer_datetime_format=True
        )

    @classmethod
    async def from_items(cls, items: List[ScheduleItem], csv_path: str):
        # Если нужно предварительно записать CSV асинхронно — ждём путь
        csv_reader = AsyncCSVReader(csv_path)
        new_csv_path = await csv_reader.write(items)   # ВАЖНО: await
        return cls(new_csv_path)

    def avg_check(self):
        df = self.df.copy()
        df = df.dropna(subset=["Код кабины"])
        df = df[(df["Пассажиры"] > 0) & df["Доход пасс"].notna()]
        df["Средний чек"] = df["Доход пасс"] / df["Пассажиры"]
        df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["Средний чек"])
        avg_df = df.groupby("Код кабины", as_index=False)["Средний чек"].mean()
        fig = px.pie(avg_df, names="Код кабины", values="Средний чек", title="Средний чек по коду кабины")
        logger.debug(_debug_trace_shapes(fig))
        return _fig_spec(fig)

    def dyn_income(self):
        df = self.df.copy()
        df = df.dropna(subset=["Дата вылета", "Код кабины", "Доход пасс"])

        daily = (df.groupby(["Дата вылета", "Код кабины"], as_index=False)["Доход пасс"].sum())
        daily["Дата вылета"] = pd.to_datetime(daily["Дата вылета"], errors="coerce").dt.strftime("%Y-%m-%d")
        daily = daily.dropna(subset=["Дата вылета", "Код кабины", "Доход пасс"]).sort_values(["Код кабины", "Дата вылета"])

        fig = px.line(
            daily,
            x="Дата вылета",
            y="Доход пасс",
            color="Код кабины",
            facet_row="Код кабины",
            title="Динамика дохода по коду кабины (сумма за день)",
            markers=True,           # вот это важно
        )
        fig.update_traces(mode="lines+markers", connectgaps=True)
        fig.update_layout(height=800, hovermode="x unified")

        spec = _fig_spec(fig)

        # sanity-check: длины x и y обязаны совпадать
        for i, tr in enumerate(spec.get("data", [])):
            if isinstance(tr.get("y"), dict):  # <- больше так быть не должно
                raise RuntimeError(f"[dyn_income] trace {i} y is still packed: keys={list(tr['y'].keys())}")
            if isinstance(tr.get("x"), dict):
                raise RuntimeError(f"[dyn_income] trace {i} x is still packed: keys={list(tr['x'].keys())}")
            if len(tr.get("x", [])) != len(tr.get("y", [])):
                logger.warning(f"[dyn_income] trace {i}: len(x)={len(tr.get('x', []))} != len(y)={len(tr.get('y', []))}")
        return spec


    def dyn_passenger(self):
        # Динамика пассажиропотока по дням и по коду кабины
        df = self.df.copy()
        df = df.dropna(subset=["Дата вылета", "Код кабины", "Пассажиры"])

        daily_passengers = (
            df.groupby(["Дата вылета", "Код кабины"], as_index=False)["Пассажиры"].sum()
        )

        # (не обязательно, но помогает порядку и стабильности JSON)
        daily_passengers["Дата вылета"] = pd.to_datetime(
            daily_passengers["Дата вылета"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        daily_passengers = daily_passengers.sort_values(["Код кабины", "Дата вылета"])

        fig = px.line(
            daily_passengers,
            x="Дата вылета",
            y="Пассажиры",
            color="Код кабины",
            facet_row="Код кабины",
            title="Динамика пассажиропотока по коду кабины (сумма за день)",
            markers=True,  # как и просили
        )
        fig.update_traces(mode="lines+markers", connectgaps=True)
        fig.update_layout(height=800, hovermode="x unified")
        return _fig_spec(fig)
