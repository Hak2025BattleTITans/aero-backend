from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ScheduleItem(BaseModel):
    date: str
    flight_number: str
    dep_airport: str
    arr_airport: str
    dep_time: str
    arr_time: str
    flight_capacity: int
    lf_cabin: float
    cabins_brones: int
    flight_type: str
    cabin_code: str
    pass_income: float
    passengers: int

class MetricPair(BaseModel):
    value: float | int
    optimized_value: float | int

class MainMetrics(BaseModel):
    passengers: MetricPair
    income: MetricPair
    avg_check: MetricPair

class Iframe(BaseModel):
    id: str
    src: str
    title: str

class SessionDoc(BaseModel):
    unoptimized_schedule: List[ScheduleItem]
    optimized_schedule: List[ScheduleItem]
    main_metrics: MainMetrics
    iframes: List[Iframe]
    file_key: str
    expires_at: str  # ISO8601 "2025-09-27T12:00:00Z"