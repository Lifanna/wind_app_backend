# app/schemas/solar.py

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel


# Базовая схема для SolarSystem
class SolarSystemBase(BaseModel):
    name: Optional[str] = None
    power_kw: Optional[float] = None
    efficiency: Optional[float] = None
    status: Optional[str] = None


# Создание
class SolarSystemCreate(SolarSystemBase):
    pass # можно сделать обязательным при создании


# Обновление (все поля опциональны)
class SolarSystemUpdate(SolarSystemBase):
    pass


# Ответ (с данными из БД)
class SolarSystem(SolarSystemBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True  # для SQLAlchemy 2.0 (ранее orm_mode = True)
        
        
class SolarForecastBase(BaseModel):
    system_id: int
    timestamp: datetime
    horizon_hours: Optional[int] = 6
    predicted_power: float
    model_name: Optional[str] = "RandomForest"
    mae: Optional[float] = None
    r2: Optional[float] = None


class SolarForecastCreate(SolarForecastBase):
    pass


class SolarForecastUpdate(BaseModel):
    timestamp: Optional[datetime] = None
    predicted_power: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None


class SolarForecast(SolarForecastBase):
    id: int
    class Config:
        from_attributes = True