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