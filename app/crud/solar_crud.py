# app/crud/solar.py

from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.solar_data_model import SolarData
from app.models.solar_system_model import SolarSystem
# app/crud/solar_crud.py

from typing import List, Optional   
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Правильные импорты моделей (из models, а НЕ из schemas!)
from app.models.solar_forecast_model import SolarForecast

# Схемы (Pydantic) — только для входа/выхода
from app.schemas.solar import (
    SolarSystemCreate,
    SolarSystemUpdate,
    SolarForecastCreate,
    SolarForecastUpdate,
)


# ====================== SOLAR SYSTEM CRUD ======================


async def get_all(db: AsyncSession) -> List[SolarSystem]:
    """Получить все солнечные системы"""
    result = await db.execute(select(SolarSystem).order_by(SolarSystem.id))
    return result.scalars().all()


async def get(db: AsyncSession, system_id: int) -> Optional[SolarSystem]:
    """Получить систему по ID"""
    result = await db.execute(select(SolarSystem).where(SolarSystem.id == system_id))
    return result.scalars().first()


async def create(db: AsyncSession, obj_in: SolarSystemCreate) -> SolarSystem:
    """Создать новую солнечную систему"""
    db_obj = SolarSystem(**obj_in.model_dump())
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def update(
    db: AsyncSession,
    db_obj: SolarSystem,
    obj_in: SolarSystemUpdate
) -> SolarSystem:
    """Обновить существующую систему"""
    update_data = obj_in.model_dump(exclude_unset=True)  # только те поля, которые переданы
    for field, value in update_data.items():
        setattr(db_obj, field, value)

    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def delete(db: AsyncSession, db_obj: SolarSystem) -> SolarSystem:
    """Удалить систему (каскадно удалятся measurements, forecasts, panels)"""
    await db.delete(db_obj)
    await db.commit()
    return db_obj


# ====================== ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ (по желанию) ======================

async def get_by_name(db: AsyncSession, name: str) -> Optional[SolarSystem]:
    result = await db.execute(select(SolarSystem).where(SolarSystem.name == name))
    return result.scalars().first()


async def get_with_measurements(
    db: AsyncSession,
    system_id: int,
    limit: int = 100
) -> Optional[SolarSystem]:
    """Получить систему вместе с последними измерениями (eager load)"""
    result = await db.execute(
        select(SolarSystem)
        .where(SolarSystem.id == system_id)
        .options(
            selectinload(SolarSystem.measurements).load_only(
                SolarData.created_at,
                SolarData.ac_power,
                SolarData.dc_power,
                SolarData.ghi
            ).order_by(SolarData.created_at.desc()).limit(limit)
        )
    )
    return result.scalars().first()




async def get_solar_forecasts(
    db: AsyncSession,
    system_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100
) -> List[SolarForecast]:
    query = select(SolarForecast).order_by(SolarForecast.timestamp.desc())
    if system_id is not None:
        query = query.where(SolarForecast.system_id == system_id)
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


async def get_solar_forecast(db: AsyncSession, forecast_id: int) -> Optional[SolarForecast]:
    result = await db.execute(select(SolarForecast).where(SolarForecast.id == forecast_id))
    return result.scalars().first()


async def create_solar_forecast(db: AsyncSession, obj_in: SolarForecastCreate) -> SolarForecast:
    db_obj = SolarForecast(**obj_in.model_dump())
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def update_solar_forecast(
    db: AsyncSession,
    db_obj: SolarForecast,
    obj_in: SolarForecastUpdate
) -> SolarForecast:
    update_data = obj_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def delete_solar_forecast(db: AsyncSession, db_obj: SolarForecast) -> SolarForecast:
    await db.delete(db_obj)
    await db.commit()
    return db_obj