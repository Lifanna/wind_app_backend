# app/crud/solar.py

from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.solar_data_model import SolarData
from app.models.solar_system_model import SolarSystem
from sqlalchemy.orm import selectinload
from app.schemas.solar import (
    SolarSystemCreate,
    SolarSystemUpdate,
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