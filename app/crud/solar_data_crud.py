# app/crud/solar_crud.py

from typing import List, Optional, Sequence
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.models.solar_data_model import SolarData
from app.schemas.solar_data_schemas import SolarDataCreate, SolarDataUpdate


# ====================== SOLAR DATA CRUD ======================


async def get_solar_data(
    db: AsyncSession,
    system_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[SolarData]:
    query = select(SolarData).order_by(SolarData.created_at)

    if system_id is not None:
        query = query.where(SolarData.system_id == system_id)
    if start_date is not None:
        query = query.where(SolarData.created_at >= start_date)
    if end_date is not None:
        query = query.where(SolarData.created_at <= end_date)

    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


async def get_solar_data_by_id(db: AsyncSession, data_id: int) -> Optional[SolarData]:
    result = await db.execute(select(SolarData).where(SolarData.id == data_id))
    return result.scalars().first()


async def create_solar_data(db: AsyncSession, obj_in: SolarDataCreate) -> SolarData:
    db_obj = SolarData(**obj_in.model_dump())
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def create_solar_data_bulk(db: AsyncSession, objects: List[SolarDataCreate]) -> int:
    """Создаёт много записей за один раз. Возвращает количество"""
    db_objs = [SolarData(**obj.model_dump()) for obj in objects]
    db.add_all(db_objs)
    await db.commit()
    return len(db_objs)


async def update_solar_data(
    db: AsyncSession,
    db_obj: SolarData,
    obj_in: SolarDataUpdate,
) -> SolarData:
    update_data = obj_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def delete_solar_data(db: AsyncSession, db_obj: SolarData) -> SolarData:
    await db.delete(db_obj)
    await db.commit()
    return db_obj