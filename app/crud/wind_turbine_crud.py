# app/crud/wind_crud.py

from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.wind_data_model import WindData
from app.models.wind_forecast_model import WindForecast
from app.models.wind_turbine_model import WindTurbine
from app.schemas.wind_turbine import (
    WindTurbineCreate, WindTurbineUpdate,
    WindDataCreate, WindDataUpdate,
    WindForecastCreate, WindForecastUpdate,
)


# === Wind Turbine ===
async def get_turbines(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[WindTurbine]:
    result = await db.execute(select(WindTurbine).offset(skip).limit(limit))
    return result.scalars().all()


async def get_turbine(db: AsyncSession, turbine_id: int) -> Optional[WindTurbine]:
    result = await db.execute(select(WindTurbine).where(WindTurbine.id == turbine_id))
    return result.scalars().first()


async def create_turbine(db: AsyncSession, obj_in: WindTurbineCreate) -> WindTurbine:
    db_obj = WindTurbine(**obj_in.model_dump())
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def update_turbine(db: AsyncSession, db_obj: WindTurbine, obj_in: WindTurbineUpdate) -> WindTurbine:
    update_data = obj_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def delete_turbine(db: AsyncSession, db_obj: WindTurbine) -> WindTurbine:
    await db.delete(db_obj)
    await db.commit()
    return db_obj


# === Wind Data ===
async def get_wind_data(
    db: AsyncSession,
    wind_turbine_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100
) -> List[WindData]:
    query = select(WindData).order_by(WindData.created_at.desc())
    if wind_turbine_id is not None:
        query = query.where(WindData.wind_turbine_id == wind_turbine_id)
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


async def get_wind_datum(db: AsyncSession, data_id: int) -> Optional[WindData]:
    result = await db.execute(select(WindData).where(WindData.id == data_id))
    return result.scalars().first()


async def create_wind_data(db: AsyncSession, obj_in: WindDataCreate) -> WindData:
    db_obj = WindData(**obj_in.model_dump())
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def update_wind_data(db: AsyncSession, db_obj: WindData, obj_in: WindDataUpdate) -> WindData:
    update_data = obj_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def delete_wind_data(db: AsyncSession, db_obj: WindData) -> WindData:
    await db.delete(db_obj)
    await db.commit()
    return db_obj


# === Wind Forecast ===
async def get_wind_forecasts(
    db: AsyncSession,
    turbine_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100
) -> List[WindForecast]:
    query = select(WindForecast).order_by(WindForecast.target_time)
    if turbine_id is not None:
        query = query.where(WindForecast.turbine_id == turbine_id)
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


async def get_wind_forecast(db: AsyncSession, forecast_id: int) -> Optional[WindForecast]:
    result = await db.execute(select(WindForecast).where(WindForecast.id == forecast_id))
    return result.scalars().first()


async def create_wind_forecast(db: AsyncSession, obj_in: WindForecastCreate) -> WindForecast:
    db_obj = WindForecast(**obj_in.model_dump())
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def update_wind_forecast(db: AsyncSession, db_obj: WindForecast, obj_in: WindForecastUpdate) -> WindForecast:
    update_data = obj_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def delete_wind_forecast(db: AsyncSession, db_obj: WindForecast) -> WindForecast:
    await db.delete(db_obj)
    await db.commit()
    return db_obj