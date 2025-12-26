# app/api/v1/wind.py

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.crud.wind_turbine_crud import (
    # Turbines
    get_turbines, get_turbine, create_turbine, update_turbine, delete_turbine,
    # Data
    get_wind_data, get_wind_datum, create_wind_data, update_wind_data, delete_wind_data,
    # Forecasts
    get_wind_forecasts, create_wind_forecast, update_wind_forecast, delete_wind_forecast, get_wind_forecast,
)
from app.schemas.wind_turbine import (
    WindTurbine, WindTurbineCreate, WindTurbineUpdate,
    WindData, WindDataCreate, WindDataUpdate,
    WindForecast, WindForecastCreate, WindForecastUpdate,
)

router = APIRouter()


# === Wind Turbines ===
@router.get("/wind_turbines/", response_model=List[WindTurbine])
async def read_turbines(db: AsyncSession = Depends(get_db), skip: int = 0, limit: int = 100):
    return await get_turbines(db, skip, limit)


@router.get("/wind_turbines/{turbine_id}", response_model=WindTurbine)
async def read_turbine(turbine_id: int, db: AsyncSession = Depends(get_db)):
    db_obj = await get_turbine(db, turbine_id)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Turbine not found")
    return db_obj


@router.post("/wind_turbines/", response_model=WindTurbine, status_code=201)
async def create_turbine_endpoint(obj_in: WindTurbineCreate, db: AsyncSession = Depends(get_db)):
    return await create_turbine(db, obj_in)


@router.put("/wind_turbines/{turbine_id}", response_model=WindTurbine)
async def update_turbine_endpoint(turbine_id: int, obj_in: WindTurbineUpdate, db: AsyncSession = Depends(get_db)):
    db_obj = await get_turbine(db, turbine_id)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Turbine not found")
    return await update_turbine(db, db_obj, obj_in)


@router.delete("/wind_turbines/{turbine_id}", response_model=WindTurbine)
async def delete_turbine_endpoint(turbine_id: int, db: AsyncSession = Depends(get_db)):
    db_obj = await get_turbine(db, turbine_id)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Turbine not found")
    return await delete_turbine(db, db_obj)


# === Wind Data ===
@router.get("/wind_data/", response_model=List[WindData])
async def read_wind_data(
    db: AsyncSession = Depends(get_db),
    wind_turbine_id: Optional[int] = Query(None),
    skip: int = 0,
    limit: int = 100,
):
    return await get_wind_data(db, wind_turbine_id, skip, limit)


@router.post("/wind_data/", response_model=WindData, status_code=201)
async def create_wind_data_endpoint(obj_in: WindDataCreate, db: AsyncSession = Depends(get_db)):
    return await create_wind_data(db, obj_in)


# === Wind Forecasts ===
@router.get("/wind_forecasts/", response_model=List[WindForecast])
async def read_wind_forecasts(
    db: AsyncSession = Depends(get_db),
    turbine_id: Optional[int] = Query(None),
    skip: int = 0,
    limit: int = 100,
):
    return await get_wind_forecasts(db, turbine_id, skip, limit)


@router.post("/wind_forecasts/", response_model=WindForecast, status_code=201)
async def create_wind_forecast_endpoint(obj_in: WindForecastCreate, db: AsyncSession = Depends(get_db)):
    return await create_wind_forecast(db, obj_in)