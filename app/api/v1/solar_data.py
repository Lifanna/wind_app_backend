# app/api/v1/solar_data.py

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.api.deps import get_db
from app.crud import solar_data_crud
from app.schemas.solar_data_schemas import SolarData, SolarDataCreate, SolarDataUpdate

router = APIRouter()


@router.get("/", response_model=List[SolarData])
async def get_solar_data(
    db: AsyncSession = Depends(get_db),
    system_id: int | None = None,   
    start_date: datetime | None = Query(None),
    end_date: datetime | None = Query(None),
    skip: int = 0,
    limit: int = 100,
):
    """
    Получить почасовые данные генерации.
    Фильтры:
    - system_id: фильтр по солнечной системе
    - start_date, end_date: диапазон дат (включительно)
    - skip, limit: пагинация
    """
    data = await solar_data_crud.get_solar_data(
        db=db,
        system_id=system_id,
        start_date=start_date,
        end_date=end_date,
        skip=skip,
        limit=limit,
    )
    return data


@router.get("/{data_id}", response_model=SolarData)
async def get_solar_data_by_id(data_id: int, db: AsyncSession = Depends(get_db)):
    """Получить одну запись по ID"""
    obj = await solar_data_crud.get_solar_data_by_id(db, data_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Solar data not found")
    return obj


@router.post("/", response_model=SolarData)
async def create_solar_data(
    obj_in: SolarDataCreate,
    db: AsyncSession = Depends(get_db),
):
    """Создать одну запись"""
    return await solar_data_crud.create_solar_data(db, obj_in)


@router.post("/bulk", response_model=int)
async def create_solar_data_bulk(
    objects: List[SolarDataCreate],
    db: AsyncSession = Depends(get_db),
):
    """
    Загрузить много записей сразу (удобно для импорта CSV)
    Возвращает количество созданных записей
    """
    count = await solar_data_crud.create_solar_data_bulk(db, objects)
    return count


@router.put("/{data_id}", response_model=SolarData)
async def update_solar_data(
    data_id: int,
    obj_in: SolarDataUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Обновить запись"""
    obj = await solar_data_crud.get_solar_data_by_id(db, data_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Solar data not found")
    return await solar_data_crud.update_solar_data(db, obj, obj_in)


@router.delete("/{data_id}", response_model=SolarData)
async def delete_solar_data(
    data_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Удалить запись"""
    obj = await solar_data_crud.get_solar_data_by_id(db, data_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Solar data not found")
    return await solar_data_crud.delete_solar_data(db, obj)