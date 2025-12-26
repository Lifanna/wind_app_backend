from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db
from app.crud import solar_crud
from app.schemas import solar

router = APIRouter(prefix="/solar_system")


@router.get("/", response_model=List[solar.SolarSystem])
async def read_solar_systems(db: AsyncSession = Depends(get_db)):
    return await solar_crud.get_all(db)


@router.get("/{id}", response_model=solar.SolarSystem)
async def read_solar_system(id: int, db: AsyncSession = Depends(get_db)):
    solar = await solar_crud.get(db, id)
    if not solar:
        raise HTTPException(status_code=404, detail="Solar system not found")
    return solar


@router.post("/", response_model=solar.SolarSystemCreate)
async def create_solar_system(
    solar_in: solar_crud.SolarSystemCreate,
    db: AsyncSession = Depends(get_db)
):
    return await solar_crud.create(db, solar_in)


@router.put("/{id}", response_model=solar.SolarSystem)
async def update_solar_system(
    id: int,
    solar_in: solar_crud.SolarSystemUpdate,
    db: AsyncSession = Depends(get_db)
):
    db_obj = await solar_crud.get(db, id)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Solar system not found")
    return await solar_crud.update(db, db_obj, solar_in)


@router.delete("/{id}", response_model=solar.SolarSystem)
async def delete_solar_system(id: int, db: AsyncSession = Depends(get_db)):
    db_obj = await solar_crud.get(db, id)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Solar system not found")
    return await solar_crud.delete(db, db_obj)
