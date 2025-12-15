from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db
from app import crud, schemas
from app.schemas import solar

router = APIRouter(prefix="/solar_system")


@router.get("/", response_model=list[solar.SolarSystem])
async def read_solar_systems(db: AsyncSession = Depends(get_db)):
    return await crud.solar.get_all(db)


@router.get("/{id}", response_model=solar.SolarSystem)
async def read_solar_system(id: int, db: AsyncSession = Depends(get_db)):
    solar = await crud.solar.get(db, id)
    if not solar:
        raise HTTPException(status_code=404, detail="Solar system not found")
    return solar


@router.post("/", response_model=solar.SolarSystem)
async def create_solar_system(
    solar_in: solar.SolarSystemCreate,
    db: AsyncSession = Depends(get_db)
):
    return await crud.solar.create(db, solar_in)


@router.put("/{id}", response_model=solar.SolarSystem)
async def update_solar_system(
    id: int,
    solar_in: solar.SolarSystemUpdate,
    db: AsyncSession = Depends(get_db)
):
    db_obj = await crud.solar.get(db, id)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Solar system not found")
    return await crud.solar.update(db, db_obj, solar_in)


@router.delete("/{id}", response_model=solar.SolarSystem)
async def delete_solar_system(id: int, db: AsyncSession = Depends(get_db)):
    db_obj = await crud.solar.get(db, id)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Solar system not found")
    return await crud.solar.delete(db, db_obj)
