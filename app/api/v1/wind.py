from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app import crud, schemas
from app.schemas import wind_turbine

router = APIRouter(prefix="/wind_turbines")


@router.get("/", response_model=list[wind_turbine.WindTurbine])
async def list_turbines(db: AsyncSession = Depends(get_db)):
    return await crud.wind_turbine.get_all(db)


@router.post("/", response_model=wind_turbine.WindTurbine)
async def create_turbine(turbine_in: wind_turbine.WindTurbineCreate, db: AsyncSession = Depends(get_db)):
    return await crud.wind_turbine.create(db, turbine_in)


@router.put("/{turbine_id}", response_model=wind_turbine.WindTurbine)
async def update_turbine(turbine_id: int, turbine_in: wind_turbine.WindTurbineUpdate, db: AsyncSession = Depends(get_db)):
    turbine = await crud.wind_turbine.update(db, turbine_id, turbine_in)
    if not turbine:
        raise HTTPException(status_code=404, detail="Turbine not found")
    return turbine


@router.delete("/{turbine_id}")
async def delete_turbine(turbine_id: int, db: AsyncSession = Depends(get_db)):
    turbine = await crud.wind_turbine.delete(db, turbine_id)
    if not turbine:
        raise HTTPException(status_code=404, detail="Turbine not found")
    return {"deleted": True, "id": turbine_id}
