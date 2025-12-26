# app/api/v1/solar_panel.py

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.crud import solar_panel_crud
from app.schemas.solar_panel_schemas import SolarPanel, SolarPanelCreate, SolarPanelUpdate

router = APIRouter()


@router.get("/", response_model=List[SolarPanel])
async def get_panels(
    db: AsyncSession = Depends(get_db),
    system_id: int | None = Query(None, description="Фильтр по ID солнечной системы"),
    skip: int = 0,
    limit: int = 100,
):
    """Получить список панелей (с фильтром по системе)"""
    panels = await solar_panel_crud.get_panels(db=db, system_id=system_id, skip=skip, limit=limit)
    return panels


@router.get("/{panel_id}", response_model=SolarPanel)
async def get_panel(panel_id: int, db: AsyncSession = Depends(get_db)):
    """Получить одну панель по ID"""
    panel = await solar_panel_crud.get_panel_by_id(db, panel_id)
    if not panel:
        raise HTTPException(status_code=404, detail="Panel not found")
    return panel


@router.post("/", response_model=SolarPanel, status_code=201)
async def create_panel(
    panel_in: SolarPanelCreate,
    db: AsyncSession = Depends(get_db),
):
    """Создать новую панель"""
    return await solar_panel_crud.create_panel(db, panel_in)


@router.put("/{panel_id}", response_model=SolarPanel)
async def update_panel(
    panel_id: int,
    panel_in: SolarPanelUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Обновить панель"""
    panel = await solar_panel_crud.get_panel_by_id(db, panel_id)
    if not panel:
        raise HTTPException(status_code=404, detail="Panel not found")
    return await solar_panel_crud.update_panel(db, panel, panel_in)


@router.delete("/{panel_id}", response_model=SolarPanel)
async def delete_panel(
    panel_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Удалить панель"""
    panel = await solar_panel_crud.get_panel_by_id(db, panel_id)
    if not panel:
        raise HTTPException(status_code=404, detail="Panel not found")
    return await solar_panel_crud.delete_panel(db, panel)