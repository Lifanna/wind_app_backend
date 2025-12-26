# app/crud/solar_crud.py (добавьте эти функции)

from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.solar_panel_model import SolarPanel
from app.schemas.solar_panel_schemas import SolarPanelCreate, SolarPanelUpdate


async def get_panels(
    db: AsyncSession,
    system_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[SolarPanel]:
    query = select(SolarPanel).order_by(SolarPanel.id)
    if system_id is not None:
        query = query.where(SolarPanel.system_id == system_id)
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


async def get_panel_by_id(db: AsyncSession, panel_id: int) -> Optional[SolarPanel]:
    result = await db.execute(select(SolarPanel).where(SolarPanel.id == panel_id))
    return result.scalars().first()


async def create_panel(db: AsyncSession, obj_in: SolarPanelCreate) -> SolarPanel:
    db_obj = SolarPanel(**obj_in.model_dump())
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def update_panel(
    db: AsyncSession,
    db_obj: SolarPanel,
    obj_in: SolarPanelUpdate,
) -> SolarPanel:
    update_data = obj_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def delete_panel(db: AsyncSession, db_obj: SolarPanel) -> SolarPanel:
    await db.delete(db_obj)
    await db.commit()
    return db_obj