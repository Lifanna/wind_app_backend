from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from app.models.wind_turbine_model import WindTurbine  # <-- путь к твоей модели, поправь если нужно
from app.schemas.wind_turbine import WindTurbineCreate, WindTurbineUpdate


async def get_all(db: AsyncSession) -> List[WindTurbine]:
    """
    Получить все ветрогенераторы
    """
    result = await db.execute(
        select(WindTurbine)
        .options(selectinload(WindTurbine.measurements))  # если нужно подгружать измерения сразу
        .order_by(WindTurbine.id)
    )
    return result.scalars().all()


async def get_all_data(db: AsyncSession) -> list:
    """
    Получить все данные ветровых турбин
    """
    result = await db.execute(
        select(WindTurbine).order_by(WindTurbine.id)
    )
    return result.scalars().all()

async def create(db: AsyncSession, turbine_in: WindTurbineCreate) -> WindTurbine:
    """
    Создать новый ветрогенератор
    """
    turbine = WindTurbine(
        name=turbine_in.name,
        power=turbine_in.power,
        location=turbine_in.location,
        status=turbine_in.status or "Активен",  # дефолт из схемы, но на всякий случай
    )
    db.add(turbine)
    await db.commit()
    await db.refresh(turbine)
    return turbine


async def update(db: AsyncSession, turbine_id: int, turbine_in: WindTurbineUpdate) -> WindTurbine | None:
    """
    Обновить ветрогенератор по ID.
    Возвращает обновлённый объект или None, если не найден.
    """
    # Собираем только те поля, которые переданы (exclude_unset=True)
    update_data = turbine_in.model_dump(exclude_unset=True)

    if not update_data:
        # Если ничего не передано — просто возвращаем существующий объект (или None)
        result = await db.execute(select(WindTurbine).where(WindTurbine.id == turbine_id))
        return result.scalar_one_or_none()

    stmt = (
        update(WindTurbine)
        .where(WindTurbine.id == turbine_id)
        .values(**update_data)
        .returning(WindTurbine)
    )
    result = await db.execute(stmt)
    await db.commit()
    return result.scalar_one_or_none()


async def delete(db: AsyncSession, turbine_id: int) -> WindTurbine | None:
    """
    Удалить ветрогенератор по ID.
    Возвращает удалённый объект или None, если не найден.
    Благодаря cascade="all, delete" в модели — удалятся и связанные WindData.
    """
    turbine = await db.get(WindTurbine, turbine_id)
    if turbine:
        await db.delete(turbine)
        await db.commit()
    return turbine