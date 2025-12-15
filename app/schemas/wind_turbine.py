from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class WindTurbineBase(BaseModel):
    name: str
    power: float
    location: Optional[str] = None
    status: Optional[str] = "Активен"


class WindTurbineCreate(WindTurbineBase):
    pass


class WindTurbineUpdate(WindTurbineBase):
    pass


class WindTurbine(WindTurbineBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
