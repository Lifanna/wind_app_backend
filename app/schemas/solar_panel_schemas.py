# app/schemas/solar_panel.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class SolarPanelBase(BaseModel):
    name: str
    power: float
    efficiency: float
    status: Optional[str] = "Активна"


class SolarPanelCreate(SolarPanelBase):
    system_id: int


class SolarPanelUpdate(SolarPanelBase):
    pass


class SolarPanel(SolarPanelBase):
    id: int
    system_id: int
    created_at: datetime

    class Config:
        from_attributes = True
