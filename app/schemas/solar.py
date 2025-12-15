from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class SolarSystemBase(BaseModel):
    name: Optional[str] = "Default Solar System"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None
    tilt: Optional[float] = None
    azimuth: Optional[float] = None
    target_kw: Optional[float] = None
    module_power_stc: Optional[float] = None
    albedo: Optional[float] = None


class SolarSystemCreate(SolarSystemBase):
    name: str


class SolarSystemUpdate(SolarSystemBase):
    pass


class SolarSystemInDBBase(SolarSystemBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True


class SolarSystem(SolarSystemInDBBase):
    pass
