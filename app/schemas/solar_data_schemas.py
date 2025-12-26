from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class SolarDataBase(BaseModel):
    system_id: int
    created_at: datetime
    ghi: Optional[float] = None
    dni: Optional[float] = None
    dhi: Optional[float] = None
    poa_global: Optional[float] = None
    temp_air: Optional[float] = None
    wind_speed: Optional[float] = None
    temp_cell: Optional[float] = None
    dc_power: Optional[float] = None
    ac_power: Optional[float] = None
    cloud_factor: Optional[float] = None


class SolarDataCreate(SolarDataBase):
    pass


class SolarDataUpdate(BaseModel):
    ghi: Optional[float] = None
    dni: Optional[float] = None
    dhi: Optional[float] = None
    poa_global: Optional[float] = None
    temp_air: Optional[float] = None
    wind_speed: Optional[float] = None
    temp_cell: Optional[float] = None
    dc_power: Optional[float] = None
    ac_power: Optional[float] = None
    cloud_factor: Optional[float] = None
    # created_at и system_id обычно не обновляем


class SolarData(SolarDataBase):
    id: int

    class Config:
        from_attributes = True