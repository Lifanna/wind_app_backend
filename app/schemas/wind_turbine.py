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
        from_attributes = True
        
    
class WindTurbineData(BaseModel):
    timestamp: datetime
    wind_speed_ref: float
    wind_speed: float
    power_raw_W: float
    power_curve_W: float
    power_ac_W: float
    rho: float
    rotor_diameter_m: float
    rated_power_W: float

    class Config:
        from_attributes = True    
