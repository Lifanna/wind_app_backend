from pydantic import BaseModel
from datetime import datetime

class SimulationLogBase(BaseModel):
    timestamp: datetime
    solar_ac: float
    wind_ac: float
    load: float
    soc: float
    action: str
    hour_sin: float | None = None
    hour_cos: float | None = None
    reward: float | None = None
    spill_kw: float | None = None
    unmet_kw: float | None = None

class SimulationLogOut(SimulationLogBase):
    id: int
    class Config:
        orm_mode = True
