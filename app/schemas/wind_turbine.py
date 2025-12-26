# app/schemas/wind.py

from typing import Optional
from datetime import datetime
from pydantic import BaseModel


# === Wind Turbine ===
class WindTurbineBase(BaseModel):
    name: str
    power: float  # кВт
    location: Optional[str] = None
    status: Optional[str] = "Активна"


class WindTurbineCreate(WindTurbineBase):
    pass


class WindTurbineUpdate(BaseModel):
    name: Optional[str] = None
    power: Optional[float] = None
    location: Optional[str] = None
    status: Optional[str] = None


class WindTurbine(WindTurbineBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# === Wind Data (измерения) ===
class WindDataBase(BaseModel):
    wind_turbine_id: int
    wind_speed_ref: float
    wind_speed: float
    power_raw_W: float
    power_curve_W: float
    power_ac_W: float
    rho: float
    rotor_diameter_m: float
    rated_power_W: float
    created_at: datetime


class WindDataCreate(WindDataBase):
    pass


class WindDataUpdate(BaseModel):
    wind_speed_ref: Optional[float] = None
    wind_speed: Optional[float] = None
    power_raw_W: Optional[float] = None
    power_curve_W: Optional[float] = None
    power_ac_W: Optional[float] = None
    rho: Optional[float] = None
    rotor_diameter_m: Optional[float] = None
    rated_power_W: Optional[float] = None


class WindData(WindDataBase):
    id: int

    class Config:
        from_attributes = True


# === Wind Forecast ===
class WindForecastBase(BaseModel):
    turbine_id: int
    target_time: datetime
    wind_speed_forecast: Optional[float] = None
    wind_power_kw_forecast: Optional[float] = None


class WindForecastCreate(WindForecastBase):
    pass


class WindForecastUpdate(BaseModel):
    target_time: Optional[datetime] = None
    wind_speed_forecast: Optional[float] = None
    wind_power_kw_forecast: Optional[float] = None


class WindForecast(WindForecastBase):
    id: int
    forecast_time: datetime

    class Config:
        from_attributes = True