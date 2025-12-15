from sqlalchemy import Column, Integer, Float, String, DateTime
from app.core.database import Base
from datetime import datetime

class SimulationLog(Base):
    __tablename__ = "simulation_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    solar_ac = Column(Float)
    wind_ac = Column(Float)
    load = Column(Float)
    soc = Column(Float)

    action = Column(String)  # например: "charge", "discharge", "idle"

    # Можно добавить и другие поля, если хочешь хранить все 24 признака:
    hour_sin = Column(Float)
    hour_cos = Column(Float)
    reward = Column(Float)
    spill_kw = Column(Float)
    unmet_kw = Column(Float)
