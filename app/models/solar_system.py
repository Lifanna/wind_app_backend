from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


class SolarSystem(Base):
    """
    Конфигурация солнечной станции
    """
    __tablename__ = "solar_systems"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), default="Default Solar System")

    latitude = Column(Float)
    longitude = Column(Float)
    timezone = Column(String(64))
    tilt = Column(Float)
    azimuth = Column(Float)
    target_kw = Column(Float)
    module_power_stc = Column(Float)
    albedo = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    # связи
    measurements = relationship("SolarData", back_populates="system", cascade="all, delete")
    forecasts = relationship("SolarForecast", back_populates="system", cascade="all, delete")
    panels = relationship("SolarPanel", back_populates="system", cascade="all, delete")

    def __repr__(self):
        return f"<SolarSystem(id={self.id}, target_kw={self.target_kw}, tilt={self.tilt})>"
