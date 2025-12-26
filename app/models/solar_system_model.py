from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


class SolarSystem(Base):
    __tablename__ = "solar_systems"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), default="Solar System")

    # Новые поля
    power_kw = Column(Float, nullable=True)           # Мощность панели (кВт)
    efficiency = Column(Float, nullable=True)         # КПД (%)
    status = Column(String(32), nullable=True)        # Статус

    created_at = Column(DateTime, default=datetime.utcnow)

    measurements = relationship("SolarData", back_populates="system", cascade="all, delete")
    forecasts = relationship("SolarForecast", back_populates="system", cascade="all, delete")
    panels = relationship("SolarPanel", back_populates="system")

    def __repr__(self):
        return f"<SolarSystem(id={self.id}, name='{self.name}', power_kw={self.power_kw})>"
