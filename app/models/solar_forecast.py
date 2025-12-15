from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship
from app.core.database import Base


class SolarForecast(Base):
    """
    Прогноз генерации солнечной станции
    """
    __tablename__ = "solar_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    system_id = Column(Integer, ForeignKey("solar_systems.id", ondelete="CASCADE"))

    timestamp = Column(DateTime(timezone=True), index=True)
    horizon_hours = Column(Integer, default=6)
    predicted_power = Column(Float)
    model_name = Column(String(64), default="RandomForest")
    mae = Column(Float)
    r2 = Column(Float)

    system = relationship("SolarSystem", back_populates="forecasts")

    def __repr__(self):
        return f"<SolarForecast(system_id={self.system_id}, horizon={self.horizon_hours}h, pred={self.predicted_power:.2f} W)>"
