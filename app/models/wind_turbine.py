from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


class WindTurbine(Base):
    """
    Основная модель ветрогенератора
    """
    __tablename__ = "wind_turbines"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    power = Column(Float, nullable=False)  # номинальная мощность (кВт)
    location = Column(String(255), nullable=True)
    status = Column(String(50), default="Активен")
    created_at = Column(DateTime, default=datetime.utcnow)

    # связь с измерениями
    measurements = relationship("WindData", back_populates="turbine", cascade="all, delete")

    def __repr__(self):
        return f"<WindTurbine(id={self.id}, name={self.name}, power={self.power} кВт)>"
