from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


class SolarData(Base):
    """
    Единая модель почасовых данных выработки солнечной системы
    """
    __tablename__ = "solar_data"

    id = Column(Integer, primary_key=True, index=True)
    
    # Связь с системой
    system_id = Column(Integer, ForeignKey("solar_systems.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Время измерения (обязательно с таймзоной для корректной работы)
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Иррадиация
    ghi = Column(Float, nullable=True)          # Global Horizontal Irradiance (Вт/м²)
    dni = Column(Float, nullable=True)          # Direct Normal Irradiance (Вт/м²)
    dhi = Column(Float, nullable=True)          # Diffuse Horizontal Irradiance (Вт/м²)
    poa_global = Column(Float, nullable=True)   # Plane of Array Irradiance (Вт/м²)
    
    # Погодные условия
    temp_air = Column(Float, nullable=True)     # Температура воздуха (°C)
    wind_speed = Column(Float, nullable=True)   # Скорость ветра (м/с)
    
    # Температура модулей
    temp_cell = Column(Float, nullable=True)    # Температура ячеек (°C)
    
    # Мощность
    dc_power = Column(Float, nullable=True)     # DC мощность (Вт)
    ac_power = Column(Float, nullable=True)     # AC мощность (Вт)
    
    # Дополнительно
    cloud_factor = Column(Float, nullable=True) # Коэффициент облачности (0–1)

    # Обратная связь
    system = relationship("SolarSystem", back_populates="measurements")

    def __repr__(self):
        return (
            f"<SolarData(id={self.id}, system_id={self.system_id}, "
            f"ts={self.timestamp}, ac_power={self.ac_power:.2f} W)>"
        )