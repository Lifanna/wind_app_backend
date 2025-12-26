from datetime import datetime
from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship
from app.core.database import Base


class WindData(Base):
    """
    Модель данных измерений ветрогенератора
    """
    __tablename__ = "wind_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Внешний ключ на ветрогенератор
    wind_turbine_id = Column(Integer, ForeignKey("wind_turbines.id", ondelete="CASCADE"), nullable=False, default=1)

    wind_speed_ref = Column(Float, nullable=False)
    wind_speed = Column(Float, nullable=False)
    power_raw_W = Column(Float, nullable=False)
    power_curve_W = Column(Float, nullable=False)
    power_ac_W = Column(Float, nullable=False)
    rho = Column(Float, nullable=False)  # Плотность воздуха
    rotor_diameter_m = Column(Float, nullable=False)
    rated_power_W = Column(Float, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Обратная связь с ветрогенератором
    turbine = relationship("WindTurbine", back_populates="measurements")

    def __repr__(self):
        return f"<WindData(id={self.id}, turbine_id={self.wind_turbine_id}, wind_speed={self.wind_speed})>"
    

