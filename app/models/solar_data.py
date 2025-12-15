from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base


class SolarData(Base):
    """
    Почасовые данные выработки солнечной системы
    """
    __tablename__ = "solar_data"

    id = Column(Integer, primary_key=True, index=True)
    system_id = Column(Integer, ForeignKey("solar_systems.id", ondelete="CASCADE"))

    timestamp = Column(DateTime(timezone=True), index=True)

    ghi = Column(Float)
    dni = Column(Float)
    dhi = Column(Float)
    poa_global = Column(Float)
    temp_air = Column(Float)
    wind_speed = Column(Float)
    temp_cell = Column(Float)
    dc_power = Column(Float)
    ac_power = Column(Float)
    cloud_factor = Column(Float)

    system = relationship("SolarSystem", back_populates="measurements")

    def __repr__(self):
        return f"<SolarData(system_id={self.system_id}, ts={self.timestamp}, ac={self.ac_power:.2f} W)>"
