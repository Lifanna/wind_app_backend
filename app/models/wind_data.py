from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship
from app.core.database import Base


class WindData(Base):
    __tablename__ = "wind_data"

    id = Column(Integer, primary_key=True, index=True)
    turbine_id = Column(Integer, ForeignKey("wind_turbines.id", ondelete="CASCADE"))

    timestamp = Column(DateTime(timezone=True), index=True)
    wind_speed_ref = Column(Float)
    wind_speed_hub = Column(Float)
    power_raw_W = Column(Float)
    power_curve_W = Column(Float)
    power_ac_W = Column(Float)
    rho = Column(Float)
    turbine_status = Column(String(32))

    turbine = relationship("WindTurbine", back_populates="measurements")

    def __repr__(self):
        return f"<WindData(turbine_id={self.turbine_id}, ts={self.timestamp})>"
