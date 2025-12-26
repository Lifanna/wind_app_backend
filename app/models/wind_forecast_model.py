from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base


class WindForecast(Base):
    __tablename__ = "wind_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    turbine_id = Column(Integer, ForeignKey("wind_turbines.id", ondelete="CASCADE"))
    forecast_time = Column(DateTime(timezone=True), index=True)  # время, когда сделан прогноз
    target_time = Column(DateTime(timezone=True), index=True)    # время, на которое прогнозируется

    wind_speed_forecast = Column(Float)
    wind_power_kw_forecast = Column(Float)

    turbine = relationship("WindTurbine", backref="forecasts")

    def __repr__(self):
        return f"<WindForecast(turbine={self.turbine_id}, target={self.target_time}, power={self.wind_power_kw_forecast:.2f} kW)>"
