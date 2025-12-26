# app/models/consumer.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from app.core.database import Base


class Consumer(Base):
    __tablename__ = "consumers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), nullable=False)
    rated_power_kw = Column(Float, nullable=False)  # номинальная мощность
    category = Column(String(32), nullable=True)     # например: "HVAC", "lighting", "EV", "appliance"
    priority = Column(Integer, default=1)            # приоритет отключения
    flexibility = Column(Float, default=0.0)         # 0–1, доля мощности, которую можно регулировать
    location = Column(String(64), nullable=True)
    description = Column(String(256), nullable=True)

    # связь с историей потребления
    measurements = relationship("ConsumerData", back_populates="consumer", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Consumer(name={self.name}, rated_power={self.rated_power_kw} kW)>"
