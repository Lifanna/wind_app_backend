# app/models/solar_panel.py
from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


class SolarPanel(Base):
    """
    Отдельная солнечная панель в составе станции
    """
    __tablename__ = "solar_panels"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    power = Column(Float, nullable=False)        # мощность панели (кВт)
    efficiency = Column(Float, nullable=False)   # КПД (%)
    status = Column(String(50), default="Активна")
    created_at = Column(DateTime, default=datetime.utcnow)

    # связь с системой
    system_id = Column(Integer, ForeignKey("solar_systems.id", ondelete="CASCADE"))
    system = relationship("SolarSystem", back_populates="panels")

    def __repr__(self):
        return f"<SolarPanel(id={self.id}, name={self.name}, power={self.power} кВт)>"
