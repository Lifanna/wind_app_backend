# app/models/consumer_data.py
from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base


class ConsumerData(Base):
    __tablename__ = "consumer_data"

    id = Column(Integer, primary_key=True, index=True)
    consumer_id = Column(Integer, ForeignKey("consumers.id", ondelete="CASCADE"))
    timestamp = Column(DateTime(timezone=True), index=True)
    load_kw = Column(Float, nullable=False)  # фактическая мощность (кВт)

    consumer = relationship("Consumer", back_populates="measurements")

    def __repr__(self):
        return f"<ConsumerData(consumer_id={self.consumer_id}, load_kw={self.load_kw:.2f} kW)>"
