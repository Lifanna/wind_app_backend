from sqlalchemy.orm import Session
from app.models.simulation_log_model import SimulationLog

def get_all_logs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(SimulationLog).order_by(SimulationLog.timestamp).offset(skip).limit(limit).all()

def create_log(db: Session, log_data: dict):
    log = SimulationLog(**log_data)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
