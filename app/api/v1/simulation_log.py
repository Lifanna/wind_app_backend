from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.crud.simulation_log import get_all_logs
from app.schemas.simulation_log import SimulationLogOut

router = APIRouter()

@router.get("/", response_model=list[SimulationLogOut])
def read_simulation_logs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    logs = get_all_logs(db, skip=skip, limit=limit)
    return logs
