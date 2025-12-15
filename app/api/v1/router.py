from fastapi import APIRouter
from app.api.v1 import solar, wind, simulation_log

router = APIRouter()

router.include_router(solar.router, prefix="/solar", tags=["solar energy"])
router.include_router(wind.router, prefix="/wind", tags=["wind energy"])
router.include_router(simulation_log.router, prefix="/simulation_logs", tags=["simulation logs"])
