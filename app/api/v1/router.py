from fastapi import APIRouter
from app.api.v1 import solar, wind, simulation_log, solar_data, solar_panel

router = APIRouter()

router.include_router(solar.router, prefix="/solar", tags=["solar energy"])
router.include_router(solar_data.router, prefix="/solar_data", tags=["solar data"])
router.include_router(solar_panel.router, prefix="/solar_panels", tags=["solar panels"])
router.include_router(wind.router, prefix="/wind", tags=["wind energy"])
router.include_router(simulation_log.router, prefix="/simulation_logs", tags=["simulation logs"])
