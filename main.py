from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.v1.router import router as api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    root_path="/api"  # здесь задаём root_path для Nginx /api/
)

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для разработки допустимо, в проде — конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Роуты ===
app.include_router(api_router, prefix="/v1")  # убираем лишний /api

# === Автозапуск через uvicorn ===
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
