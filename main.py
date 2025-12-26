from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.v1.router import router as api_router
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для разработки допустимо, в проде — конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Роуты ===
app.include_router(api_router, prefix="/api/v1")


# === Автозапуск через uvicorn ===
if __name__ == "__main__":
    uvicorn.run(
        "main:app",          # имя файла : объект FastAPI
        host="0.0.0.0",
        port=8000,
        reload=True          # авто-перезапуск при изменениях (только для dev)
    )
