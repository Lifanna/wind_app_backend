from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyUrl, Field


class PostgresDsn(AnyUrl):
    allowed_schemes = {"postgresql", "postgresql+asyncpg"}


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI App"
    ENV: str = Field(default="development", description="environment: development/staging/production")
    DEBUG: bool = True

    DATABASE_URL: PostgresDsn = Field(..., description="SQLAlchemy DSN для подключения к PostgreSQL")

    SECRET_KEY: str = Field(..., description="Секретный ключ для JWT и прочего")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
