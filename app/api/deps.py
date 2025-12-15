from typing import AsyncGenerator
from app.core.database import async_session_maker


async def get_db() -> AsyncGenerator:
    async with async_session_maker() as session:
        yield session
