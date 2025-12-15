from __future__ import with_statement
from logging.config import fileConfig
from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
import asyncio
import os
import sys

# Добавляем корень проекта, чтобы видеть пакет app/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.core.config import settings
from app.core.database import Base
from app import models  # импорт всех моделей

# Настройка логов Alembic
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Устанавливаем URL базы данных
config.set_main_option("sqlalchemy.url", str(settings.DATABASE_URL))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Запуск миграций в офлайн-режиме."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Запуск миграций в онлайн-режиме (async)."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
