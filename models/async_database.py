from typing import Optional, Type

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from services.interfaces import IAsyncDatabase
from settings import MysqlSettings


class AsyncDatabase(IAsyncDatabase):
    def __init__(self, config: MysqlSettings) -> None:
        self.engine: AsyncEngine = create_async_engine(
            f"mysql+aiomysql://{config.user}:{config.password}@{config.host}/{config.db}"
        )
        self._session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def __aenter__(self) -> 'AsyncDatabase':
        return self

    async def __aexit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[object]
    ) -> None:
        await self.engine.dispose()

    async def session(self) -> async_sessionmaker[AsyncSession]:
        """Provide the session maker directly."""
        return self._session_maker
