from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from services.interfaces import IAsyncDatabase


class AsyncDatabase(IAsyncDatabase):
    def __init__(self, host: str, user: str, password: str, db: str):
        self.engine = create_async_engine(f"mysql+aiomysql://{user}:{password}@{host}/{db}")
        self._session_maker = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def __aenter__(self) -> 'AsyncDatabase':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.engine.dispose()

    async def session(self) -> async_sessionmaker:
        """Provide the session maker directly."""
        return self._session_maker
