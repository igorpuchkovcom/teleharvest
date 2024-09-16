from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session

from services.interfaces import IAsyncDatabase


class AsyncDatabase(IAsyncDatabase):
    def __init__(self, host: str, user: str, password: str, db: str):
        self.engine = create_async_engine(f"mysql+aiomysql://{user}:{password}@{host}/{db}")

    async def __aenter__(self) -> 'AsyncDatabase':
        self.session = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.engine.dispose()

    async def session(self) -> async_sessionmaker[Session]:
        return self.session()
