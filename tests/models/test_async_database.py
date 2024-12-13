import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models.async_database import AsyncDatabase
from settings import Settings


@pytest.fixture
def config():
    settings = Settings()
    return settings.mysql


@pytest.fixture
async def async_database(config):
    """Fixture to initialize AsyncDatabase instance."""
    return AsyncDatabase(config)


@pytest.mark.asyncio
async def test_async_database_initialization(config):
    """Test AsyncDatabase initialization."""
    db = AsyncDatabase(config)
    assert db.engine is not None
    assert db.engine.url.host == config.host
    assert db.engine.url.database == config.db


@pytest.mark.asyncio
async def test_async_database_context_manager(config):
    """Test AsyncDatabase context manager functionality."""
    async with AsyncDatabase(config) as db:
        assert db.session is not None
        assert callable(db.session)


@pytest.mark.asyncio
async def test_async_database_session(async_database):
    """Test the session method of AsyncDatabase."""

    database = await async_database

    database._session = async_sessionmaker(
        class_=AsyncSession,
        expire_on_commit=False
    )

    session_maker = await database.session()

    assert session_maker is not None
    assert isinstance(session_maker, async_sessionmaker)
