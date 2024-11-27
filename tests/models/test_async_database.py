import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models.async_database import AsyncDatabase


@pytest.fixture
def db_settings():
    """Provide test database settings."""
    return {
        "host": "localhost",
        "user": "test_user",
        "password": "test_password",
        "db": "test_db"
    }


@pytest.fixture
async def async_database(db_settings):
    """Fixture to initialize AsyncDatabase instance."""
    return AsyncDatabase(**db_settings)


@pytest.mark.asyncio
async def test_async_database_initialization(db_settings):
    """Test AsyncDatabase initialization."""
    db = AsyncDatabase(**db_settings)
    assert db.engine is not None
    assert db.engine.url.host == db_settings["host"]
    assert db.engine.url.database == db_settings["db"]


@pytest.mark.asyncio
async def test_async_database_context_manager(db_settings):
    """Test AsyncDatabase context manager functionality."""
    async with AsyncDatabase(**db_settings) as db:
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
