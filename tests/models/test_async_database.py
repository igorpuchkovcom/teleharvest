import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

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
    db = AsyncDatabase(**db_settings)
    async with db as session:
        yield session


@pytest.fixture
async def test_engine(db_settings):
    """Fixture to create a test database engine."""
    engine = create_async_engine(
        f"mysql+aiomysql://{db_settings['user']}:{db_settings['password']}@{db_settings['host']}/{db_settings['db']}"
    )
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session(test_engine):
    """Fixture to create an async session for testing."""
    async_session_maker = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    async with async_session_maker() as session:
        yield session


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
