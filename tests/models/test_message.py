from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from sqlalchemy import literal, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models.message import Message


# Mocking database session and query results
@pytest.fixture
def session():
    """Fixture to create a mocked database session."""
    session = MagicMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = MagicMock()
    return session


@pytest.fixture
def message():
    """Fixture to create a sample message object."""
    return Message(
        id=1,
        channel="test_channel",
        timestamp=datetime.now(),
        text="Test message",
        score=90,
        alt="Alternative text",
        score_alt=80
    )


# Test for get_last_message_id
@pytest.mark.asyncio
async def test_get_last_message_id(session):
    """Test for the get_last_message_id method."""
    # Mock the query result
    result = AsyncMock()
    result.scalar_one_or_none = Mock(return_value=123)
    session.execute = AsyncMock(return_value=result)

    result = await Message.get_last_message_id(session, "test_channel")

    assert result == 123

    statement = (
        select(Message.id).
        where(Message.channel == literal("test_channel")).
        order_by(desc(Message.id)).
        limit(1)
    )
    assert str(statement) == str(session.execute.call_args[0][0])


# Test for the case when no last message is found
@pytest.mark.asyncio
async def test_get_last_message_id_no_result(session):
    """Test for the get_last_message_id method when no result is found."""
    # Mock the query result to return None
    result = AsyncMock()
    result.scalar_one_or_none = Mock(return_value=None)
    session.execute = AsyncMock(return_value=result)

    # Call the method
    result = await Message.get_last_message_id(session, "non_existent_channel")

    # Assertions
    assert result is None  # Ensure it returns None when no message is found

    # Verify the query
    statement = (
        select(Message.id)
        .where(Message.channel == literal("non_existent_channel"))
        .order_by(desc(Message.id))
        .limit(1)
    )
    assert str(statement) == str(session.execute.call_args[0][0])


# Test for get_published_messages
@pytest.mark.asyncio
async def test_get_published_messages(session, message):
    """Test for the get_published_messages method."""

    # Create a mock result for scalars().all()
    scalars = MagicMock()
    scalars.all.return_value = [message]

    # Mock execute().scalars() to return scalars
    session.execute.return_value.scalars = MagicMock(return_value=scalars)

    result = await Message.get_published_messages(session)

    # Assertions
    assert len(result) == 1
    assert result[0].channel == "test_channel"

    # Verify the query
    last_month = datetime.now() - timedelta(days=30)
    expected_query = select(Message).where(
        Message.timestamp > last_month,
        Message.published.isnot(None)
    )
    assert str(expected_query) == str(session.execute.call_args[0][0])


# Test for get_unpublished_messages
@pytest.mark.asyncio
async def test_get_unpublished_messages(session, message):
    """Test for the get_unpublished_messages method."""

    # Create a mock result for scalars().all()
    scalars = MagicMock()
    scalars.all.return_value = [message]

    # Mock execute().scalars() to return scalars
    session.execute.return_value.scalars = MagicMock(return_value=scalars)

    result = await Message.get_unpublished_messages(session)

    # Assertions
    assert len(result) == 1
    assert result[0].channel == "test_channel"

    # Verify the query
    expected_query = (
        select(Message).where(
            Message.embedding.isnot(None),
            Message.published.is_(None)
        )
    )

    assert str(expected_query) == str(session.execute.call_args[0][0])


# Test for saving a message
@pytest.mark.asyncio
async def test_save_message(session, message):
    """Test for the save method."""
    await message.save(session)

    session.add.assert_called_once_with(message)
    session.commit.assert_called_once()


# Test for the case when saving a message raises an exception
@pytest.mark.asyncio
async def test_save_message_error(session, message):
    """Test save method when an error occurs during commit."""
    session.commit.side_effect = Exception("Database error")

    with pytest.raises(Exception):
        await message.save(session)

    session.rollback.assert_called_once()


# Test for __repr__ method
@pytest.mark.asyncio
async def test_repr(message):
    """Test for the __repr__ method."""
    result = message.__repr__()

    assert "Test message" in result
    assert str(message.id) in result
    assert "test_channel" in result


@pytest.mark.asyncio
async def test_get_last_message_id_exception(session):
    """Test get_last_message_id for exception handling."""

    session.execute.side_effect = Exception("Database error")

    with pytest.raises(Exception, match="Database error"):
        await Message.get_last_message_id(session, "test_channel")


@pytest.mark.asyncio
async def test_get_published_messages_exception(session):
    """Test get_published_messages for exception handling."""

    session.execute.side_effect = Exception("Database error")

    with pytest.raises(Exception, match="Database error"):
        await Message.get_published_messages(session)


@pytest.mark.asyncio
async def test_get_unpublished_messages_exception(session):
    """Test get_unpublished_messages for exception handling."""

    session.execute.side_effect = Exception("Database error")

    with pytest.raises(Exception, match="Database error"):
        await Message.get_unpublished_messages(session)
