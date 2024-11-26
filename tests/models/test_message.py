from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from sqlalchemy import literal, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models.message import Message


# Mocking database session and query results
@pytest.fixture
def mock_session():
    """Fixture to create a mocked database session."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.execute = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = MagicMock()
    return mock_session


@pytest.fixture
def message_instance():
    """Fixture to create a sample message object."""
    return Message(id=1, channel="test_channel", timestamp=datetime.now(), text="Test message", score=90,
                   alt="Alternative text", score_alt=80)


# Test for get_last_message_id
@pytest.mark.asyncio
async def test_get_last_message_id(mock_session):
    """Test for the get_last_message_id method."""
    # Mock the query result
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none = Mock(return_value=123)
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await Message.get_last_message_id(mock_session, "test_channel")

    assert result == 123

    statement = (
        select(Message.id).
        where(Message.channel == literal("test_channel")).
        order_by(desc(Message.id)).
        limit(1)
    )
    assert str(statement) == str(mock_session.execute.call_args[0][0])


# Test for the case when no last message is found
@pytest.mark.asyncio
async def test_get_last_message_id_no_result(mock_session):
    """Test for the get_last_message_id method when no result is found."""
    # Mock the query result to return None
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none = Mock(return_value=None)
    mock_session.execute = AsyncMock(return_value=mock_result)

    # Call the method
    result = await Message.get_last_message_id(mock_session, "non_existent_channel")

    # Assertions
    assert result is None  # Ensure it returns None when no message is found

    # Verify the query
    statement = (
        select(Message.id)
        .where(Message.channel == literal("non_existent_channel"))
        .order_by(desc(Message.id))
        .limit(1)
    )
    assert str(statement) == str(mock_session.execute.call_args[0][0])


# Test for get_published_messages
@pytest.mark.asyncio
async def test_get_published_messages(mock_session, message_instance):
    """Test for the get_published_messages method."""

    # Create a mock result for scalars().all()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [message_instance]

    # Mock execute().scalars() to return mock_scalars
    mock_session.execute.return_value.scalars = MagicMock(return_value=mock_scalars)

    result = await Message.get_published_messages(mock_session)

    # Assertions
    assert len(result) == 1
    assert result[0].channel == "test_channel"

    # Verify the query
    last_month = datetime.now() - timedelta(days=30)
    expected_query = select(Message).where(
        Message.timestamp > last_month,
        Message.published.isnot(None)
    )
    assert str(expected_query) == str(mock_session.execute.call_args[0][0])


# Test for get_unpublished_messages
@pytest.mark.asyncio
async def test_get_unpublished_messages(mock_session, message_instance):
    """Test for the get_unpublished_messages method."""

    # Create a mock result for scalars().all()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [message_instance]

    # Mock execute().scalars() to return mock_scalars
    mock_session.execute.return_value.scalars = MagicMock(return_value=mock_scalars)

    result = await Message.get_unpublished_messages(mock_session)

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

    assert str(expected_query) == str(mock_session.execute.call_args[0][0])


# Test for saving a message
@pytest.mark.asyncio
async def test_save_message(mock_session, message_instance):
    """Test for the save method."""
    await message_instance.save(mock_session)

    mock_session.add.assert_called_once_with(message_instance)
    mock_session.commit.assert_called_once()


# Test for the case when saving a message raises an exception
@pytest.mark.asyncio
async def test_save_message_error(mock_session, message_instance):
    """Test save method when an error occurs during commit."""
    mock_session.commit.side_effect = Exception("Database error")

    with pytest.raises(Exception):
        await message_instance.save(mock_session)

    mock_session.rollback.assert_called_once()


# Test for __repr__ method
@pytest.mark.asyncio
async def test_repr(message_instance):
    """Test for the __repr__ method."""
    result = message_instance.__repr__()

    assert "Test message" in result
    assert str(message_instance.id) in result
    assert "test_channel" in result
