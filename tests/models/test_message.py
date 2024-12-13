import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from sqlalchemy import literal, desc, asc
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
        score_alt=80,
        improve="Improved text",
        score_improve=80
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


# Test for get_message
@pytest.mark.asyncio
async def test_get_message(session, message):
    """Test for the get_message method."""
    # Mock the query result
    result = AsyncMock()
    result.scalar_one_or_none = Mock(return_value=message)
    session.execute = AsyncMock(return_value=result)

    retrieved_message = await Message.get_message(session, message.id, message.channel)

    assert retrieved_message == message

    # Verify the query
    statement = (
        select(Message)
        .where(
            Message.id == literal(message.id),
            Message.channel == literal(message.channel)
        )
    )
    assert str(statement) == str(session.execute.call_args[0][0])


@pytest.mark.asyncio
async def test_get_message_not_found(session, message):
    """Test for the get_message method when no result is found."""
    # Mock the query result
    result = AsyncMock()
    result.scalar_one_or_none = Mock(return_value=None)
    session.execute = AsyncMock(return_value=result)

    retrieved_message = await Message.get_message(session, message.id, message.channel)

    assert retrieved_message is None


@pytest.mark.asyncio
async def test_get_message_exception(session, message):
    """Test get_message for exception handling."""
    session.execute.side_effect = Exception("Database error")

    with pytest.raises(Exception, match="Database error"):
        await Message.get_message(session, message.id, message.channel)


# Test for get_first_message_id
@pytest.mark.asyncio
async def test_get_first_message_id(session):
    """Test for the get_first_message_id method."""
    # Mock the query result
    result = AsyncMock()
    result.scalars = Mock(return_value=Mock(all=Mock(return_value=[1, 2, 3])))
    session.execute = AsyncMock(return_value=result)

    first_id = await Message.get_first_message_id(session, "test_channel")

    assert first_id == 1

    # Verify the query
    statement = (
        select(Message.id)
        .where(Message.channel == literal("test_channel"))
        .order_by(asc(Message.id))
        .limit(1000)
    )
    assert str(statement) == str(session.execute.call_args[0][0])


@pytest.mark.asyncio
async def test_get_first_message_id_no_messages(session):
    """Test for get_first_message_id when no messages exist."""
    result = AsyncMock()
    result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))
    session.execute = AsyncMock(return_value=result)

    first_id = await Message.get_first_message_id(session, "test_channel")

    assert first_id is None


@pytest.mark.asyncio
async def test_get_first_message_id_exception(session):
    """Test get_first_message_id for exception handling."""
    session.execute.side_effect = Exception("Database error")

    with pytest.raises(Exception, match="Database error"):
        await Message.get_first_message_id(session, "test_channel")


# Test for update
@pytest.mark.asyncio
async def test_update_message(session, message):
    """Test for the update method."""
    # Mock get_message to return the message
    Message.get_message = AsyncMock(return_value=message)

    await message.update(session, text="Updated text", score=100)

    # Verify that attributes were updated
    assert message.text == "Updated text"
    assert message.score == 100

    # Ensure the session was committed
    session.add.assert_called_once_with(message)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_message_not_found(session, message):
    """Test for update method when the message does not exist."""
    Message.get_message = AsyncMock(return_value=None)

    await message.update(session, text="Updated text")

    # Ensure no changes were made
    session.add.assert_not_called()
    session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_update_message_exception(session, message):
    """Test update method for exception handling."""
    Message.get_message = AsyncMock(side_effect=Exception("Database error"))

    # Mock rollback as an async method
    session.rollback = AsyncMock()

    with pytest.raises(Exception, match="Database error"):
        await message.update(session, text="Updated text")

    session.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_message_invalid_field(session, message, caplog):
    """Test for update method when an invalid field is passed."""
    # Mock get_message to return the message
    Message.get_message = AsyncMock(return_value=message)

    with caplog.at_level(logging.DEBUG):
        await message.update(session, invalid_field="Invalid value")

    # Ensure the session was not updated
    session.add.assert_called_once_with(message)
    session.commit.assert_awaited_once()

    # Verify the log message
    assert any(
        "Field 'invalid_field' does not exist on Message. Ignored." in record.message
        for record in caplog.records
    )
