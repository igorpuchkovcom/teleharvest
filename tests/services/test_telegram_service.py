from datetime import datetime
from unittest.mock import Mock, AsyncMock

import pytest

from models.message import Message
from services.telegram_service import TelegramService


@pytest.fixture
def client():
    client = Mock()
    client.get_messages = AsyncMock()
    return client


@pytest.fixture
def telegram_service(client):
    channels = ["test_channel"]
    return TelegramService(client=client, channels=channels)


@pytest.fixture
def messages():
    message = Mock()
    message.id = 123
    message.text = "Test message"
    message.date = datetime(2024, 1, 1, 12)

    return [message]


@pytest.mark.asyncio
async def test_fetch_messages_with_last_message_id(telegram_service, client, messages):
    # Arrange
    client.get_messages.return_value = messages

    # Act
    result = await telegram_service.fetch_messages("test_channel", last_message_id=100)

    # Assert
    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert result[0].id == 123
    assert result[0].channel == "test_channel"
    assert result[0].text == "Test message"
    assert result[0].timestamp == "2024-01-01 12:00:00"
    client.get_messages.assert_called_once_with("test_channel", min_id=100, limit=None)


@pytest.mark.asyncio
async def test_fetch_messages_without_last_message_id(telegram_service, client, messages):
    # Arrange
    client.get_messages.return_value = messages

    # Act
    result = await telegram_service.fetch_messages("test_channel")

    # Assert
    assert len(result) == 1
    client.get_messages.assert_called_once_with("test_channel", limit=10)


@pytest.mark.asyncio
async def test_fetch_messages_empty_response(telegram_service, client):
    # Arrange
    client.get_messages.return_value = []

    # Act
    result = await telegram_service.fetch_messages("test_channel")

    # Assert
    assert len(result) == 0


@pytest.mark.asyncio
async def test_fetch_messages_handles_exception(telegram_service, client):
    # Arrange
    client.get_messages.side_effect = Exception("Test error")

    # Act
    result = await telegram_service.fetch_messages("test_channel")

    # Assert
    assert len(result) == 0


@pytest.mark.asyncio
async def test_create_message_objects(messages):
    # Act
    result = TelegramService._create_message_objects(messages, "test_channel")

    # Assert
    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert result[0].id == 123
    assert result[0].channel == "test_channel"
    assert result[0].text == "Test message"
    assert result[0].timestamp == "2024-01-01 12:00:00"
