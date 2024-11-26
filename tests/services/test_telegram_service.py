from datetime import datetime
from unittest.mock import Mock, AsyncMock

import pytest

from models.message import Message
from services.telegram_service import TelegramService


@pytest.fixture
def mock_client():
    client = Mock()
    client.get_messages = AsyncMock()
    return client


@pytest.fixture
def telegram_service(mock_client):
    channels = ["test_channel"]
    return TelegramService(client=mock_client, channels=channels)


@pytest.mark.asyncio
async def test_fetch_messages_with_last_message_id(telegram_service, mock_client):
    # Arrange
    mock_message = Mock()
    mock_message.id = 123
    mock_message.text = "Test message"
    mock_message.date = datetime(2024, 1, 1, 12)
    mock_client.get_messages.return_value = [mock_message]

    # Act
    result = await telegram_service.fetch_messages("test_channel", last_message_id=100)

    # Assert
    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert result[0].id == 123
    assert result[0].channel == "test_channel"
    assert result[0].text == "Test message"
    assert result[0].timestamp == "2024-01-01 12:00:00"
    mock_client.get_messages.assert_called_once_with("test_channel", min_id=100, limit=None)


@pytest.mark.asyncio
async def test_fetch_messages_without_last_message_id(telegram_service, mock_client):
    # Arrange
    mock_message = Mock()
    mock_message.id = 123
    mock_message.text = "Test message"
    mock_message.date = datetime(2024, 1, 1, 12)
    mock_client.get_messages.return_value = [mock_message]

    # Act
    result = await telegram_service.fetch_messages("test_channel")

    # Assert
    assert len(result) == 1
    mock_client.get_messages.assert_called_once_with("test_channel", limit=10)


@pytest.mark.asyncio
async def test_fetch_messages_empty_response(telegram_service, mock_client):
    # Arrange
    mock_client.get_messages.return_value = []

    # Act
    result = await telegram_service.fetch_messages("test_channel")

    # Assert
    assert len(result) == 0


@pytest.mark.asyncio
async def test_fetch_messages_handles_exception(telegram_service, mock_client):
    # Arrange
    mock_client.get_messages.side_effect = Exception("Test error")

    # Act
    result = await telegram_service.fetch_messages("test_channel")

    # Assert
    assert len(result) == 0


@pytest.mark.asyncio
async def test_create_message_objects():
    # Arrange
    mock_message = Mock()
    mock_message.id = 123
    mock_message.text = "Test message"
    mock_message.date = datetime(2024, 1, 1, 12)
    messages = [mock_message]

    # Act
    result = TelegramService._create_message_objects(messages, "test_channel")

    # Assert
    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert result[0].id == 123
    assert result[0].channel == "test_channel"
    assert result[0].text == "Test message"
    assert result[0].timestamp == "2024-01-01 12:00:00"
