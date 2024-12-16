from datetime import datetime
from typing import List
from unittest.mock import Mock, AsyncMock

import pytest

from models.message import Message
from services.telegram_service import TelegramService


@pytest.fixture
def client() -> Mock:
    client = Mock()
    client.get_messages = AsyncMock()
    return client


@pytest.fixture
def telegram_service(client: Mock) -> TelegramService:
    channels = ["test_channel"]
    return TelegramService(client=client, channels=channels)


@pytest.fixture
def messages() -> List[Mock]:
    message = Mock()
    message.id = 123
    message.text = "Test message"
    message.date = datetime(2024, 1, 1, 12)
    message.reactions = Mock()
    message.reactions.results = [Mock(count=3), Mock(count=2)]

    return [message]


@pytest.mark.asyncio
async def test_fetch_messages_with_last_message_id(telegram_service: TelegramService, client: Mock, messages: List[Mock]) -> None:
    # Arrange
    client.get_messages.return_value = messages

    # Act
    result = await telegram_service.fetch_messages("test_channel", min_id=100)

    # Assert
    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert result[0].id == 123
    assert result[0].channel == "test_channel"
    assert result[0].text == "Test message"
    assert result[0].timestamp == "2024-01-01 12:00:00"
    client.get_messages.assert_called_once_with("test_channel", min_id=100, limit=None)


@pytest.mark.asyncio
async def test_fetch_messages_without_last_message_id(telegram_service: TelegramService, client: Mock, messages: List[Mock]) -> None:
    # Arrange
    client.get_messages.return_value = messages

    # Act
    result = await telegram_service.fetch_messages("test_channel")

    # Assert
    assert len(result) == 1
    client.get_messages.assert_called_once_with("test_channel", limit=10)


@pytest.mark.asyncio
async def test_fetch_messages_empty_response(telegram_service: TelegramService, client: Mock) -> None:
    # Arrange
    client.get_messages.return_value = []

    # Act
    result = await telegram_service.fetch_messages("test_channel")

    # Assert
    assert len(result) == 0


@pytest.mark.asyncio
async def test_fetch_messages_handles_exception(telegram_service: TelegramService, client: Mock) -> None:
    # Arrange
    client.get_messages.side_effect = Exception("Test error")

    # Act
    result = await telegram_service.fetch_messages("test_channel")

    # Assert
    assert len(result) == 0


@pytest.mark.asyncio
async def test_create_message_objects(messages: List[Mock]) -> None:
    # Act
    result = TelegramService._create_message_objects(messages, "test_channel")

    # Assert
    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert result[0].id == 123
    assert result[0].channel == "test_channel"
    assert result[0].text == "Test message"
    assert result[0].timestamp == "2024-01-01 12:00:00"


@pytest.mark.asyncio
async def test_get_reactions(messages: List[Mock]) -> None:
    # Act
    reactions_count = TelegramService._get_reactions(messages[0])

    # Assert
    assert reactions_count == 5  # 3 + 2 from the mocked reactions


@pytest.mark.asyncio
async def test_get_reactions_no_reactions() -> None:
    # Arrange
    message = Mock()
    message.reactions = None

    # Act
    reactions_count = TelegramService._get_reactions(message)

    # Assert
    assert reactions_count == 0


@pytest.mark.asyncio
async def test_get_messages_with_max_id(client: Mock, telegram_service: TelegramService) -> None:
    # Arrange
    max_id = 200
    client.get_messages.return_value = []

    # Act
    await telegram_service._get_messages("test_channel", max_id=max_id)

    # Assert
    client.get_messages.assert_called_once_with("test_channel", min_id=None, max_id=max_id)


@pytest.mark.asyncio
async def test_get_messages_without_min_id_and_max_id(client: Mock, telegram_service: TelegramService) -> None:
    # Act
    await telegram_service._get_messages("test_channel")

    # Assert
    client.get_messages.assert_called_once_with("test_channel", limit=10)
