from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from main import main


@pytest.fixture
def settings_mock():
    """Mock for Settings."""
    return MagicMock(telegram_phone="+1234567890")


@pytest.fixture
def telegram_client_mock():
    """Mock for Telegram client."""
    client = AsyncMock()
    client.start = AsyncMock()
    return client


@pytest.fixture
def processor_mock():
    """Mock for Processor."""
    processor = AsyncMock()
    processor.fetch_and_process = AsyncMock()
    processor.update_similarity = AsyncMock()
    return processor


@pytest.fixture
def container_mock(settings_mock, telegram_client_mock, processor_mock):
    """Mock for Container."""
    container = MagicMock()
    container.get_telegram_client.return_value.__aenter__.return_value = telegram_client_mock
    container.get_telegram_client.return_value.__aexit__ = AsyncMock()
    container.get_processor.return_value.__aenter__.return_value = processor_mock
    container.get_processor.return_value.__aexit__ = AsyncMock()
    return container


@patch("main.Settings", autospec=True)
@patch("main.Container", autospec=True)
@patch("main.logging.getLogger", autospec=True)
@pytest.mark.asyncio
async def test_main_success(mock_logger, mock_container_class, mock_settings_class, settings_mock, container_mock):
    """Test main function with successful execution."""
    mock_settings_class.return_value = settings_mock
    mock_container_class.return_value = container_mock

    # Mock logger to capture the logs
    mock_logger.return_value = MagicMock()

    # Run the main function
    await main()

    # Assert the settings are initialized
    mock_settings_class.assert_called_once()

    # Assert the container is initialized with settings
    mock_container_class.assert_called_once_with(settings_mock)

    # Check TelegramClient interactions
    telegram_client = container_mock.get_telegram_client.return_value.__aenter__.return_value
    telegram_client.start.assert_called_once_with(phone=settings_mock.telegram_phone)

    # Check Processor interactions
    processor = container_mock.get_processor.return_value.__aenter__.return_value
    processor.fetch_and_process.assert_called_once()
    processor.update_similarity.assert_called_once()


@pytest.mark.asyncio
async def test_main_telegram_client_exception():
    # Mock all the dependencies
    mock_settings = MagicMock()
    mock_settings.telegram_phone = "+1234567890"

    mock_container = MagicMock()
    mock_client = AsyncMock()

    # Configure the client to raise an exception
    mock_client.start.side_effect = Exception("Telegram client error")
    mock_container.get_telegram_client.return_value.__aenter__.return_value = mock_client

    # Patch the dependencies and logger
    with patch('main.Settings', return_value=mock_settings), \
            patch('main.Container', return_value=mock_container), \
            patch('main.logging.getLogger') as mock_logger:
        await main()

        # Assert that the error was logged
        mock_logger.return_value.error.assert_called_once_with(
            "An error occurred: Telegram client error"
        )


@pytest.mark.asyncio
async def test_main_processor_exception():
    # Mock all the dependencies
    mock_settings = MagicMock()
    mock_settings.telegram_phone = "+1234567890"

    mock_container = MagicMock()
    mock_client = AsyncMock()
    mock_processor = AsyncMock()

    # Configure the processor to raise an exception
    mock_processor.fetch_and_process.side_effect = Exception("Processor error")
    mock_container.get_telegram_client.return_value.__aenter__.return_value = mock_client
    mock_container.get_processor.return_value.__aenter__.return_value = mock_processor

    # Patch the dependencies and logger
    with patch('main.Settings', return_value=mock_settings), \
            patch('main.Container', return_value=mock_container), \
            patch('main.logging.getLogger') as mock_logger:
        await main()

        # Assert that the error was logged
        mock_logger.return_value.error.assert_called_once_with(
            "An error occurred: Processor error"
        )
