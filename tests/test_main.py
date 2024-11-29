from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from main import main


@pytest.fixture
def settings():
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
def container(settings, telegram_client_mock, processor_mock):
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
async def test_main_success(logger, container_class, settings_class, settings, container):
    """Test main function with successful execution."""
    settings_class.return_value = settings
    container_class.return_value = container

    # Mock logger to capture the logs
    logger.return_value = MagicMock()

    # Run the main function
    await main()

    # Assert the settings are initialized
    settings_class.assert_called_once()

    # Assert the container is initialized with settings
    container_class.assert_called_once_with(settings)

    # Check TelegramClient interactions
    telegram_client = container.get_telegram_client.return_value.__aenter__.return_value
    telegram_client.start.assert_called_once_with(phone=settings.telegram_phone)

    # Check Processor interactions
    processor = container.get_processor.return_value.__aenter__.return_value
    processor.fetch_and_process.assert_called_once()
    processor.update_similarity.assert_called_once()


@pytest.mark.asyncio
async def test_main_telegram_client_exception():
    # Mock all the dependencies
    settings = MagicMock()
    settings.telegram_phone = "+1234567890"

    container = MagicMock()
    client = AsyncMock()

    # Configure the client to raise an exception
    client.start.side_effect = Exception("Telegram client error")
    container.get_telegram_client.return_value.__aenter__.return_value = client

    # Patch the dependencies and logger
    with patch('main.Settings', return_value=settings), \
            patch('main.Container', return_value=container), \
            patch('main.logging.getLogger') as logger:
        await main()

        # Assert that the error was logged
        logger.return_value.error.assert_called_once_with(
            "An error occurred: Telegram client error"
        )


@pytest.mark.asyncio
async def test_main_processor_exception():
    # Mock all the dependencies
    settings = MagicMock()
    settings.telegram_phone = "+1234567890"

    container = MagicMock()
    client = AsyncMock()
    processor = AsyncMock()

    # Configure the processor to raise an exception
    processor.fetch_and_process.side_effect = Exception("Processor error")
    container.get_telegram_client.return_value.__aenter__.return_value = client
    container.get_processor.return_value.__aenter__.return_value = processor

    # Patch the dependencies and logger
    with patch('main.Settings', return_value=settings), \
            patch('main.Container', return_value=container), \
            patch('main.logging.getLogger') as logger:
        await main()

        # Assert that the error was logged
        logger.return_value.error.assert_called_once_with(
            "An error occurred: Processor error"
        )
