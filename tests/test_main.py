import logging
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from main import main


@pytest.fixture
def telegram_client_mock() -> AsyncMock:
    """Mock for Telegram client."""
    client = AsyncMock()
    client.start = AsyncMock()
    return client


@pytest.fixture
def processor_mock() -> AsyncMock:
    """Mock for Processor."""
    processor = AsyncMock()
    processor.fetch_and_process = AsyncMock()
    processor.update_similarity = AsyncMock()
    return processor


@pytest.fixture
def container(telegram_client_mock: AsyncMock, processor_mock: AsyncMock) -> MagicMock:
    """Mock for Container."""
    container = MagicMock()
    container.get_telegram_client.return_value.__aenter__.return_value = telegram_client_mock
    container.get_telegram_client.return_value.__aexit__ = AsyncMock()
    container.get_processor.return_value.__aenter__.return_value = processor_mock
    container.get_processor.return_value.__aexit__ = AsyncMock()
    return container


@patch("main.Container", autospec=True)
@patch("main.logging.getLogger", autospec=True)
@pytest.mark.asyncio
async def test_main_success(logger: MagicMock, container_class: MagicMock, container: MagicMock) -> None:
    """Test main function with successful execution."""
    container_class.return_value = container

    # Mock logger to capture the logs
    logger.return_value = MagicMock()

    # Run the main function
    await main()

    # Check Processor interactions
    processor = container.get_processor.return_value.__aenter__.return_value
    processor.fetch_and_process.assert_called_once()
    processor.update_similarity.assert_called_once()


@pytest.mark.asyncio
async def test_main_telegram_client_exception() -> None:
    container = MagicMock()
    client = AsyncMock()

    # Configure the client to raise an exception
    client.start.side_effect = Exception("Telegram client error")
    container.get_telegram_client.return_value.__aenter__.return_value = client

    # Patch the dependencies and logger
    with patch('main.Container', return_value=container), patch('main.logging.getLogger') as logger:
        await main()

        # Assert that the error was logged
        logger.return_value.error.assert_called_once_with(
            "An error occurred: Telegram client error"
        )


@pytest.mark.asyncio
async def test_main_processor_exception() -> None:
    container = MagicMock()
    client = AsyncMock()
    processor = AsyncMock()

    # Configure the processor to raise an exception
    processor.fetch_and_process.side_effect = Exception("Processor error")
    container.get_telegram_client.return_value.__aenter__.return_value = client
    container.get_processor.return_value.__aenter__.return_value = processor

    # Patch the dependencies and logger
    with patch('main.Container', return_value=container), patch('main.logging.getLogger') as logger:
        await main()

        # Assert that the error was logged
        logger.return_value.error.assert_called_once_with(
            "An error occurred: Processor error"
        )


@patch("main.logging.getLogger")
@pytest.mark.asyncio
async def test_main_invalid_log_level(get_logger_mock: MagicMock) -> None:
    """Test main function with an invalid log level."""
    # Mock the logger
    main_logger = MagicMock()
    httpx_logger = MagicMock()

    get_logger_mock.side_effect = lambda name: main_logger if name == "main" else httpx_logger

    # Mock asynchronous dependencies to avoid any real execution
    with patch("main.Container") as container_mock, \
            patch.dict(os.environ, {
                "LOG_LEVEL": "INVALID_LEVEL"
            }):
        container_mock.return_value.get_telegram_client.return_value.__aenter__.return_value = AsyncMock()
        container_mock.return_value.get_processor.return_value.__aenter__.return_value = AsyncMock()

        # Run the main function
        await main()

        # Assertions
        get_logger_mock.assert_any_call("main")
        get_logger_mock.assert_any_call("httpx")
        httpx_logger.setLevel.assert_called_once_with(logging.WARNING)
        main_logger.info.assert_any_call("Starting TelegramClient with phone: +1234567890")
        main_logger.info.assert_any_call("Fetching and processing messages")
        main_logger.info.assert_any_call("Updating similarity score")
        main_logger.info.assert_any_call("Updating metrics")
