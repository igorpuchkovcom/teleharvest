from unittest import mock

import pytest
from telethon import TelegramClient

from container import Container
from models.async_database import AsyncDatabase
from processor import Processor
from services.embedding_service import EmbeddingService
from services.interfaces import ITelegramService, IOpenAIService, IAsyncDatabase, IEmbeddingService
from services.openai_service import OpenAIService
from services.telegram_service import TelegramService
from settings import Settings


@pytest.fixture
def mock_settings():
    mock_settings = mock.Mock(spec=Settings)
    mock_settings.telegram_api_id = 12345
    mock_settings.telegram_api_hash = 'test_api_hash'
    mock_settings.telegram_channels_list = ['channel1', 'channel2']
    mock_settings.openai_api_key = 'openai_key'
    mock_settings.openai_model = 'text-davinci'
    mock_settings.openai_max_tokens = 1000
    mock_settings.load_prompt.return_value = 'test_prompt'
    mock_settings.mysql_host = 'localhost'
    mock_settings.mysql_user = 'user'
    mock_settings.mysql_password = 'password'
    mock_settings.mysql_db = 'database'
    return mock_settings


@pytest.fixture
def container(mock_settings):
    return Container(mock_settings)


def test_get_telegram_client(container, mock_settings):
    with mock.patch.object(TelegramClient, '__init__', return_value=None):
        telegram_client = container.get_telegram_client()
        assert telegram_client is not None


def test_get_telegram_service(container, mock_settings):
    with mock.patch.object(TelegramService, '__init__', return_value=None):
        telegram_service = container.get_telegram_service()
        assert isinstance(telegram_service, ITelegramService)
        assert telegram_service is not None


def test_get_openai_service(container, mock_settings):
    with mock.patch.object(OpenAIService, '__init__', return_value=None):
        openai_service = container.get_openai_service()
        assert isinstance(openai_service, IOpenAIService)
        assert openai_service is not None


def test_get_database(container, mock_settings):
    with mock.patch.object(AsyncDatabase, '__init__', return_value=None):
        database = container.get_database()
        assert isinstance(database, IAsyncDatabase)
        assert database is not None


def test_get_embedding_service(container, mock_settings):
    with mock.patch.object(EmbeddingService, '__init__', return_value=None):
        embedding_service = container.get_embedding_service()
        assert isinstance(embedding_service, IEmbeddingService)
        assert embedding_service is not None


def test_get_processor(container, mock_settings):
    with mock.patch.object(Processor, '__init__', return_value=None):
        processor = container.get_processor()
        assert isinstance(processor, Processor)
        assert processor is not None
