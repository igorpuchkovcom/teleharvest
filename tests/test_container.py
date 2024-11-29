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
def settings():
    settings = mock.Mock(spec=Settings)
    settings.telegram_api_id = 12345
    settings.telegram_api_hash = 'test_api_hash'
    settings.telegram_channels_list = ['channel1', 'channel2']
    settings.openai_api_key = 'openai_key'
    settings.openai_model = 'text-davinci'
    settings.openai_max_tokens = 1000
    settings.load_prompt.return_value = 'test_prompt'
    settings.mysql_host = 'localhost'
    settings.mysql_user = 'user'
    settings.mysql_password = 'password'
    settings.mysql_db = 'database'
    return settings


@pytest.fixture
def container(settings):
    return Container(settings)


def test_get_telegram_client(container, settings):
    with mock.patch.object(TelegramClient, '__init__', return_value=None):
        telegram_client = container.get_telegram_client()
        assert telegram_client is not None


def test_get_telegram_service(container, settings):
    with mock.patch.object(TelegramService, '__init__', return_value=None):
        telegram_service = container.get_telegram_service()
        assert isinstance(telegram_service, ITelegramService)
        assert telegram_service is not None


def test_get_openai_service(container, settings):
    with mock.patch.object(OpenAIService, '__init__', return_value=None):
        openai_service = container.get_openai_service()
        assert isinstance(openai_service, IOpenAIService)
        assert openai_service is not None


def test_get_database(container, settings):
    with mock.patch.object(AsyncDatabase, '__init__', return_value=None):
        database = container.get_database()
        assert isinstance(database, IAsyncDatabase)
        assert database is not None


def test_get_embedding_service(container, settings):
    with mock.patch.object(EmbeddingService, '__init__', return_value=None):
        embedding_service = container.get_embedding_service()
        assert isinstance(embedding_service, IEmbeddingService)
        assert embedding_service is not None


def test_get_processor(container, settings):
    with mock.patch.object(Processor, '__init__', return_value=None):
        processor = container.get_processor()
        assert isinstance(processor, Processor)
        assert processor is not None
