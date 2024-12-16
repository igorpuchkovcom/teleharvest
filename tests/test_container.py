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
def settings() -> Settings:
    settings = Settings()
    return settings


@pytest.fixture
def container(settings: Settings) -> Container:
    return Container(settings)


def test_get_telegram_service(container: Container) -> None:
    with mock.patch.object(TelegramService, '__init__', return_value=None):
        telegram_service: ITelegramService = container.get_telegram_service()
        assert isinstance(telegram_service, ITelegramService)
        assert telegram_service is not None


def test_get_openai_service(container: Container) -> None:
    with mock.patch.object(OpenAIService, '__init__', return_value=None):
        openai_service: IOpenAIService = container.get_openai_service()
        assert isinstance(openai_service, IOpenAIService)
        assert openai_service is not None


def test_get_database(container: Container) -> None:
    with mock.patch.object(AsyncDatabase, '__init__', return_value=None):
        database: IAsyncDatabase = container.get_database()
        assert isinstance(database, IAsyncDatabase)
        assert database is not None


def test_get_embedding_service(container: Container) -> None:
    with mock.patch.object(EmbeddingService, '__init__', return_value=None):
        embedding_service: IEmbeddingService = container.get_embedding_service()
        assert isinstance(embedding_service, IEmbeddingService)
        assert embedding_service is not None


def test_get_processor(container: Container) -> None:
    with mock.patch.object(TelegramClient, '__init__', return_value=None), \
         mock.patch.object(TelegramClient, 'connect', return_value=None), \
         mock.patch.object(TelegramClient, 'disconnect', return_value=None), \
         mock.patch.object(Processor, '__init__', return_value=None):
        processor: Processor = container.get_processor()
        assert isinstance(processor, Processor)
        assert processor is not None
