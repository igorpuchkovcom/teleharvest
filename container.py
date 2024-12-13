from telethon import TelegramClient

from models.async_database import AsyncDatabase
from processor import Processor
from services.embedding_service import EmbeddingService
from services.interfaces import ITelegramService, IOpenAIService, IAsyncDatabase, IEmbeddingService
from services.openai_service import OpenAIService
from services.telegram_service import TelegramService
from settings import Settings


class Container:
    def __init__(self, settings: Settings):
        self.settings: Settings = settings
        self._services = {}

    def _get_service(self, service_name: str, constructor: callable, *args, **kwargs):
        if service_name not in self._services:
            self._services[service_name] = constructor(*args, **kwargs)
        return self._services[service_name]

    def get_telegram_client(self) -> TelegramClient:
        return self._get_service(
            'telegram_client',
            TelegramClient,
            'session_name',
            self.settings.telegram.api_id,
            self.settings.telegram.api_hash
        )

    def get_telegram_service(self) -> ITelegramService:
        return self._get_service(
            'telegram_service',
            TelegramService,
            self.get_telegram_client(),
            self.settings.telegram.channels_list
        )

    def get_openai_service(self) -> IOpenAIService:
        return self._get_service(
            'openai_service',
            OpenAIService,
            self.settings.openai,
            self.settings.load_prompt('process.txt'),
            self.settings.load_prompt('evaluate.txt'),
            self.settings.load_prompt('improve.txt')
        )

    def get_database(self) -> IAsyncDatabase:
        return self._get_service(
            'db',
            AsyncDatabase,
            self.settings.mysql
        )

    def get_embedding_service(self) -> IEmbeddingService:
        return self._get_service(
            'embedding_service',
            EmbeddingService
        )

    def get_processor(self) -> Processor:

        return self._get_service(
            'processor',
            Processor,
            self.get_telegram_service(),
            self.get_openai_service(),
            self.get_database(),
            self.get_embedding_service(),
            self.settings.processor,
        )
