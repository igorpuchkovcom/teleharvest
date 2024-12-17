from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import aiohttp
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlalchemy.orm import Session
from telethon import TelegramClient

from models.message import Message


class ITelegramService(ABC):
    client: TelegramClient
    channels: List[str]

    @abstractmethod
    async def fetch_messages(self, channel: str, min_id: Optional[int] = None, max_id: Optional[int] = None) -> List[Message]:
        raise NotImplementedError


class IOpenAIService(ABC):
    api_key: str
    model: str
    max_tokens: int
    prompt_process: str
    prompt_evaluate: str
    session: Optional[aiohttp.ClientSession]

    @abstractmethod
    async def __aenter__(self) -> 'IOpenAIService':
        raise NotImplementedError

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_evaluation(self, text: str) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    async def get_alt(self, text: str) -> Optional[str]:
        raise NotImplementedError


class IAsyncDatabase(ABC):
    engine: AsyncEngine
    session: async_sessionmaker[Session]

    @abstractmethod
    async def __aenter__(self) -> 'IAsyncDatabase':
        raise NotImplementedError

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        raise NotImplementedError

    @abstractmethod
    async def session(self) -> async_sessionmaker[Session]:
        raise NotImplementedError


class IEmbeddingService(ABC):
    model: SentenceTransformer

    @abstractmethod
    async def generate_embedding(self, text: str) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    async def calculate_max_similarity(self, embedding: List[float], messages: Sequence['Message']) -> float:
        raise NotImplementedError
