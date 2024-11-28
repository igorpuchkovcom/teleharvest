import asyncio
import json
import logging
import re
from typing import List, Optional, Type, Sequence

from models.message import Message
from services.interfaces import ITelegramService, IOpenAIService, IAsyncDatabase, IEmbeddingService

MIN_LEN = 200
MIN_SCORE = 85
MIN_SCORE_ALT = 90
STOP_WORDS = ["эфир", "запись", "астролог", "зодиак", "таро", "эзотери"]

logger = logging.getLogger(__name__)


class Processor:
    published_messages: Sequence[Message]

    def __init__(
            self,
            telegram_service: ITelegramService,
            openai_service: IOpenAIService,
            db: IAsyncDatabase,
            embedding_service: IEmbeddingService
    ):
        self.telegram_service = telegram_service
        self.openai_service = openai_service
        self.db = db
        self.embedding_service = embedding_service
        self.published_messages = []

    async def __aenter__(self) -> 'Processor':
        await asyncio.gather(
            self.openai_service.__aenter__(),
            self.db.__aenter__()
        )
        return self

    async def __aexit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[Type[BaseException]],
    ) -> None:
        await asyncio.gather(
            self.openai_service.__aexit__(exc_type, exc_val, exc_tb),
            self.db.__aexit__(exc_type, exc_val, exc_tb)
        )

    async def fetch_and_process(self):
        session_maker = await self.db.session()
        async with session_maker() as session:
            self.published_messages: Sequence[Message] = await Message.get_published_messages(session)
            for channel in self.telegram_service.channels:
                last_message_id = await Message.get_last_message_id(session, channel)
                messages = await self.telegram_service.fetch_messages(channel, last_message_id)
                await self.process(messages)

    async def process(self, messages: List[Message]) -> None:
        session_maker = await self.db.session()
        async with session_maker() as session:
            for message in messages:
                res = await self._process_message(message)
                if res:
                    await message.save(session)

    @staticmethod
    async def _check_stop_words(text: str) -> str:
        for word in STOP_WORDS:
            if re.search(word, text):
                logger.debug(f"Stop word '{word}' found in '{text}'")
                return word

    async def _process_message(self, message: Message) -> bool:
        if not message.text:
            logger.debug(f"Skipping message ID {message.id}. No text content found")
            return False

        if not message.channel:
            logger.debug(f"Skipping message ID {message.id}. No channel name found")
            return False

        message.text = re.sub(r'\s*\[.*?]\(https?://[^)]+\)$', '', message.text, flags=re.MULTILINE)
        if len(message.text) < MIN_LEN:
            logger.debug(f"Skipping message ID {message.id}. Text is too short.")
            return False

        stop_word = await self._check_stop_words(message.text)
        if stop_word:
            logger.debug(f"Skipping message ID {message.id}. Stop word '{stop_word}' found")
            return False

        message.score = await self.openai_service.get_evaluation(message.text)
        if message.score is None or message.score <= MIN_SCORE:
            logger.debug(f"Skipping message ID {message.id} with score {message.score}")
            return False

        logger.debug(f"Processing message ID {message.id}, channel: {message.channel}, text: {message.text[:50]}...")
        message.alt = await self.openai_service.get_alt(message.text)
        message.score_alt = await self.openai_service.get_evaluation(message.alt)
        if message.score_alt is None or message.score_alt <= MIN_SCORE_ALT:
            logger.debug(f"Skipping message ID {message.id} with score_alt {message.score_alt}")
            return False

        message.embedding = await self.embedding_service.generate_embedding(message.alt)
        message.similarity_score = 0.0

        if self.published_messages:
            message.similarity_score = await self.embedding_service.calculate_max_similarity(
                json.loads(message.embedding), self.published_messages
            )

        return True

    async def update_similarity(self):
        session_maker = await self.db.session()
        async with session_maker() as session:
            published_messages: Sequence[Message] = await Message.get_published_messages(session)
            unpublished_messages: Sequence[Message] = await Message.get_unpublished_messages(session)

            if not published_messages:
                logger.warning("No published messages found.")
                return

            if not unpublished_messages:
                logger.warning("No unpublished messages found.")
                return

            for message in unpublished_messages:
                if message.embedding:
                    message.similarity_score = await self.embedding_service.calculate_max_similarity(
                        json.loads(message.embedding), published_messages
                    )
                    await message.save(session)
