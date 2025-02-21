import asyncio
import json
import logging
import re
from typing import List, Optional, Type, Sequence

from models.message import Message
from services.interfaces import ITelegramService, IOpenAIService, IAsyncDatabase, IEmbeddingService
from settings import ProcessorSettings

logger = logging.getLogger(__name__)


class Processor:
    published_messages: Sequence[Message]
    channel_min_id = {}

    def __init__(
            self,
            telegram_service: ITelegramService,
            openai_service: IOpenAIService,
            db: IAsyncDatabase,
            embedding_service: IEmbeddingService,
            config: ProcessorSettings
    ) -> None:
        self.telegram_service = telegram_service
        self.openai_service = openai_service
        self.db = db
        self.embedding_service = embedding_service
        self.config = config
        self.published_messages = []
        self.credits_available = False

    async def async_init(self):
        self.credits_available = await self.openai_service.check_credits_available()

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
            exc_tb: Optional[Type[BaseException]]
    ) -> None:
        await asyncio.gather(
            self.openai_service.__aexit__(exc_type, exc_val, exc_tb),
            self.db.__aexit__(exc_type, exc_val, exc_tb)
        )

    async def fetch_and_process(self) -> None:
        session_maker = await self.db.session()
        async with session_maker() as session:
            self.published_messages = await Message.get_published_messages(session)
            for channel in self.telegram_service.channels:
                min_id = await Message.get_last_message_id(session, channel)
                self.channel_min_id[channel] = min_id
                messages = await self.telegram_service.fetch_messages(channel, min_id)
                await self.process(messages)

    async def fetch_and_update_metrics(self) -> None:
        session_maker = await self.db.session()
        async with session_maker() as session:
            for channel in self.telegram_service.channels:
                min_id = await Message.get_first_message_id(session, channel, self.config.limit)
                messages = await self.telegram_service.fetch_messages(channel, min_id, self.channel_min_id[channel])
                await self.update_metrics(messages)

    async def process(self, messages: List[Message]) -> None:
        session_maker = await self.db.session()
        async with session_maker() as session:
            for index, message in enumerate(messages):
                is_last_message = (index == len(messages) - 1)
                await self._process_message(message, is_last_message)
                await message.save(session)

    async def update_metrics(self, messages: List[Message]) -> None:
        session_maker = await self.db.session()
        async with session_maker() as session:
            for message in messages:
                res = await self._update_metrics(message)
                if res:
                    await message.update(session, views=message.views, reactions=message.reactions,
                                         forwards=message.forwards)

    async def _check_stop_words(self, text: str) -> Optional[str]:
        for word in self.config.stop_words_list:
            if re.search(word, text):
                logger.debug(f"Stop word '{word}' found in '{text}'")
                return word
        return None

    async def _process_message(self, message: Message, last_message: bool = False) -> bool:
        if not message.text:
            logger.debug(f"Skipping message ID {message.id}. No text content found")
            return False

        if not message.channel:
            logger.debug(f"Skipping message ID {message.id}. No channel name found")
            return False

        message.text = re.sub(r'\s*\[.*?]\(https?://[^)]+\)$', '', message.text, flags=re.MULTILINE)
        if len(message.text) < self.config.min_len:
            logger.debug(f"Skipping message ID {message.id}. Text is too short.")
            return False

        stop_word = await self._check_stop_words(message.text)
        if stop_word:
            logger.debug(f"Skipping message ID {message.id}. Stop word '{stop_word}' found")
            return False

        er = (message.reactions + message.forwards) / message.views if message.views else 0
        if (er < self.config.min_er) and (message.views > self.config.min_views) and not last_message:
            logger.debug(f"Skipping message ID {message.id} with ER {er}")
            return False

        if not self.credits_available:
            return True

        message.score = await self.openai_service.get_evaluation(message.text)
        if message.score is None or message.score <= self.config.min_score:
            logger.debug(f"Skipping message ID {message.id} with score {message.score}")
            return False

        logger.debug(
            f"Processing message ID {message.id}, channel: {message.channel}, text: {message.text[:50]}...")
        message.alt = await self.openai_service.get_alt(message.text)
        message.score_alt = await self.openai_service.get_evaluation(message.alt)
        if message.score_alt is None or message.score_alt <= self.config.min_score_alt:
            logger.debug(f"Skipping message ID {message.id} with score_alt {message.score_alt}")
            return False

        message.embedding = await self.embedding_service.generate_embedding(message.alt)
        message.similarity_score = 0.0

        if self.published_messages:
            message.similarity_score = await self.embedding_service.calculate_max_similarity(
                json.loads(message.embedding), self.published_messages
            )

        return True

    @staticmethod
    async def _update_metrics(message: Message) -> bool:
        if not message.views:
            logger.debug(f"Skipping message ID {message.id}. No views value found")
            return False

        if not message.reactions:
            logger.debug(f"Skipping message ID {message.id}. No reactions value found")
            return False

        return True

    async def update_similarity(self) -> None:
        session_maker = await self.db.session()
        async with session_maker() as session:
            published_messages = await Message.get_published_messages(session)
            unpublished_messages = await Message.get_unpublished_messages(session)

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
