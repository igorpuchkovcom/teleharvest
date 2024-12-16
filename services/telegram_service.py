import logging
from typing import List, Optional

from telethon import TelegramClient
from telethon.tl.custom.message import Message as TelethonMessage

from models.message import Message
from services.interfaces import ITelegramService

logger = logging.getLogger(__name__)


class TelegramService(ITelegramService):
    def __init__(self, client: TelegramClient, channels: List[str], messages_limit: int = 10) -> None:
        self.client: TelegramClient = client
        self.channels: List[str] = channels
        self.messages_limit: int = messages_limit

    async def fetch_messages(self, channel: str, min_id: Optional[int] = None, max_id: Optional[int] = None) -> List[Message]:
        logger.debug(f"Fetching messages for channel: {channel} with min_id: {min_id} and max_id: {max_id}")
        try:
            messages = await self._get_messages(channel, min_id, max_id)
            if not len(messages):
                return []

            logger.info(f"Fetched {len(messages)} messages for channel: {channel}.")
            return self._create_message_objects(messages, channel)
        except Exception as e:
            logger.error(f"Error occurred while fetching messages for channel {channel}: {e}")
            return []

    async def _get_messages(self, channel: str, min_id: Optional[int] = None, max_id: Optional[int] = None):
        if max_id:
            return await self.client.get_messages(channel, min_id=min_id, max_id=max_id)

        if min_id:
            return await self.client.get_messages(channel, min_id=min_id, limit=None)

        return await self.client.get_messages(channel, limit=self.messages_limit)

    @staticmethod
    def _get_reactions(message: TelethonMessage) -> int:
        if not message.reactions:
            return 0

        return sum(reaction.count for reaction in message.reactions.results)

    @staticmethod
    def _create_message_objects(messages, channel: str) -> List[Message]:
        return [
            Message(
                id=msg.id,
                channel=channel,
                text=msg.text,
                timestamp=msg.date.strftime("%Y-%m-%d %H:%M:%S"),
                views=msg.views,
                reactions=TelegramService._get_reactions(msg),
                forwards=msg.forwards
            ) for msg in messages
        ]
