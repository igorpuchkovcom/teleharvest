import logging
from typing import List, Optional

from telethon import TelegramClient

from models.message import Message
from services.interfaces import ITelegramService

logger = logging.getLogger(__name__)


class TelegramService(ITelegramService):
    def __init__(self, client: TelegramClient, channels: List[str], messages_limit: int = 1):
        self.client: TelegramClient = client
        self.channels: List[str] = channels
        self.messages_limit: int = messages_limit

    async def fetch_messages(self, channel: str, last_message_id: Optional[int] = None) -> List[Message]:
        logger.debug(f"Fetching new messages for channel: {channel} with last_message_id: {last_message_id}")
        try:
            messages = await self._get_messages(channel, last_message_id)
            if not len(messages):
                return []

            logger.info(f"Fetched {len(messages)} new messages for channel: {channel}.")
            return self._create_message_objects(messages, channel)
        except Exception as e:
            logger.error(f"Error occurred while fetching new messages for channel {channel}: {e}")
            return []

    async def _get_messages(self, channel: str, last_message_id: Optional[int] = None):
        if last_message_id:
            return await self.client.get_messages(channel, min_id=last_message_id, limit=None)

        return await self.client.get_messages(channel, limit=self.messages_limit)

    @staticmethod
    def _create_message_objects(messages, channel: str) -> List[Message]:
        return [
            Message(
                id=msg.id,
                channel=channel,
                text=msg.text,
                timestamp=msg.date.strftime("%Y-%m-%d %H:%M:%S")
            ) for msg in messages
        ]
