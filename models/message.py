import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Union, Sequence

from sqlalchemy import Column, Integer, String, DateTime, Float, select, desc, asc, literal, PrimaryKeyConstraint
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()

DAYS_IN_LAST_MONTH = 30


class Message(Base):
    __tablename__ = 'post'

    id = Column(Integer, primary_key=True)
    channel = Column(String(255), primary_key=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    text = Column(String, nullable=True)
    score = Column(Integer, nullable=True)
    alt = Column(String, nullable=True)
    score_alt = Column(Integer, nullable=True)
    embedding = Column(String, nullable=True)
    similarity_score = Column(Float, nullable=True)
    published = Column(DateTime, nullable=True)
    views = Column(Integer, nullable=True)
    reactions = Column(Integer, nullable=True)
    forwards = Column(Integer, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint('id', 'channel'),
    )

    def __init__(self,
                 id: int,
                 channel: str,
                 timestamp: datetime,
                 text: Optional[str] = None,
                 score: Optional[int] = None,
                 alt: Optional[str] = None,
                 score_alt: Optional[int] = None,
                 embedding: Optional[Union[str, List[float]]] = None,
                 similarity_score: Optional[float] = None,
                 published: datetime = None,
                 views: Optional[int] = None,
                 reactions: Optional[int] = None,
                 forwards: Optional[int] = None
                 ):
        super().__init__(
            id=id,
            channel=channel,
            timestamp=timestamp,
            text=text,
            score=score,
            alt=alt,
            score_alt=score_alt,
            embedding=json.dumps(embedding) if isinstance(embedding, list) else embedding,
            similarity_score=similarity_score,
            published=published,
            views=views,
            reactions=reactions,
            forwards=forwards
        )

    def __repr__(self):
        return (
            f"<Message(id={self.id}, channel='{self.channel}', timestamp={self.timestamp},"
            f""f"text='{self.text[:20]}...', score={self.score})>"
        )

    @staticmethod
    async def get_last_message_id(session: AsyncSession, channel: str) -> Optional[int]:
        try:
            result = await session.execute(
                select(Message.id)
                .where(Message.channel == literal(channel))
                .order_by(desc(Message.id))
                .limit(1)
            )
            id = result.scalar_one_or_none()
            logger.debug(f"Last message ID for channel '{channel}': {id}")
            return id
        except Exception as e:
            logger.error(f"Error getting last message ID for channel '{channel}': {e}")
            raise

    @staticmethod
    async def get_first_message_id(session: AsyncSession, channel: str, limit: int = 1000) -> Optional[int]:
        try:
            result = await session.execute(
                select(Message.id)
                .where(Message.channel == literal(channel))
                .order_by(asc(Message.id))
                .limit(limit)
            )
            ids = result.scalars().all()
            if ids:
                first_id = min(ids)
                logger.debug(f"First message ID for channel '{channel}' with limit {limit}: {first_id}")
                return first_id
            else:
                logger.debug(f"No messages found for channel '{channel}'")
                return None
        except Exception as e:
            logger.error(f"Error getting first message ID for channel '{channel}': {e}")
            raise

    @staticmethod
    async def get_published_messages(session: AsyncSession) -> Sequence['Message']:
        last_month = datetime.now() - timedelta(days=DAYS_IN_LAST_MONTH)
        query = select(Message).where(
            Message.timestamp > last_month,
            Message.published.isnot(None)
        )
        try:
            result = await session.execute(query)
            messages = result.scalars().all()
            logger.debug(f"Retrieved {len(messages)} messages from last month")
            return messages
        except Exception as e:
            logger.error(f"Error getting messages from last month: {e}")
            raise

    @staticmethod
    async def get_unpublished_messages(session: AsyncSession) -> Sequence['Message']:
        query = select(Message).where(
            Message.embedding.isnot(None),
            Message.published.is_(None)
        )
        try:
            result = await session.execute(query)
            messages = result.scalars().all()
            logger.debug(f"Retrieved {len(messages)} messages from last month")
            return messages
        except Exception as e:
            logger.error(f"Error getting messages from last month: {e}")
            raise

    @staticmethod
    async def get_message(session: AsyncSession, id: int, channel: str) -> Optional['Message']:
        query = select(Message).where(
            Message.id == literal(id),
            Message.channel == literal(channel)
        )
        try:
            result = await session.execute(query)
            message = result.scalar_one_or_none()
            if message:
                logger.debug(f"Retrieved message with id={id} and channel='{channel}': {message}")
            else:
                logger.debug(f"No message found with id={id} and channel='{channel}'")
            return message
        except Exception as e:
            logger.error(f"Error retrieving message with id={id} and channel='{channel}': {e}")
            raise

    async def save(self, session: AsyncSession) -> None:
        try:
            session.add(self)
            await session.commit()
            logger.debug(f"Message {self.id} channel {self.channel} saved successfully.")
        except Exception as e:
            logger.error(f"Error saving message {self.id} channel channel {self.channel}: {e}")
            await session.rollback()

    async def update(self, session: AsyncSession, **kwargs) -> None:
        try:
            message = await self.get_message(session, self.id, self.channel)
            if not message:
                logger.debug(f"Message with id={self.id} and channel={self.channel} not found. Update skipped.")
                return

            for key, value in kwargs.items():
                if hasattr(message, key):
                    setattr(message, key, value)
                else:
                    logger.debug(f"Field '{key}' does not exist on Message. Ignored.")

            session.add(message)
            await session.commit()
            logger.debug(f"Message {self.id} channel {self.channel} updated successfully.")
        except Exception as e:
            logger.error(f"Error updating message {self.id} channel {self.channel}: {e}")
            await session.rollback()
