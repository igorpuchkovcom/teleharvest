import json
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from models.message import Message
from processor import Processor


@pytest.fixture
def telegram_service():
    service = AsyncMock()
    service.channels = ["test_channel"]
    return service


@pytest.fixture
def openai_service():
    service = AsyncMock()
    service.get_evaluation.return_value = 86
    service.get_alt.return_value = "Alternative text"
    return service


@pytest.fixture
def db():
    session = AsyncMock()

    class AsyncSessionContextManager:
        async def __aenter__(self):
            return session

        async def __aexit__(self, exc_type, exc_value, traceback):
            pass

    def session_maker():
        return AsyncSessionContextManager()

    db = AsyncMock()
    db.session.return_value = session_maker
    return db


@pytest.fixture
def message():
    message = Message(
        id=1,
        text="Text message" * 30,
        channel="test_channel",
        timestamp=datetime(2024, 11, 27, 12),
        reactions=10,
        forwards=5,
        views=100,
    )

    return message


@pytest.fixture
def embedding_service():
    service = AsyncMock()
    service.generate_embedding.return_value = "embedding"
    service.calculate_max_similarity.return_value = 0.5
    return service


@pytest.fixture
def processor(telegram_service, openai_service, db, embedding_service):
    return Processor(
        telegram_service=telegram_service,
        openai_service=openai_service,
        db=db,
        embedding_service=embedding_service,
    )


@pytest.mark.asyncio
async def test_fetch_and_process(processor, telegram_service, db, message):
    # Mock database methods
    Message.get_published_messages = AsyncMock(return_value=[])
    Message.get_last_message_id = AsyncMock(return_value=0)
    Message.save = AsyncMock()

    # Mock Telegram service to return messages
    telegram_service.fetch_messages.return_value = [message]

    await processor.fetch_and_process()

    telegram_service.fetch_messages.assert_called_once_with("test_channel", 0)


@pytest.mark.asyncio
async def test_process_message_short_text(processor, message):
    message.text = "Short"
    await processor._process_message(message)

    assert not message.score  # Message should not be processed


@pytest.mark.asyncio
async def test_process_message_valid(processor, openai_service, embedding_service, message):
    openai_service.get_evaluation.side_effect = [86, 96]
    embedding_service.generate_embedding.return_value = "embedding"

    await processor._process_message(message)

    assert message.score == 86
    assert message.score_alt == 96
    assert message.embedding == "embedding"
    openai_service.get_alt.assert_called_once()
    embedding_service.generate_embedding.assert_called_once()


@pytest.mark.asyncio
async def test_process_message_no_text(processor, openai_service, embedding_service, message):
    message.text = None
    openai_service.get_evaluation.side_effect = [None, None]
    embedding_service.generate_embedding.return_value = "embedding"

    await processor._process_message(message)

    assert message.score is None
    assert message.score_alt is None
    assert message.embedding is None
    openai_service.get_alt.assert_not_called()
    embedding_service.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_low_score(processor, openai_service, embedding_service, message):
    openai_service.get_evaluation.side_effect = [50, 96]
    embedding_service.generate_embedding.return_value = "embedding"

    await processor._process_message(message)

    assert message.score == 50
    assert message.score_alt is None
    assert message.embedding is None
    openai_service.get_alt.assert_not_called()
    embedding_service.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_low_alt_score(processor, openai_service, embedding_service, message):
    openai_service.get_evaluation.side_effect = [86, 50]
    embedding_service.generate_embedding.return_value = "embedding"

    await processor._process_message(message)

    assert message.score == 86
    assert message.score_alt == 50
    assert message.embedding is None
    openai_service.get_alt.assert_called_once()
    embedding_service.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_low_er(processor, openai_service, embedding_service, message):
    message.reactions = 0
    message.forwards = 0
    message.views = 100
    openai_service.get_evaluation.side_effect = [None, None]
    embedding_service.generate_embedding.return_value = "embedding"

    await processor._process_message(message)

    assert message.score is None
    assert message.score_alt is None
    assert message.embedding is None
    openai_service.get_alt.assert_not_called()
    embedding_service.generate_embedding.assert_not_called()

@pytest.mark.asyncio
async def test_process_message_no_channel(processor, openai_service, embedding_service, message):
    openai_service.get_evaluation.side_effect = [None, None]
    embedding_service.generate_embedding.return_value = "embedding"
    message.channel = ""

    await processor._process_message(message)

    assert message.score is None
    assert message.score_alt is None
    assert message.embedding is None
    openai_service.get_alt.assert_not_called()
    embedding_service.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_stop_words(processor, openai_service, embedding_service, message):
    openai_service.get_evaluation.side_effect = [None, None]
    embedding_service.generate_embedding.return_value = "embedding"
    message.text = "This is an астролог message" * 20

    await processor._process_message(message)

    assert message.score is None
    assert message.score_alt is None
    assert message.embedding is None
    openai_service.get_alt.assert_not_called()
    embedding_service.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_update_similarity(processor, db, embedding_service):
    message = Message(
        id=1,
        text="Text message" * 30,
        channel="test_channel",
        embedding=json.dumps({"vector": [0.1, 0.2, 0.3]}),
        timestamp=datetime(2024, 11, 27, 12)
    )

    published = [message]
    unpublished = [message]

    Message.get_published_messages = AsyncMock(return_value=published)
    Message.get_unpublished_messages = AsyncMock(return_value=unpublished)
    Message.save = AsyncMock()

    await processor.update_similarity()

    embedding_service.calculate_max_similarity.assert_called_once_with(
        json.loads(unpublished[0].embedding), published
    )
    Message.save.assert_called_once()


@pytest.mark.asyncio
async def test_process_message_similarity_score(processor, openai_service, embedding_service, message):
    openai_service.get_evaluation.side_effect = [86, 96]

    processor.published_messages = [message]

    embedding_service.generate_embedding.return_value = json.dumps({"vector": [0.4, 0.5, 0.6]})
    embedding_service.calculate_max_similarity.return_value = 0.8

    await processor._process_message(message)

    assert message.similarity_score == 0.8
    embedding_service.calculate_max_similarity.assert_called_once_with(
        json.loads(message.embedding), processor.published_messages
    )


@pytest.mark.asyncio
async def test_update_similarity_no_published(processor, db, embedding_service, message):
    Message.get_published_messages = AsyncMock(return_value=[])
    Message.get_unpublished_messages = AsyncMock(return_value=[message])

    await processor.update_similarity()

    embedding_service.calculate_max_similarity.assert_not_called()


@pytest.mark.asyncio
async def test_update_similarity_no_unpublished(processor, db, embedding_service, message):
    Message.get_published_messages = AsyncMock(return_value=[message])
    Message.get_unpublished_messages = AsyncMock(return_value=[])

    await processor.update_similarity()

    embedding_service.calculate_max_similarity.assert_not_called()


@pytest.mark.asyncio
async def test_processor_context_manager(
        openai_service, db, embedding_service, telegram_service
):
    processor = Processor(
        openai_service=openai_service,
        db=db,
        embedding_service=embedding_service,
        telegram_service=telegram_service
    )

    openai_service.__aenter__.return_value = openai_service
    openai_service.__aexit__.return_value = None
    db.__aenter__.return_value = db
    db.__aexit__.return_value = None

    async with processor as p:
        assert p is processor

    openai_service.__aenter__.assert_called_once()
    openai_service.__aexit__.assert_called_once_with(None, None, None)
    db.__aenter__.assert_called_once()
    db.__aexit__.assert_called_once_with(None, None, None)


@pytest.mark.asyncio
async def test_fetch_and_update_metrics(processor, telegram_service, db, message):
    # Mock database methods
    Message.get_first_message_id = AsyncMock(return_value=0)
    Message.save = AsyncMock()

    # Mock Telegram service to return messages
    telegram_service.fetch_messages.return_value = [message]

    await processor.fetch_and_update_metrics()

    telegram_service.fetch_messages.assert_called_once_with("test_channel", 0, 0)



@pytest.mark.asyncio
async def test_update_metrics_no_views_or_reactions(processor, message):
    message.views = None
    message.reactions = None

    result = await processor._update_metrics(message)
    assert not result

    message.views = 10
    result = await processor._update_metrics(message)
    assert not result

    message.reactions = 5
    result = await processor._update_metrics(message)
    assert result
