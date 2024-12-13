import json
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from models.message import Message
from processor import Processor
from settings import Settings


@pytest.fixture
def config():
    settings = Settings()
    return settings.processor


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
def processor(telegram_service, openai_service, db, embedding_service, config):
    return Processor(
        telegram_service=telegram_service,
        openai_service=openai_service,
        db=db,
        embedding_service=embedding_service,
        config=config
    )


@pytest.mark.asyncio
async def test_fetch_and_process(processor, telegram_service, message):
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
    processor.published_messages = [message]

    openai_service.get_evaluation.side_effect = [86, 96, 96]
    embedding_service.generate_embedding.return_value = json.dumps([0.1, 0.2, 0.3])
    embedding_service.calculate_max_similarity.return_value = 0.8

    result = await processor._process_message(message)

    assert result
    assert message.score == 86
    assert message.alt == "Alternative text"
    assert message.score_alt == 96
    assert message.embedding == json.dumps([0.1, 0.2, 0.3])
    assert message.score_improve == 96
    assert message.similarity_score == 0.8


async def edge_cases(processor, openai_service, embedding_service, message):
    openai_service.get_evaluation.side_effect = [None, None]
    embedding_service.generate_embedding.return_value = "embedding"

    await processor._process_message(message)

    assert message.score is None
    assert message.score_alt is None
    assert message.embedding is None
    openai_service.get_alt.assert_not_called()
    embedding_service.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_no_text(processor, openai_service, embedding_service, message):
    message.text = None
    await edge_cases(processor, openai_service, embedding_service, message)


@pytest.mark.asyncio
async def test_process_message_low_score(processor, openai_service, embedding_service, message):
    openai_service.get_evaluation.side_effect = [50, 96]
    embedding_service.generate_embedding.return_value = "embedding"

    result = await processor._process_message(message)

    assert not result
    assert message.score == 50
    assert message.score_alt is None
    assert message.embedding is None
    openai_service.get_alt.assert_not_called()
    embedding_service.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_low_alt_score(processor, openai_service, embedding_service, message):
    openai_service.get_evaluation.side_effect = [86, 50]
    embedding_service.generate_embedding.return_value = "embedding"

    result = await processor._process_message(message)

    assert not result
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
    await edge_cases(processor, openai_service, embedding_service, message)


@pytest.mark.asyncio
async def test_process_message_no_channel(processor, openai_service, embedding_service, message):
    message.channel = ""
    await edge_cases(processor, openai_service, embedding_service, message)


@pytest.mark.asyncio
async def test_process_message_stop_words(processor, openai_service, embedding_service, message):
    message.text = "This is an астролог message" * 20
    await edge_cases(processor, openai_service, embedding_service, message)


@pytest.mark.asyncio
async def test_update_similarity(processor, embedding_service, message):
    message.embedding = json.dumps({"vector": [0.1, 0.2, 0.3]})
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
async def test_process_message_low_er_and_views(processor, message):
    message.views = 200
    message.reactions = 1
    message.forwards = 2
    processor.config.min_er = 0.05
    processor.config.min_views = 150

    result = await processor._process_message(message)

    assert not result
    assert not message.score


@pytest.mark.asyncio
async def test_process_message_similarity_score(processor, openai_service, embedding_service, message):
    openai_service.get_evaluation.side_effect = [86, 96, 96]

    processor.published_messages = [message]

    embedding_service.generate_embedding.return_value = json.dumps({"vector": [0.4, 0.5, 0.6]})
    embedding_service.calculate_max_similarity.return_value = 0.8

    await processor._process_message(message)

    assert message.similarity_score == 0.8
    embedding_service.calculate_max_similarity.assert_called_once_with(
        json.loads(message.embedding), processor.published_messages
    )


@pytest.mark.asyncio
async def test_update_similarity_no_published(processor, embedding_service, message):
    Message.get_published_messages = AsyncMock(return_value=[])
    Message.get_unpublished_messages = AsyncMock(return_value=[message])

    await processor.update_similarity()

    embedding_service.calculate_max_similarity.assert_not_called()


@pytest.mark.asyncio
async def test_update_similarity_no_unpublished(processor, embedding_service, message):
    Message.get_published_messages = AsyncMock(return_value=[message])
    Message.get_unpublished_messages = AsyncMock(return_value=[])

    await processor.update_similarity()

    embedding_service.calculate_max_similarity.assert_not_called()


@pytest.mark.asyncio
async def test_processor_context_manager(processor, openai_service, db):
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
async def test_fetch_and_update_metrics(processor, telegram_service, message):
    Message.get_first_message_id = AsyncMock(return_value=0)
    Message.save = AsyncMock()

    telegram_service.fetch_messages.return_value = [message]
    processor.update_metrics = AsyncMock()

    await processor.fetch_and_update_metrics()

    telegram_service.fetch_messages.assert_called_once_with("test_channel", 0, processor.channel_min_id["test_channel"])
    processor.update_metrics.assert_called_once_with([message])


@pytest.mark.asyncio
async def test_fetch_and_update_metrics_no_messages(processor, telegram_service):
    Message.get_first_message_id = AsyncMock(return_value=0)
    telegram_service.fetch_messages.return_value = []
    processor.update_metrics = AsyncMock()

    await processor.fetch_and_update_metrics()

    telegram_service.fetch_messages.assert_called_once()


@pytest.mark.asyncio
async def test_update_metrics_no_views_or_reactions(processor, message):
    message.views = None
    message.reactions = None

    result = await processor._update_metrics(message)
    assert not result  # Should return False since no views or reactions are present

    message.views = 10
    result = await processor._update_metrics(message)
    assert not result  # Should return False since no reactions are present

    message.reactions = 5
    result = await processor._update_metrics(message)
    assert result  # Should return True since views and reactions are present


@pytest.mark.asyncio
async def test_update_metrics_invalid_data(processor, message):
    # Test with missing views
    message.views = None
    message.reactions = 10
    result = await processor._update_metrics(message)
    assert not result  # Should return False because of missing views

    # Test with missing reactions
    message.views = 100
    message.reactions = None
    result = await processor._update_metrics(message)
    assert not result  # Should return False because of missing reactions


@pytest.mark.asyncio
async def test_update_metrics_valid(processor, message):
    # Arrange
    message.views = 100
    message.reactions = 10
    message.forwards = 5
    processor._update_metrics = AsyncMock(return_value=True)
    message.update = AsyncMock()

    # Act
    await processor.update_metrics([message])

    # Assert
    processor._update_metrics.assert_called_once_with(message)

    args, kwargs = message.update.call_args

    assert kwargs['views'] == 100
    assert kwargs['reactions'] == 10
    assert kwargs['forwards'] == 5


@pytest.mark.asyncio
async def test_update_metrics_missing_views(processor, message):
    # Arrange
    message.views = None
    message.reactions = 10
    processor._update_metrics = AsyncMock(return_value=False)

    # Act
    result = await processor.update_metrics([message])

    # Assert
    processor._update_metrics.assert_called_once_with(message)
    assert result is None  # No update should occur because of missing views


@pytest.mark.asyncio
async def test_update_metrics_missing_reactions(processor, message):
    # Arrange
    message.views = 100
    message.reactions = None
    processor._update_metrics = AsyncMock(return_value=False)

    # Act
    result = await processor.update_metrics([message])

    # Assert
    processor._update_metrics.assert_called_once_with(message)
    assert result is None  # No update should occur because of missing reactions


@pytest.mark.asyncio
async def test_update_metrics_no_valid_data(processor, message):
    # Arrange
    message.views = None
    message.reactions = None
    processor._update_metrics = AsyncMock(return_value=False)
    message.update = AsyncMock()

    # Act
    await processor.update_metrics([message])

    # Assert
    processor._update_metrics.assert_called_once_with(message)
    message.update.assert_not_called()  # Update should not be called due to invalid data


@pytest.mark.asyncio
async def test_process_message_low_score_improve(processor, openai_service, embedding_service, message):
    openai_service.get_evaluation.side_effect = [86, 96, 50]
    embedding_service.generate_embedding.return_value = json.dumps([0.1, 0.2, 0.3])
    embedding_service.calculate_max_similarity.return_value = 0.8

    result = await processor._process_message(message)

    assert not result
    assert message.score == 86
    assert message.score_alt == 96
    assert message.score_improve == 50
    embedding_service.generate_embedding.assert_called_once()
    openai_service.get_alt.assert_called_once()
    openai_service.get_improve.assert_called_once()
