import json
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from models.message import Message
from processor import Processor


@pytest.fixture
def telegram_service_mock():
    service = AsyncMock()
    service.channels = ["test_channel"]
    return service


@pytest.fixture
def openai_service_mock():
    service = AsyncMock()
    service.get_evaluation.return_value = 86
    service.get_alt.return_value = "Alternative text"
    return service


@pytest.fixture
def db_mock():
    session_mock = AsyncMock()

    class AsyncSessionContextManager:
        async def __aenter__(self):
            return session_mock

        async def __aexit__(self, exc_type, exc_value, traceback):
            pass

    def session_maker():
        return AsyncSessionContextManager()

    db = AsyncMock()
    db.session.return_value = session_maker
    return db


@pytest.fixture
def embedding_service_mock():
    service = AsyncMock()
    service.generate_embedding.return_value = "mock_embedding"
    service.calculate_max_similarity.return_value = 0.5
    return service


@pytest.fixture
def processor(telegram_service_mock, openai_service_mock, db_mock, embedding_service_mock):
    return Processor(
        telegram_service=telegram_service_mock,
        openai_service=openai_service_mock,
        db=db_mock,
        embedding_service=embedding_service_mock,
    )


@pytest.mark.asyncio
async def test_fetch_and_process(processor, telegram_service_mock, db_mock):
    # Mock database methods
    Message.get_published_messages = AsyncMock(return_value=[])
    Message.get_last_message_id = AsyncMock(return_value=0)
    Message.save = AsyncMock()

    # Mock Telegram service to return messages
    telegram_service_mock.fetch_messages.return_value = [
        Message(
            id=1,
            text="Test message",
            channel="test_channel",
            timestamp=datetime(2024, 11, 27, 12)
        )
    ]

    await processor.fetch_and_process()

    telegram_service_mock.fetch_messages.assert_called_once_with("test_channel", 0)


@pytest.mark.asyncio
async def test_process_message_short_text(processor):
    message = Message(
        id=1,
        text="Short",
        channel="test_channel",
        timestamp=datetime(2024, 11, 27, 12)
    )
    await processor._process_message(message)

    assert not message.score  # Message should not be processed


@pytest.mark.asyncio
async def test_process_message_with_stop_words(processor):
    message = Message(
        id=1,
        text="This is an астролог message",
        channel="test_channel",
        timestamp=datetime(2024, 11, 27, 12)
    )
    await processor._process_message(message)

    assert not message.score  # Message should not be processed


@pytest.mark.asyncio
async def test_process_message_valid(processor, openai_service_mock, embedding_service_mock):
    openai_service_mock.get_evaluation.side_effect = [86, 96]
    embedding_service_mock.generate_embedding.return_value = "mock_embedding"

    message = Message(
        id=1,
        text="Valid text message" * 20,
        channel="test_channel",
        timestamp=datetime(2024, 11, 27, 12)
    )
    await processor._process_message(message)

    assert message.score == 86
    assert message.score_alt == 96
    assert message.embedding == "mock_embedding"
    openai_service_mock.get_alt.assert_called_once()
    embedding_service_mock.generate_embedding.assert_called_once()


@pytest.mark.asyncio
async def test_process_message_no_text(processor, openai_service_mock, embedding_service_mock):
    openai_service_mock.get_evaluation.side_effect = [None, None]
    embedding_service_mock.generate_embedding.return_value = "mock_embedding"

    message = Message(id=1, channel="test_channel", timestamp=datetime(2024, 11, 27, 12))
    await processor._process_message(message)

    assert message.score is None
    assert message.score_alt is None
    assert message.embedding is None
    openai_service_mock.get_alt.assert_not_called()
    embedding_service_mock.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_low_score(processor, openai_service_mock, embedding_service_mock):
    openai_service_mock.get_evaluation.side_effect = [50, 96]
    embedding_service_mock.generate_embedding.return_value = "mock_embedding"

    message = Message(
        id=1,
        text="Valid text message" * 20,
        channel="test_channel",
        timestamp=datetime(2024, 11, 27, 12)
    )
    await processor._process_message(message)

    assert message.score == 50
    assert message.score_alt is None
    assert message.embedding is None
    openai_service_mock.get_alt.assert_not_called()
    embedding_service_mock.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_low_alt_score(processor, openai_service_mock, embedding_service_mock):
    openai_service_mock.get_evaluation.side_effect = [86, 50]
    embedding_service_mock.generate_embedding.return_value = "mock_embedding"

    message = Message(
        id=1,
        text="Valid text message" * 20,
        channel="test_channel",
        timestamp=datetime(2024, 11, 27, 12)
    )
    await processor._process_message(message)

    assert message.score == 86
    assert message.score_alt == 50
    assert message.embedding is None
    openai_service_mock.get_alt.assert_called_once()
    embedding_service_mock.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_no_channel(processor, openai_service_mock, embedding_service_mock):
    openai_service_mock.get_evaluation.side_effect = [None, None]
    embedding_service_mock.generate_embedding.return_value = "mock_embedding"

    message = Message(
        id=1,
        text="Valid text message" * 20,
        channel="",
        timestamp=datetime(2024, 11, 27, 12)
    )

    await processor._process_message(message)

    assert message.score is None
    assert message.score_alt is None
    assert message.embedding is None
    openai_service_mock.get_alt.assert_not_called()
    embedding_service_mock.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_stop_words(processor, openai_service_mock, embedding_service_mock):
    openai_service_mock.get_evaluation.side_effect = [None, None]
    embedding_service_mock.generate_embedding.return_value = "mock_embedding"

    message = Message(
        id=1,
        text="эфир запись астролог зодиак таро эзотери" * 20,
        channel="test_channel",
        timestamp=datetime(2024, 11, 27, 12)
    )

    await processor._process_message(message)

    assert message.score is None
    assert message.score_alt is None
    assert message.embedding is None
    openai_service_mock.get_alt.assert_not_called()
    embedding_service_mock.generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_update_similarity(processor, db_mock, embedding_service_mock):
    published_mock = [
        Message(
            id=1,
            text="Published message",
            embedding=json.dumps({"vector": [0.1, 0.2, 0.3]}),
            timestamp=datetime(2024, 11, 27, 12),
            channel="test_channel"
        )
    ]
    unpublished_mock = [
        Message(
            id=2,
            text="Unpublished message",
            embedding=json.dumps({"vector": [0.4, 0.5, 0.6]}),
            timestamp=datetime(2024, 11, 27, 12),
            channel="test_channel"
        )
    ]

    Message.get_published_messages = AsyncMock(return_value=published_mock)
    Message.get_unpublished_messages = AsyncMock(return_value=unpublished_mock)
    Message.save = AsyncMock()

    await processor.update_similarity()

    embedding_service_mock.calculate_max_similarity.assert_called_once_with(
        json.loads(unpublished_mock[0].embedding), published_mock
    )
    Message.save.assert_called_once()


@pytest.mark.asyncio
async def test_process_message_similarity_score(processor, openai_service_mock, embedding_service_mock):
    openai_service_mock.get_evaluation.side_effect = [86, 96]

    processor.published_messages = [
        Message(
            id=1,
            text="Published message" * 20,
            embedding=json.dumps({"vector": [0.1, 0.2, 0.3]}),
            timestamp=datetime(2024, 11, 27, 12),
            channel="test_channel"
        )
    ]

    embedding_service_mock.generate_embedding.return_value = json.dumps({"vector": [0.4, 0.5, 0.6]})
    embedding_service_mock.calculate_max_similarity.return_value = 0.8

    message = Message(id=2, text="This is a valid message" * 20, channel="test_channel",
                      timestamp=datetime(2024, 11, 27, 12))
    await processor._process_message(message)

    assert message.similarity_score == 0.8
    embedding_service_mock.calculate_max_similarity.assert_called_once_with(
        json.loads(message.embedding), processor.published_messages
    )


@pytest.mark.asyncio
async def test_update_similarity_no_published(processor, db_mock, embedding_service_mock):
    Message.get_published_messages = AsyncMock(return_value=[])
    Message.get_unpublished_messages = AsyncMock(return_value=[
        Message(
            id=2,
            text="Unpublished message",
            embedding=json.dumps({"vector": [0.4, 0.5, 0.6]}),
            timestamp=datetime(2024, 11, 27, 12),
            channel="test_channel"
        )
    ])

    await processor.update_similarity()

    embedding_service_mock.calculate_max_similarity.assert_not_called()


@pytest.mark.asyncio
async def test_update_similarity_no_unpublished(processor, db_mock, embedding_service_mock):
    Message.get_published_messages = AsyncMock(return_value=[
        Message(
            id=1,
            text="Published message",
            embedding=json.dumps({"vector": [0.1, 0.2, 0.3]}),
            timestamp=datetime(2024, 11, 27, 12),
            channel="test_channel"
        )
    ])
    Message.get_unpublished_messages = AsyncMock(return_value=[])

    await processor.update_similarity()

    embedding_service_mock.calculate_max_similarity.assert_not_called()


@pytest.mark.asyncio
async def test_processor_context_manager(
        openai_service_mock, db_mock, embedding_service_mock, telegram_service_mock
):
    processor = Processor(
        openai_service=openai_service_mock,
        db=db_mock,
        embedding_service=embedding_service_mock,
        telegram_service=telegram_service_mock
    )

    openai_service_mock.__aenter__.return_value = openai_service_mock
    openai_service_mock.__aexit__.return_value = None
    db_mock.__aenter__.return_value = db_mock
    db_mock.__aexit__.return_value = None

    async with processor as p:
        assert p is processor

    openai_service_mock.__aenter__.assert_called_once()
    openai_service_mock.__aexit__.assert_called_once_with(None, None, None)
    db_mock.__aenter__.assert_called_once()
    db_mock.__aexit__.assert_called_once_with(None, None, None)
