import json
from datetime import datetime
from unittest.mock import patch

import pytest

from models.message import Message
from services.embedding_service import EmbeddingService


@pytest.fixture
def embedding_service():
    return EmbeddingService()


@pytest.fixture
def messages():
    message1 = Message(
        id=1,
        channel="test_channel",
        timestamp=datetime.now(),
        embedding=json.dumps([0.1, 0.2, 0.3])
    )

    message2 = Message(
        id=2,
        channel="test_channel",
        timestamp=datetime.now(),
        embedding=json.dumps([0.4, 0.5, 0.6])
    )

    return [message1, message2]


@pytest.mark.asyncio
async def test_generate_embedding_with_valid_text(embedding_service):
    text = "Sample text for embedding"

    result = await embedding_service.generate_embedding(text)

    assert result is not None
    # Verify the result is a valid JSON string containing a list of floats
    embedded_list = json.loads(result)
    assert isinstance(embedded_list, list)
    assert all(isinstance(x, float) for x in embedded_list)


@pytest.mark.asyncio
async def test_generate_embedding_with_empty_text(embedding_service):
    text = ""

    result = await embedding_service.generate_embedding(text)

    assert result is None


@pytest.mark.asyncio
async def test_calculate_max_similarity_with_messages(embedding_service, messages):
    test_embedding = [0.1, 0.2, 0.3]

    result = await embedding_service.calculate_max_similarity(test_embedding, messages)

    assert result > 0.0
    assert isinstance(result, float)
    assert result <= 1.0


@pytest.mark.asyncio
async def test_calculate_max_similarity_with_empty_messages(embedding_service):
    test_embedding = [0.2, 0.3, 0.4]
    result = await embedding_service.calculate_max_similarity(test_embedding, [])

    assert result == 0.0


@pytest.mark.asyncio
async def test_generate_embedding_handles_value_error(embedding_service):
    with patch.object(embedding_service.model, 'encode') as encode:
        encode.side_effect = ValueError("Test error")

        result = await embedding_service.generate_embedding("test text")

        assert result is None


@pytest.mark.asyncio
async def test_generate_embedding_handles_general_exception(embedding_service):
    with patch.object(embedding_service.model, 'encode') as encode:
        encode.side_effect = Exception("Unexpected error")

        result = await embedding_service.generate_embedding("test text")

        assert result is None
