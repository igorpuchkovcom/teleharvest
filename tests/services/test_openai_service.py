from unittest.mock import Mock, patch, AsyncMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from services.openai_service import OpenAIService
from settings import Settings


@pytest.fixture
def config():
    settings = Settings()
    return settings.openai


@pytest.fixture
def openai_service(config):
    service = OpenAIService(
        config=config,
        prompt_process="Process: {text}",
        prompt_evaluate="Evaluate: {text}"
    )
    return service


@pytest.mark.asyncio
async def test_context_manager(openai_service):
    async with openai_service as service:
        assert service.session is not None
    assert service.session.closed


@pytest.mark.asyncio
async def test_make_request_success(openai_service):
    response = Mock(spec=ChatCompletion)
    message = Mock(spec=ChatCompletionMessage)
    message.content = "Test response"
    choice = Mock(spec=Choice)
    choice.message = message
    response.choices = [choice]

    with patch.object(openai_service.client.chat.completions, 'create', return_value=response):
        response = await openai_service.make_request("Test prompt")
        assert response == "Test response"


@pytest.mark.asyncio
async def test_make_request_failure(openai_service):
    with patch.object(openai_service.client.chat.completions, 'create', side_effect=Exception("API Error")):
        response = await openai_service.make_request("Test prompt")
        assert response is None


@pytest.mark.asyncio
async def test_get_evaluation_valid_response(openai_service):
    with patch.object(openai_service, 'make_request', return_value="0.85"):
        result = await openai_service.get_evaluation("Test text")
        assert result == 0.85


@pytest.mark.asyncio
async def test_get_evaluation_invalid_response(openai_service):
    with patch.object(openai_service, 'make_request', return_value="invalid"):
        result = await openai_service.get_evaluation("Test text")
        assert result is None


@pytest.mark.asyncio
async def test_get_evaluation_empty_text(openai_service):
    result = await openai_service.get_evaluation("")
    assert result is None


@pytest.mark.asyncio
async def test_get_alt_success(openai_service):
    with patch.object(openai_service, 'make_request', return_value="Processed text"):
        result = await openai_service.get_alt("Test text")
        assert result == "Processed text"


@pytest.mark.asyncio
async def test_get_alt_empty_text(openai_service):
    result = await openai_service.get_alt("")
    assert result is None
