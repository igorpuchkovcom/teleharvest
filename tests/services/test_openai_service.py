from unittest.mock import Mock, patch

import httpx
import pytest
from openai import OpenAIError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from services.openai_service import OpenAIService
from settings import Settings


@pytest.fixture
def config() -> Settings:
    settings = Settings()
    return settings.openai


@pytest.fixture
def openai_service(config: Settings) -> OpenAIService:
    service = OpenAIService(
        config=config,
        prompt_process="Process: {text}",
        prompt_evaluate="Evaluate: {text}"
    )
    return service


@pytest.mark.asyncio
async def test_context_manager(openai_service: OpenAIService) -> None:
    async with openai_service as service:
        assert service.session is not None
    assert service.session.closed


@pytest.mark.asyncio
async def test_make_request_success(openai_service: OpenAIService) -> None:
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
async def test_make_request_failure(openai_service: OpenAIService) -> None:
    with patch.object(openai_service.client.chat.completions, 'create', side_effect=Exception("API Error")):
        with pytest.raises(Exception, match="API Error"):
            await openai_service.make_request("Test prompt")


@pytest.mark.asyncio
async def test_get_evaluation_valid_response(openai_service: OpenAIService) -> None:
    with patch.object(openai_service, 'make_request', return_value="0.85"):
        result = await openai_service.get_evaluation("Test text")
        assert result == 0.85


@pytest.mark.asyncio
async def test_get_evaluation_invalid_response(openai_service: OpenAIService) -> None:
    with patch.object(openai_service, 'make_request', return_value="invalid"):
        result = await openai_service.get_evaluation("Test text")
        assert result is None


@pytest.mark.asyncio
async def test_get_evaluation_empty_text(openai_service: OpenAIService) -> None:
    result = await openai_service.get_evaluation("")
    assert result is None


@pytest.mark.asyncio
async def test_get_alt_success(openai_service: OpenAIService) -> None:
    with patch.object(openai_service, 'make_request', return_value="Processed text"):
        result = await openai_service.get_alt("Test text")
        assert result == "Processed text"


@pytest.mark.asyncio
async def test_get_alt_empty_text(openai_service: OpenAIService) -> None:
    result = await openai_service.get_alt("")
    assert result is None


@pytest.mark.asyncio
async def test_check_credits_available_success(openai_service: OpenAIService) -> None:
    with patch.object(openai_service, 'make_request', return_value="Test response"):
        result = await openai_service.check_credits_available()
        assert result is True


@pytest.mark.asyncio
async def test_check_credits_available_api_error(openai_service: OpenAIService) -> None:
    with patch.object(openai_service, 'make_request', side_effect=OpenAIError("API error")):
        result = await openai_service.check_credits_available()
        assert result is False


@pytest.mark.asyncio
async def test_check_credits_available_rate_limit_error(openai_service: OpenAIService) -> None:
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.headers = {}
    with patch.object(
            openai_service,
            'make_request',
            side_effect=RateLimitError("Rate limit exceeded", body=None, response=mock_response)
    ):
        result = await openai_service.check_credits_available()
        assert result is False
