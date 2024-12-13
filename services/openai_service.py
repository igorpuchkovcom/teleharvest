import logging
from typing import Optional, Type

import aiohttp
from openai import OpenAI

from services.interfaces import IOpenAIService
from settings import OpenAISettings

logger = logging.getLogger(__name__)


class OpenAIService(IOpenAIService):

    def __init__(self, config: OpenAISettings, prompt_process: str, prompt_evaluate: str):
        self.api_key = config.api_key
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.prompt_process = prompt_process
        self.prompt_evaluate = prompt_evaluate
        self.client = OpenAI(api_key=self.api_key)

    async def __aenter__(self) -> 'OpenAIService':
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[Type[BaseException]],
    ) -> None:
        await self.session.close()

    async def make_request(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error occurred while making request to OpenAI: {e}")
            return None

    async def get_evaluation(self, text: str) -> Optional[float]:
        if not text:
            return None

        prompt: str = self.prompt_evaluate.format(text=text)
        response: Optional[str] = await self.make_request(prompt)

        if response:
            try:
                return float(response.replace('"', '').strip())
            except ValueError:
                logger.error(f"Invalid response received from OpenAI: {response}")
                return None

    async def get_alt(self, text: str) -> Optional[str]:
        if not text:
            return None

        prompt: str = self.prompt_process.format(text=text)
        return await self.make_request(prompt)
