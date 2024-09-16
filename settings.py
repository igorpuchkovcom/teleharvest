from functools import cached_property
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    telegram_api_id: int
    telegram_api_hash: str
    telegram_phone: str
    telegram_channels: str

    @cached_property
    def telegram_channels_list(self) -> List[str]:
        return self.telegram_channels.split(',')

    mysql_host: str
    mysql_user: str
    mysql_password: str
    mysql_db: str

    openai_api_key: str
    openai_model: str = "gpt-4o-2024-05-13"
    openai_max_tokens: int = 2048

    class Config:
        env_file = ".env"

    @staticmethod
    def load_prompt(file_name: str) -> str:
        try:
            file_path = Path(file_name)
            if not file_path.is_file():
                raise FileNotFoundError(f"File '{file_name}' not found.")

            with file_path.open(encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise RuntimeError(f"Error loading file: {e}")
