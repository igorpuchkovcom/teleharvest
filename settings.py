from functools import cached_property
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_prefix='PROCESSOR_', extra='allow')

    limit: int = 1000
    min_views: int = 50
    min_len: int = 200
    min_er: float = 0.025
    min_score: int = 80
    min_score_alt: int = 85
    stop_words: str = ''

    @cached_property
    def stop_words_list(self) -> List[str]:
        return self.stop_words.split(',')


class TelegramSettings(BaseSettings):
    api_id: int
    api_hash: str
    phone: str
    channels: str

    model_config = SettingsConfigDict(env_file='.env', env_prefix='TELEGRAM_', extra='allow')

    @cached_property
    def channels_list(self) -> List[str]:
        return self.channels.split(',')


class MysqlSettings(BaseSettings):
    host: str
    user: str
    password: str
    db: str

    model_config = SettingsConfigDict(env_file='.env', env_prefix='MYSQL_', extra='allow')


class OpenAISettings(BaseSettings):
    api_key: str
    model: str = "gpt-4o-2024-05-13"
    max_tokens: int = 2048

    model_config = SettingsConfigDict(env_file='.env', env_prefix='OPENAI_', extra='allow')


class Settings(BaseSettings):
    telegram: TelegramSettings = TelegramSettings()
    mysql: MysqlSettings = MysqlSettings()
    openai: OpenAISettings = OpenAISettings()
    processor: ProcessorSettings = ProcessorSettings()

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file='.env', extra='allow')

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
