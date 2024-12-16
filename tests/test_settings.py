
from pathlib import Path

import pytest

from settings import Settings, TelegramSettings


def test_settings_initialization() -> None:
    settings = Settings()

    assert settings.telegram.api_id == 12345
    assert settings.telegram.api_hash == "abcdef123456"
    assert settings.telegram.phone == "+1234567890"
    assert settings.telegram.channels == "channel1,channel2,channel3"
    assert settings.mysql.host == "localhost"
    assert settings.mysql.user == "user"
    assert settings.mysql.password == "password"
    assert settings.mysql.db == "database"
    assert settings.openai.api_key == "sk-123456"
    assert settings.openai.model == "gpt-4o-2024-05-13"  # default value
    assert settings.openai.max_tokens == 2048  # default value
    assert settings.processor.limit == 1000
    assert settings.processor.min_views == 50
    assert settings.processor.min_len == 200
    assert settings.processor.min_er == 0.025
    assert settings.processor.min_score == 80
    assert settings.processor.min_score_alt == 85
    assert settings.processor.min_score_improve == 85
    assert settings.processor.stop_words == 'эфир,запись,астролог,зодиак,таро,эзотери'


def test_telegram_channels_list() -> None:
    telegram_settings = TelegramSettings()
    channels = telegram_settings.channels_list

    assert isinstance(channels, list)
    assert len(channels) == 3
    assert channels == ["channel1", "channel2", "channel3"]


@pytest.fixture
def temp_prompt_file(tmp_path: Path) -> Path:
    """Fixture to create a temporary prompt file"""
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_content = "This is a test prompt"
    prompt_file.write_text(prompt_content)
    return prompt_file


def test_load_prompt_success(temp_prompt_file: Path) -> None:
    content = Settings.load_prompt(str(temp_prompt_file))
    assert content == "This is a test prompt"


def test_load_prompt_file_not_found() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        Settings.load_prompt("nonexistent_file.txt")
    assert "Error loading file: File 'nonexistent_file.txt' not found" in str(exc_info.value)


def test_load_prompt_with_invalid_encoding(tmp_path: Path) -> None:
    # Create a binary file that's not valid UTF-8
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_bytes(b'\x80\x81')

    with pytest.raises(RuntimeError) as exc_info:
        Settings.load_prompt(str(invalid_file))
    assert "Error loading file" in str(exc_info.value)


def test_processor_stop_words_list() -> None:
    processor_settings = Settings().processor
    stop_words_list = processor_settings.stop_words_list

    assert isinstance(stop_words_list, list)
    assert len(stop_words_list) > 0
    assert stop_words_list == ["эфир", "запись", "астролог", "зодиак", "таро", "эзотери"]
