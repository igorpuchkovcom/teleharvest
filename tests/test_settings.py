import os

import pytest

from settings import Settings


@pytest.fixture
def env_vars():
    """Fixture to set up environment variables for testing"""
    os.environ["TELEGRAM_API_ID"] = "12345"
    os.environ["TELEGRAM_API_HASH"] = "abcdef123456"
    os.environ["TELEGRAM_PHONE"] = "+1234567890"
    os.environ["TELEGRAM_CHANNELS"] = "channel1,channel2,channel3"
    os.environ["MYSQL_HOST"] = "localhost"
    os.environ["MYSQL_USER"] = "user"
    os.environ["MYSQL_PASSWORD"] = "password"
    os.environ["MYSQL_DB"] = "database"
    os.environ["OPENAI_API_KEY"] = "sk-123456"

    yield

    # Clean up environment variables after tests
    vars_to_delete = [
        "TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_PHONE",
        "TELEGRAM_CHANNELS", "MYSQL_HOST", "MYSQL_USER",
        "MYSQL_PASSWORD", "MYSQL_DB", "OPENAI_API_KEY"
    ]
    for var in vars_to_delete:
        os.environ.pop(var, None)


def test_settings_initialization(env_vars):
    settings = Settings()

    assert settings.telegram_api_id == 12345
    assert settings.telegram_api_hash == "abcdef123456"
    assert settings.telegram_phone == "+1234567890"
    assert settings.telegram_channels == "channel1,channel2,channel3"
    assert settings.mysql_host == "localhost"
    assert settings.mysql_user == "user"
    assert settings.mysql_password == "password"
    assert settings.mysql_db == "database"
    assert settings.openai_api_key == "sk-123456"
    assert settings.openai_model == "gpt-4o-2024-05-13"  # default value
    assert settings.openai_max_tokens == 2048  # default value


def test_telegram_channels_list(env_vars):
    settings = Settings()
    channels = settings.telegram_channels_list

    assert isinstance(channels, list)
    assert len(channels) == 3
    assert channels == ["channel1", "channel2", "channel3"]

    # Test caching
    assert settings.telegram_channels_list is settings.telegram_channels_list


@pytest.fixture
def temp_prompt_file(tmp_path):
    """Fixture to create a temporary prompt file"""
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_content = "This is a test prompt"
    prompt_file.write_text(prompt_content)
    return prompt_file


def test_load_prompt_success(temp_prompt_file):
    content = Settings.load_prompt(str(temp_prompt_file))
    assert content == "This is a test prompt"


def test_load_prompt_file_not_found():
    with pytest.raises(RuntimeError) as exc_info:
        Settings.load_prompt("nonexistent_file.txt")
    assert "Error loading file: File 'nonexistent_file.txt' not found" in str(exc_info.value)


def test_load_prompt_with_invalid_encoding(tmp_path):
    # Create a binary file that's not valid UTF-8
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_bytes(b'\x80\x81')

    with pytest.raises(RuntimeError) as exc_info:
        Settings.load_prompt(str(invalid_file))
    assert "Error loading file" in str(exc_info.value)
