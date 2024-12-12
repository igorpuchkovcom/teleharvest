# Telegram Message Processing

This project is a Python-based system that processes messages from Telegram channels, evaluates them using OpenAI,
calculates text similarity using embeddings, and stores data asynchronously in a MySQL database. It leverages dependency
injection for service management and includes asynchronous processing for efficiency.

## Features

- **Telegram Integration**: Fetches messages from specified Telegram channels using the Telegram API.
- **OpenAI Evaluation**: Evaluates the content of messages using OpenAI's GPT models for generating scores and
  alternative text.
- **Text Embeddings**: Generates embeddings for messages and calculates similarity between them.
- **Asynchronous Database**: Uses SQLAlchemy with async support to manage database interactions efficiently.
- **Unit Tests**: Includes comprehensive tests for services and models using `pytest` with support for asynchronous
  testing.

## Setup

### Prerequisites

- Python 3.8+
- MySQL (or compatible database)
- Telegram API credentials
- OpenAI API key

### Installation

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables for your Telegram and OpenAI API keys. You can create a .env file or set them
   directly:

```makefile
# Telegram API Configuration
TELEGRAM_API_ID=your_telegram_api_id
TELEGRAM_API_HASH=your_telegram_api_hash
TELEGRAM_PHONE=your_telegram_phone_number
TELEGRAM_CHANNELS=@channel1,@channel2,@channel3,... # Add your Telegram channels here

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# MySQL Database Configuration
MYSQL_HOST=your_mysql_host
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DB=your_mysql_database
```

### 4. Initialize the database schema by running your database migration script.

Run the following SQL script to create the necessary table:

```sql
CREATE TABLE `post` (
  `id` int(11) unsigned NOT NULL,
  `channel` varchar(128) NOT NULL,
  `timestamp` timestamp NOT NULL,
  `text` text DEFAULT NULL,
  `score` int(3) unsigned DEFAULT NULL,
  `alt` text DEFAULT NULL,
  `score_alt` int(3) unsigned DEFAULT NULL,
  `embedding` text DEFAULT NULL,
  `similarity_score` float DEFAULT NULL,
  `published` datetime DEFAULT NULL,
  PRIMARY KEY (`id`,`channel`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

## Configuration

The project uses Pydantic for settings management, with the following configuration options:

- **Telegram settings:** Channels to monitor and API keys.
- **OpenAI settings:** Model configuration, token limits, and prompts.
- **Database settings:** Connection parameters for the MySQL database.

## Running the Application

You can start the main application by running:

```bash
python main.py
```

This will start processing messages from the specified Telegram channels.

## Testing

To run unit tests, use ``pytest``:

```bash
pytest
```

### Tests

Tests are written for services and models, including:

- **TelegramService:** Testing message fetching, filtering by stop words, and exception handling.
- **EmbeddingService:** Testing embedding generation using the `SentenceTransformer` model and similarity score
  calculation.
- **OpenAIService:** Testing API request handling, response formatting, and evaluation based on custom prompts.
- **AsyncDatabase:** Testing database initialization, context management, asynchronous queries, and connection pooling.
- **Processor:** Testing message processing pipeline, including fetching messages, embedding generation, and similarity
  score updates.
- **Message Model:** Testing creation, validation, and manipulation of `Message` objects for processing.
- **Configuration (settings.py):** Testing correct loading of environment variables, defaults, and settings parsing.
- **Container:** Testing the initialization and dependency injection of services
  like `TelegramClient`, `TelegramService`, `OpenAIService`, `AsyncDatabase`, and `EmbeddingService`.

You can also run the tests asynchronously with the `pytest-asyncio` plugin.

## Directory Structure

```bash
.
├── main.py                  # Main entry point for the application
├── requirements.txt         # Project dependencies
├── pytest.ini               # Pytest configuration
├── settings.py              # Configuration management with Pydantic
├── services/                # Service implementations (Telegram, OpenAI, Embedding, etc.)
│   ├── openai_service.py
│   ├── telegram_service.py
│   └── embedding_service.py
├── models/                  # Models and database interactions
│   ├── message.py
│   └── async_database.py
└── tests/                   # Unit tests for services and models
    ├── services/
    │   ├── test_telegram_service.py
    │   ├── test_embedding_service.py
    │   └── test_openai_service.py
    ├── models/
    │   └── test_async_database.py
    ├── conftest.py          # Pytest fixtures and configuration
    └── test_message.py      # Tests for the Message model
```

## Contributions

Feel free to fork this repository and submit pull requests with improvements, bug fixes, or new features. Ensure that
tests are included for new features or bug fixes.

[![codecov](https://codecov.io/github/igorpuchkovcom/teleharvest/graph/badge.svg?token=941ZCWBM6T)](https://codecov.io/github/igorpuchkovcom/teleharvest)