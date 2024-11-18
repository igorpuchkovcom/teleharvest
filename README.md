# Telegram Message Processing Service

## Description
This project is an asynchronous service designed to fetch, process, and evaluate messages from Telegram channels. It uses advanced technologies like OpenAI for text evaluation, Sentence Transformers for embeddings, and SQLAlchemy for database interaction. The goal is to identify high-quality messages, calculate their similarity scores, and manage their publication.

## Technology Stack
- **Programming Language**: Python 3.10+
- **Frameworks & Libraries**:
  - [Telethon](https://github.com/LonamiWebs/Telethon): Telegram API client for fetching messages.
  - [OpenAI API](https://openai.com/api/): For text evaluation and analysis.
  - [Sentence Transformers](https://www.sbert.net/): For embedding generation and similarity calculations.
  - [SQLAlchemy](https://www.sqlalchemy.org/): ORM for database interaction.
  - [aiomysql](https://aiomysql.readthedocs.io/): Async MySQL driver.
- **Database**: MySQL
- **Other Tools**:
  - `aiohttp`: Asynchronous HTTP requests.
  - `logging`: For structured and scalable logging.

## Project Structure
```plaintext
.
├── models/
│   ├── async_database.py         # Handles async database engine and sessions
│   ├── message.py                # SQLAlchemy ORM model for messages
├── services/
│   ├── embedding_service.py      # Handles text embeddings using Sentence Transformers
│   ├── interfaces.py             # Interface definitions for service abstractions
│   ├── openai_service.py         # Interacts with OpenAI's API for evaluations
│   ├── telegram_service.py       # Fetches messages from Telegram
├── container.py                  # Dependency injection container
├── processor.py                  # Core logic for message processing and evaluation
├── main.py                       # Entry point for the application
├── settings.py                   # Configuration and environment variables
└── requirements.txt              # Python dependencies
```

## Other Mentioned Technologies and Practices

1. **Asynchronous Programming:** Leveraging Python’s asyncio to handle concurrent operations efficiently.
2. **Dependency Injection:** Implemented via a Container class for managing services and configurations.
3. **Logging:** Structured logging to track system events and debug efficiently.
4. **Prompts for OpenAI:** Using templated prompts stored in text files for consistent evaluations.
5. **Stop Words Filtering:** Custom filters to exclude unwanted messages based on predefined keywords.

## How It Works
1. **Fetch Messages:** TelegramService retrieves messages from configured Telegram channels.
2. **Text Processing:** Messages are evaluated using OpenAIService for scoring and relevance.
3. **Embedding and Similarity:** EmbeddingService generates embeddings and calculates similarity scores.
4. **Database Interaction:** Messages are stored, retrieved, and updated in a MySQL database using AsyncDatabase.
5. **Publishing Workflow:** Relevant messages are marked and processed for publication.

## Getting Started
#### 1. Install dependencies:
```bash
pip install -r requirements.txt
```
#### 2. Configure settings in `settings.py`:
- Telegram API keys
- OpenAI API key
- MySQL credentials
- 
#### 3. Run the application:
```bash
python main.py
```
