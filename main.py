import asyncio
import logging
import os

from container import Container
from settings import Settings


async def main() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    try:
        logging_level = getattr(logging, log_level)
    except AttributeError:
        logging_level = logging.INFO

    logging.basicConfig(level=logging_level)
    logger = logging.getLogger(__name__)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings: Settings = Settings()
    container: Container = Container(settings)

    try:
        async with container.get_telegram_client() as client:
            logger.info(f"Starting TelegramClient with phone: {settings.telegram_phone}")
            await client.start(phone=settings.telegram_phone)

            async with container.get_processor() as processor:
                logger.info("Fetching and processing messages")
                await processor.fetch_and_process()
                logger.info("Updating similarity score")
                await processor.update_similarity()
                logger.info("Updating metrics")
                await processor.fetch_and_update_metrics()

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
