import asyncio
import logging

from container import Container
from settings import Settings


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
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
