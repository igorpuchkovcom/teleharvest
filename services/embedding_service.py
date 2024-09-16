import json
import logging
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from models.message import Message
from services.interfaces import IEmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingService(IEmbeddingService):
    def __init__(self, model_name: str = 'paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    async def generate_embedding(self, text: str) -> Optional[str]:
        if not text:
            logger.warning("Received empty text for embedding generation.")
            return

        try:
            embedding = self.model.encode(text)
            logger.debug(f"Generated embedding for text: {text[:30]}...")
            return json.dumps(embedding.tolist())
        except (ValueError, TypeError) as e:
            logger.error(f"Value error while generating embedding: {e}")
        except Exception as e:
            logger.error(f"Error occurred while generating embedding: {e}")

    async def calculate_max_similarity(self, embedding: List[float], messages: List['Message']) -> float:
        if not messages:
            logger.warning("No messages provided for similarity calculation.")
            return 0.0

        max_similarity = 0.0
        for message in messages:
            if message.embedding:
                similarity = float(self.model.similarity(embedding, json.loads(message.embedding)))
                max_similarity = max(max_similarity, similarity)
            logger.debug(f"Maximum similarity calculated: {max_similarity}")
        return max_similarity
