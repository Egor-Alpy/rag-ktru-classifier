import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from loguru import logger
from config import settings
import torch


class EmbeddingService:
    """Service for creating text embeddings"""

    def __init__(self):
        self.model_name = settings.embedding_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing embedding model: {self.model_name} on {self.device}")

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def encode(
            self,
            texts: Union[str, List[str]],
            batch_size: int = None,
            show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts into embeddings

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or settings.embedding_batch_size

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    def encode_product(self, product_info: dict) -> np.ndarray:
        """
        Create embedding for product information

        Args:
            product_info: Product information dictionary

        Returns:
            Product embedding
        """
        # Combine relevant fields for embedding
        text_parts = []

        if product_info.get("title"):
            text_parts.append(f"Название: {product_info['title']}")

        if product_info.get("description"):
            text_parts.append(f"Описание: {product_info['description']}")

        if product_info.get("category"):
            text_parts.append(f"Категория: {product_info['category']}")

        if product_info.get("brand") and product_info["brand"] != "Нет данных":
            text_parts.append(f"Бренд: {product_info['brand']}")

        # Add attributes
        attributes = product_info.get("attributes", [])
        for attr in attributes:
            if isinstance(attr, dict) and "name" in attr and "value" in attr:
                text_parts.append(f"{attr['name']}: {attr['value']}")

        combined_text = " ".join(text_parts)
        return self.encode(combined_text)[0]

    def encode_ktru(self, ktru_info: dict) -> np.ndarray:
        """
        Create embedding for KTRU code information

        Args:
            ktru_info: KTRU code information dictionary

        Returns:
            KTRU embedding
        """
        text_parts = []

        if ktru_info.get("code"):
            text_parts.append(f"Код КТРУ: {ktru_info['code']}")

        if ktru_info.get("name"):
            text_parts.append(f"Наименование: {ktru_info['name']}")

        if ktru_info.get("description"):
            text_parts.append(f"Описание: {ktru_info['description']}")

        combined_text = " ".join(text_parts)
        return self.encode(combined_text)[0]


# Global instance
embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service instance"""
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service