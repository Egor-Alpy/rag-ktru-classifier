from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

from config import HF_TOKEN, EMBEDDING_MODEL, BATCH_SIZE
from logging_config import setup_logging

logger = setup_logging("embedding")


class Embedder:
    def __init__(self):
        """
        Инициализация класса для создания эмбеддингов.
        Используется модель SentenceTransformer для создания векторных представлений текста.
        """
        logger.info(f"Loading embedding model {EMBEDDING_MODEL}")

        # Инициализация модели
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Загрузка модели БЕЗ токена
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Model {EMBEDDING_MODEL} loaded successfully without token")
        except Exception as e:
            logger.error(f"Error loading model {EMBEDDING_MODEL}: {e}")
            logger.info("Trying fallback model all-MiniLM-L6-v2")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Fallback model loaded successfully")

        self.model.to(self.device)
        logger.info(f"Embedding model loaded successfully on {self.device}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги для списка текстов.
        Обрабатывает тексты батчами для оптимизации производительности.
        """
        if not texts:
            return []

        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            logger.debug(f"Creating embeddings for batch {i // BATCH_SIZE + 1} of {(len(texts) - 1) // BATCH_SIZE + 1}")

            # Создание эмбеддингов для текущего батча
            batch_embeddings = self.model.encode(batch_texts)
            embeddings.extend(batch_embeddings)

        logger.info(f"Created embeddings for {len(texts)} texts")
        return embeddings

    def create_embedding(self, text: str) -> List[float]:
        """
        Создает эмбеддинг для одного текста.
        """
        embedding = self.model.encode(text)
        return embedding.tolist()