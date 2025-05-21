from typing import List, Optional
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from loguru import logger


class EmbeddingModelLoader:
    """Загрузчик модели для создания эмбеддингов"""

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.max_length = 512  # Максимальная длина текста для модели

    async def load_model(self):
        """Загружает модель и токенизатор"""
        from concurrent.futures import ThreadPoolExecutor
        import asyncio

        # Загрузка модели выполняется в отдельном потоке, чтобы не блокировать сервер
        with ThreadPoolExecutor() as executor:
            await asyncio.get_event_loop().run_in_executor(
                executor, self._load_model_sync
            )

    def _load_model_sync(self):
        """Синхронная загрузка модели"""
        try:
            logger.info(f"Загрузка токенизатора для модели {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            logger.info(f"Загрузка модели {self.model_id}...")
            self.model = AutoModel.from_pretrained(self.model_id)

            # Перемещаем модель на указанное устройство
            self.model = self.model.to(self.device)

            # Устанавливаем режим оценки (не обучения)
            self.model.eval()

            logger.info(f"Модель {self.model_id} успешно загружена на устройство {self.device}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def is_model_loaded(self) -> bool:
        """Проверяет, загружена ли модель"""
        return self.model is not None and self.tokenizer is not None

    async def get_embedding(self, text: str) -> List[float]:
        """Возвращает эмбеддинг для текста"""
        if not self.is_model_loaded():
            raise ValueError("Модель не загружена")

        from concurrent.futures import ThreadPoolExecutor
        import asyncio

        # Векторизация выполняется в отдельном потоке, чтобы не блокировать сервер
        with ThreadPoolExecutor() as executor:
            embedding = await asyncio.get_event_loop().run_in_executor(
                executor, lambda: self._get_embedding_sync(text)
            )

            return embedding

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Возвращает эмбеддинги для пакета текстов"""
        if not self.is_model_loaded():
            raise ValueError("Модель не загружена")

        from concurrent.futures import ThreadPoolExecutor
        import asyncio

        # Пакетная векторизация выполняется в отдельном потоке
        with ThreadPoolExecutor() as executor:
            embeddings = await asyncio.get_event_loop().run_in_executor(
                executor, lambda: self._get_embeddings_batch_sync(texts)
            )

            return embeddings

    def _get_embedding_sync(self, text: str) -> List[float]:
        """Синхронное создание эмбеддинга для одного текста"""
        try:
            # Подготовка текста в зависимости от модели
            if "e5" in self.model_id.lower():
                # Для моделей E5 префикс 'query: ' улучшает качество эмбеддингов
                text = f"query: {text}"

            # Токенизация
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Получение эмбеддинга
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Используем среднее значение по последнему слою
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Нормализация вектора (для косинусного сходства)
            embedding = embeddings[0]
            embedding = embedding / np.linalg.norm(embedding)

            return embedding.tolist()
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга: {e}")
            raise

    def _get_embeddings_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Синхронное создание эмбеддингов для пакета текстов"""
        try:
            prepared_texts = texts

            # Подготовка текстов в зависимости от модели
            if "e5" in self.model_id.lower():
                prepared_texts = [f"query: {text}" for text in texts]

            # Токенизация
            inputs = self.tokenizer(
                prepared_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Получение эмбеддингов
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Используем среднее значение по последнему слою
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Нормализация векторов
            normalized_embeddings = []
            for embedding in embeddings:
                normalized = embedding / np.linalg.norm(embedding)
                normalized_embeddings.append(normalized.tolist())

            return normalized_embeddings
        except Exception as e:
            logger.error(f"Ошибка при создании пакета эмбеддингов: {e}")
            raise