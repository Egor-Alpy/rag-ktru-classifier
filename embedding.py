"""
Модуль для создания векторных представлений текста
Оптимизирован для GPU и batch processing
"""

import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import logging
from functools import lru_cache
import gc

from config import EMBEDDING_MODEL, VECTOR_SIZE, DEVICE, BATCH_SIZE, USE_CACHE

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Менеджер для эффективной работы с эмбеддингами"""

    def __init__(self):
        """Инициализация модели эмбеддингов"""
        logger.info(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL}")

        # Загружаем модель с оптимизациями
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

        # Включаем eval mode для inference
        self.model.eval()

        # Проверяем размерность
        test_embedding = self.model.encode("test", convert_to_numpy=True)
        self.vector_size = len(test_embedding)

        if self.vector_size != VECTOR_SIZE:
            logger.warning(f"Размерность модели {self.vector_size} != конфигурации {VECTOR_SIZE}")
            logger.info(f"Используем фактическую размерность: {self.vector_size}")

        logger.info(f"✅ Модель загружена. Размерность: {self.vector_size}, Устройство: {DEVICE}")

        # Кэш для эмбеддингов
        if USE_CACHE:
            self._cache = {}

    @lru_cache(maxsize=10000)
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для улучшения качества эмбеддингов"""
        # Удаляем лишние пробелы
        text = " ".join(text.split())
        # Приводим к нижнему регистру для консистентности
        text = text.lower()
        return text.strip()

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Создание эмбеддинга для одного текста"""
        if not text:
            return np.zeros(self.vector_size)

        # Нормализация
        if normalize:
            text = self._normalize_text(text)

        # Проверяем кэш
        if USE_CACHE and text in self._cache:
            return self._cache[text]

        # Создаем эмбеддинг
        with torch.no_grad():
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 нормализация для косинусной метрики
                show_progress_bar=False
            )

        # Сохраняем в кэш
        if USE_CACHE:
            self._cache[text] = embedding

        return embedding

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Создание эмбеддингов для пакета текстов"""
        if not texts:
            return np.zeros((0, self.vector_size))

        # Нормализация
        if normalize:
            texts = [self._normalize_text(text) for text in texts]

        # Фильтруем пустые тексты
        valid_indices = [i for i, text in enumerate(texts) if text]
        valid_texts = [texts[i] for i in valid_indices]

        if not valid_texts:
            return np.zeros((len(texts), self.vector_size))

        # Создаем эмбеддинги батчами
        embeddings = []

        for i in range(0, len(valid_texts), BATCH_SIZE):
            batch = valid_texts[i:i + BATCH_SIZE]

            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=BATCH_SIZE,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

            embeddings.append(batch_embeddings)

        # Объединяем результаты
        all_embeddings = np.vstack(embeddings)

        # Восстанавливаем порядок с учетом пустых текстов
        result = np.zeros((len(texts), self.vector_size))
        for idx, valid_idx in enumerate(valid_indices):
            result[valid_idx] = all_embeddings[idx]

        return result

    def prepare_ktru_text(self, ktru_data: dict) -> str:
        """Подготовка текста из KTRU записи для эмбеддинга"""
        parts = []

        # Код KTRU
        if ktru_data.get('ktru_code'):
            parts.append(f"Код: {ktru_data['ktru_code']}")

        # Название - самое важное
        if ktru_data.get('title'):
            # Добавляем название дважды для увеличения веса
            parts.append(ktru_data['title'])
            parts.append(f"Товар: {ktru_data['title']}")

        # Описание
        if ktru_data.get('description'):
            desc = ktru_data['description'][:200]  # Ограничиваем длину
            parts.append(f"Описание: {desc}")

        # Единица измерения
        if ktru_data.get('unit'):
            parts.append(f"Единица: {ktru_data['unit']}")

        # Ключевые слова
        if ktru_data.get('keywords'):
            keywords = ' '.join(ktru_data['keywords'][:10])  # Максимум 10 ключевых слов
            parts.append(f"Ключевые слова: {keywords}")

        # Атрибуты - берем самые важные
        if ktru_data.get('attributes'):
            attr_texts = []
            for attr in ktru_data['attributes'][:5]:  # Максимум 5 атрибутов
                attr_name = attr.get('attr_name', '')

                # Обработка значений
                values = []
                if 'attr_values' in attr:
                    for val in attr['attr_values'][:3]:  # Максимум 3 значения
                        if val.get('value'):
                            values.append(val['value'])
                elif 'attr_value' in attr:
                    values.append(attr['attr_value'])

                if attr_name and values:
                    attr_texts.append(f"{attr_name}: {', '.join(values)}")

            if attr_texts:
                parts.append("Характеристики: " + "; ".join(attr_texts))

        # Объединяем все части
        return " | ".join(parts)

    def prepare_product_text(self, product_data: dict) -> str:
        """Подготовка текста из товара для эмбеддинга"""
        parts = []

        # Название
        if product_data.get('title'):
            parts.append(product_data['title'])

        # Описание
        if product_data.get('description'):
            parts.append(product_data['description'])

        # Категория
        if product_data.get('category'):
            parts.append(f"Категория: {product_data['category']}")

        # Бренд
        if product_data.get('brand') and product_data['brand'] != "Нет данных":
            parts.append(f"Бренд: {product_data['brand']}")

        # Атрибуты
        if product_data.get('attributes'):
            attr_texts = []
            for attr in product_data['attributes']:
                if isinstance(attr, dict):
                    name = attr.get('attr_name', '')
                    value = attr.get('attr_value', '')
                    if name and value:
                        attr_texts.append(f"{name}: {value}")

            if attr_texts:
                parts.append("Характеристики: " + "; ".join(attr_texts))

        return " | ".join(parts)

    def clear_cache(self):
        """Очистка кэша эмбеддингов"""
        if USE_CACHE:
            self._cache.clear()
            gc.collect()

    def get_model_info(self) -> dict:
        """Получение информации о модели"""
        return {
            "model_name": EMBEDDING_MODEL,
            "vector_size": self.vector_size,
            "device": str(self.model.device),
            "cache_size": len(self._cache) if USE_CACHE else 0
        }


# Глобальный экземпляр менеджера
embedding_manager = EmbeddingManager()


# Функции-обертки для обратной совместимости
def encode_text(text: str) -> np.ndarray:
    """Создание эмбеддинга для текста"""
    return embedding_manager.encode_single(text)


def encode_texts(texts: List[str]) -> np.ndarray:
    """Создание эмбеддингов для списка текстов"""
    return embedding_manager.encode_batch(texts)