import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Используем модель из sentence-transformers вместо rubert-tiny2
SENTENCE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Глобальная переменная для хранения модели
_model = None


def _load_model():
    """Загружает модель эмбеддингов только один раз"""
    global _model

    if _model is None:
        logger.info(f"Загрузка модели эмбеддингов: {SENTENCE_MODEL}")
        _model = SentenceTransformer(SENTENCE_MODEL)
        logger.info(f"Модель эмбеддингов загружена")


def generate_embedding(text):
    """Генерирует эмбеддинг для текста"""
    # Загрузка модели при первом вызове
    if _model is None:
        _load_model()

    # Проверка на пустой текст
    if not text or text.strip() == "":
        logger.warning("Получен пустой текст для эмбеддинга")
        return np.zeros(384)  # Размерность MiniLM-L12

    try:
        # Генерируем эмбеддинг
        embedding = _model.encode(text)

        # Нормализуем вектор
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    except Exception as e:
        logger.error(f"Ошибка при создании эмбеддинга: {e}")
        return np.zeros(384)  # Размерность MiniLM-L12


def generate_batch_embeddings(texts, batch_size=32):
    """Генерирует эмбеддинги для пакета текстов"""
    # Загрузка модели при первом вызове
    if _model is None:
        _load_model()

    try:
        # Генерируем эмбеддинги для всего пакета
        embeddings = _model.encode(texts, batch_size=batch_size)

        # Нормализуем векторы
        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])

        logger.info(f"Обработано {len(embeddings)} текстов")
        return embeddings

    except Exception as e:
        logger.error(f"Ошибка при создании пакетных эмбеддингов: {e}")
        return [np.zeros(384) for _ in range(len(texts))]  # Размерность MiniLM-L12