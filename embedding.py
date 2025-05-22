import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging
from config import EMBEDDING_MODEL

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, model_name=EMBEDDING_MODEL):
        """Инициализация модели для создания эмбеддингов"""
        logger.info(f"Загрузка модели эмбеддингов: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Режим оценки

        # Если доступен CUDA, переносим модель на GPU
        if torch.cuda.is_available():
            logger.info("Используем CUDA для модели эмбеддингов")
            self.model = self.model.cuda()

        self.device = self.model.device
        logger.info(f"Модель эмбеддингов загружена на устройство: {self.device}")

    def generate_embedding(self, text):
        """Генерирует эмбеддинг для текста"""
        if not text or text.strip() == "":
            logger.warning("Получен пустой текст для эмбеддинга")
            # Возвращаем нулевой вектор
            return np.zeros(312)

        try:
            # Токенизация текста
            inputs = self.tokenizer(text,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=512)

            # Перемещаем тензоры на устройство модели
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Генерируем эмбеддинг
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Используем CLS токен как представление предложения
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            # Преобразуем в одномерный массив, если размерность > 1
            if embedding.ndim > 1 and embedding.shape[0] == 1:
                embedding = embedding[0]

            # Нормализуем вектор
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга: {e}")
            # В случае ошибки возвращаем нулевой вектор
            return np.zeros(312)

    def generate_batch_embeddings(self, texts, batch_size=32):
        """Генерирует эмбеддинги для пакета текстов"""
        embeddings = []

        # Обрабатываем пакетами
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                embedding = self.generate_embedding(text)
                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)
            logger.info(f"Обработано {len(embeddings)}/{len(texts)} текстов")

        return embeddings


# Создаем глобальный экземпляр модели для многократного использования
embedding_model = EmbeddingModel()


def generate_embedding(text):
    """Функция-обертка для создания эмбеддинга"""
    return embedding_model.generate_embedding(text)


def generate_batch_embeddings(texts, batch_size=32):
    """Функция-обертка для создания пакетных эмбеддингов"""
    return embedding_model.generate_batch_embeddings(texts, batch_size)