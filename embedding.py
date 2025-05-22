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

        # Определяем устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Используемое устройство: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Режим оценки

        # Переносим модель на нужное устройство
        self.model = self.model.to(self.device)

        logger.info(f"Модель эмбеддингов загружена на устройство: {self.device}")

    def generate_embedding(self, text):
        """Генерирует эмбеддинг для текста"""
        if not text or text.strip() == "":
            logger.warning("Получен пустой текст для эмбеддинга")
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

            # Логируем исходную размерность для отладки
            original_size = len(embedding)

            # Обрабатываем размерность более аккуратно
            target_size = 312
            if len(embedding) != target_size:
                if len(embedding) > target_size:
                    # Используем среднее pooling вместо обрезания
                    step = len(embedding) // target_size
                    if step > 1:
                        embedding = embedding[::step][:target_size]
                    else:
                        embedding = embedding[:target_size]
                    logger.debug(f"Уменьшена размерность с {original_size} до {len(embedding)}")
                else:
                    # Дополняем нулями до нужной размерности
                    padding = np.zeros(target_size - len(embedding))
                    embedding = np.concatenate([embedding, padding])
                    logger.debug(f"Увеличена размерность с {original_size} до {len(embedding)}")

            # Нормализуем вектор
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга: {e}")
            return np.zeros(312)


# Создаем глобальный экземпляр модели для многократного использования
embedding_model = EmbeddingModel()


def generate_embedding(text):
    """Функция-обертка для создания эмбеддинга"""
    return embedding_model.generate_embedding(text)


def generate_batch_embeddings(texts, batch_size=32):
    """Функция-обертка для создания пакетных эмбеддингов"""
    return embedding_model.generate_batch_embeddings(texts, batch_size)