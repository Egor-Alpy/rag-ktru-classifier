import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging
from config import EMBEDDING_MODEL, VECTOR_SIZE

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

        # Загрузка токенизера и модели
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Режим оценки

        # Переносим модель на нужное устройство
        self.model = self.model.to(self.device)

        # Определяем реальную размерность модели
        self._determine_model_dimension()

        logger.info(f"Модель эмбеддингов загружена на устройство: {self.device}")
        logger.info(f"Размерность модели: {self.model_dimension}")

    def _determine_model_dimension(self):
        """Определяем реальную размерность модели"""
        try:
            # Тестовый запуск для определения размерности
            test_text = "тест"
            inputs = self.tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Используем CLS токен
            test_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            if test_embedding.ndim > 0:
                self.model_dimension = len(test_embedding)
            else:
                self.model_dimension = test_embedding.shape[0] if hasattr(test_embedding, 'shape') else VECTOR_SIZE

            logger.info(f"✅ Определена размерность модели: {self.model_dimension}")

        except Exception as e:
            logger.warning(f"⚠️ Не удалось определить размерность модели: {e}")
            self.model_dimension = VECTOR_SIZE

    def generate_embedding(self, text):
        """Генерирует эмбеддинг для текста"""
        if not text or text.strip() == "":
            logger.warning("Получен пустой текст для эмбеддинга")
            return np.zeros(self.model_dimension)

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

            # Проверяем корректность размерности
            if len(embedding) != self.model_dimension:
                logger.warning(
                    f"Неожиданная размерность эмбеддинга: {len(embedding)}, ожидалась: {self.model_dimension}")

                # Простое решение для несоответствия размерности
                if len(embedding) > self.model_dimension:
                    embedding = embedding[:self.model_dimension]
                else:
                    # Дополняем нулями
                    padding = np.zeros(self.model_dimension - len(embedding))
                    embedding = np.concatenate([embedding, padding])

            # Нормализуем вектор
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                logger.warning("Получен нулевой вектор при нормализации")

            return embedding

        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return np.zeros(self.model_dimension)

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

            if i % (batch_size * 10) == 0:  # Логируем каждые 10 пакетов
                logger.info(f"Обработано {len(embeddings)}/{len(texts)} текстов")

        return embeddings

    def test_embedding_quality(self):
        """Тестирование качества эмбеддингов"""
        logger.info("🧪 Тестирование качества эмбеддингов...")

        test_cases = [
            "ноутбук компьютер",
            "ручка канцелярские товары",
            "стол мебель офисная",
            "бумага копировальная",
            "принтер оргтехника"
        ]

        embeddings = []
        for text in test_cases:
            emb = self.generate_embedding(text)
            embeddings.append(emb)
            logger.info(f"   '{text}' -> размерность: {len(emb)}, норма: {np.linalg.norm(emb):.3f}")

        # Проверяем схожесть между парами
        logger.info("🔍 Схожесть между тестовыми эмбеддингами:")
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j])
                logger.info(f"   '{test_cases[i]}' <-> '{test_cases[j]}': {similarity:.3f}")


# Создаем глобальный экземпляр модели для многократного использования
embedding_model = EmbeddingModel()


def generate_embedding(text):
    """Функция-обертка для создания эмбеддинга"""
    return embedding_model.generate_embedding(text)


def generate_batch_embeddings(texts, batch_size=32):
    """Функция-обертка для создания пакетных эмбеддингов"""
    return embedding_model.generate_batch_embeddings(texts, batch_size)


def test_embeddings():
    """Функция для тестирования эмбеддингов"""
    embedding_model.test_embedding_quality()


# Автоматический тест при импорте (только в отладочном режиме)
if __name__ == "__main__":
    test_embeddings()