from typing import List, Dict, Any, Optional, Tuple
import time
import json

from embedding import Embedder
from vector_store import QdrantStore
from llm import LanguageModel
from text_processing import format_product_text
from config import CONFIDENCE_THRESHOLD
from logging_config import setup_logging

logger = setup_logging("classifier")


class ProductClassifier:
    def __init__(self):
        """
        Инициализация классификатора товаров.
        """
        self.embedder = Embedder()
        self.vector_store = QdrantStore()
        self.language_model = LanguageModel()

        # Простой кэш для уже классифицированных товаров
        self.cache = {}

        logger.info("ProductClassifier initialized")

    def get_cache_key(self, product: Dict[str, Any]) -> str:
        """
        Создает ключ кэша для товара.
        """
        # Используем только основные поля для создания кэша
        cache_fields = {
            "title": product.get("title", ""),
            "description": product.get("description", ""),
            "article": product.get("article", ""),
            "brand": product.get("brand", "")
        }
        return json.dumps(cache_fields, sort_keys=True)

    def classify_product(self, product: Dict[str, Any], top_k: int = 5) -> str:
        """
        Классифицирует товар с использованием RAG и возвращает код КТРУ или "код не найден".
        """
        # Проверяем кэш
        cache_key = self.get_cache_key(product)
        if cache_key in self.cache:
            logger.info(f"Found classification in cache: {self.cache[cache_key]}")
            return self.cache[cache_key]

        start_time = time.time()

        # Форматируем текст товара
        product_text = format_product_text(product)

        # Создаем эмбеддинг для текста товара
        product_embedding = self.embedder.create_embedding(product_text)

        # Ищем релевантные документы в векторной базе
        relevant_docs = self.vector_store.search(product_embedding, top_k=top_k)

        if not relevant_docs:
            logger.warning("No relevant documents found for product")
            return "код не найден"

        # Создаем промпт для классификации
        prompt = self.language_model.create_prompt(product_text, relevant_docs)

        # Классифицируем товар
        result = self.language_model.classify(prompt)

        # Сохраняем результат в кэш
        self.cache[cache_key] = result

        end_time = time.time()
        logger.info(f"Classification completed in {end_time - start_time:.2f} seconds: {result}")

        return result