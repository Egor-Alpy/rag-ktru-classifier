import json
import httpx
from typing import List, Dict, Any, Optional
from loguru import logger

from app.config import Settings
from app.models.sku import SKUModel, KTRUMatch, ClassificationResult
from app.utils.text_preprocessor import preprocess_text


class KTRUClassifier:
    """Сервис классификации товаров по кодам КТРУ"""

    def __init__(
            self,
            qdrant_host: str,
            qdrant_port: int,
            qdrant_collection: str,
            embedding_service_url: str,
            llm_service_url: str
    ):
        self.settings = Settings()
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_collection = qdrant_collection
        self.embedding_service_url = embedding_service_url
        self.llm_service_url = llm_service_url

        # Клиенты для взаимодействия с сервисами
        self.embedding_client = httpx.AsyncClient(base_url=embedding_service_url, timeout=60.0)
        self.llm_client = httpx.AsyncClient(base_url=llm_service_url, timeout=300.0)

        # Клиент для Qdrant
        from qdrant_client import QdrantClient
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

    def __del__(self):
        """Закрываем клиенты при уничтожении объекта"""
        self.embedding_client.aclose()
        self.llm_client.aclose()

    async def get_embedding(self, text: str) -> List[float]:
        """Получает эмбеддинг для текста"""
        response = await self.embedding_client.post("/embed", json={"text": text})
        response.raise_for_status()
        return response.json()["embedding"]

    async def search_similar_ktru(self, embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Ищет похожие коды КТРУ в Qdrant"""
        from qdrant_client.models import Filter

        search_result = self.qdrant_client.search(
            collection_name=self.qdrant_collection,
            query_vector=embedding,
            limit=limit,
            score_threshold=self.settings.similarity_threshold
        )

        results = []
        for scored_point in search_result:
            results.append({
                "ktru_code": scored_point.payload.get("ktru_code"),
                "title": scored_point.payload.get("title"),
                "description": scored_point.payload.get("description", ""),
                "attributes": scored_point.payload.get("attributes", []),
                "score": scored_point.score
            })

        return results

    async def generate_llm_decision(
            self,
            sku: SKUModel,
            ktru_matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Генерирует решение о присвоении кода КТРУ с помощью LLM"""

        # Формируем промпт для LLM
        prompt = self._create_llm_prompt(sku, ktru_matches)

        # Отправляем промпт в LLM
        response = await self.llm_client.post(
            "/generate",
            json={
                "prompt": prompt,
                "max_new_tokens": 1024,
                "temperature": 0.1
            }
        )
        response.raise_for_status()

        llm_response = response.json()["text"]

        # Парсим ответ LLM
        try:
            result = self._parse_llm_response(llm_response)
            return result
        except Exception as e:
            logger.error(f"Ошибка при парсинге ответа LLM: {e}")
            return {
                "has_match": False,
                "matched_ktru_code": None,
                "explanation": f"Произошла ошибка при анализе ответа модели: {str(e)}"
            }

    def _create_llm_prompt(self, sku: SKUModel, ktru_matches: List[Dict[str, Any]]) -> str:
        """Создает промпт для LLM"""
        sku_json = sku.json(indent=2, ensure_ascii=False)

        ktru_descriptions = []
        for i, match in enumerate(ktru_matches, 1):
            ktru_desc = (
                f"Код КТРУ #{i}: {match['ktru_code']}\n"
                f"Название: {match['title']}\n"
                f"Описание: {match.get('description', 'Нет описания')}\n"
                f"Схожесть: {match['score']:.4f}\n"
            )

            if match.get('attributes'):
                ktru_desc += "Атрибуты:\n"
                for attr in match['attributes']:
                    attr_name = attr.get('attr_name', '')
                    attr_values = attr.get('attr_values', [])
                    values_str = ', '.join([str(v.get('value', '')) for v in attr_values])
                    ktru_desc += f"- {attr_name}: {values_str}\n"

            ktru_descriptions.append(ktru_desc)

        return f"""Ты — умный алгоритм, который помогает классифицировать товары по кодам КТРУ (Каталог товаров, работ, услуг).
Твоя задача — определить, соответствует ли представленное SKU одному из предложенных кодов КТРУ.

Данные о товаре (SKU):
```
json {sku_json}
        Потенциально подходящие коды КТРУ:
        {"".join(ktru_descriptions)}
        Внимательно проанализируй соответствие между товаром и кодами КТРУ. Обрати особое внимание на:

        Название товара и его соответствие названию КТРУ
        Соответствие атрибутов товара атрибутам в КТРУ
        Функциональное назначение товара

        Формат ответа должен быть в JSON:
        json{{
          "has_match": true/false,
          "matched_ktru_code": "код КТРУ или null, если нет соответствия",
          "explanation": "подробное объяснение твоего решения"
        }}
        Твой ответ должен содержать ТОЛЬКО JSON, без дополнительного текста.
        """

    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """Парсит ответ LLM для извлечения структурированных данных"""
        # Извлекаем JSON из ответа
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
        else:
            # Если нет маркеров кода, пытаемся парсить весь ответ
            json_str = llm_response

        # Парсим JSON
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            # Если не удалось распарсить JSON, возвращаем ошибку
            return {
                "has_match": False,
                "matched_ktru_code": None,
                "explanation": "Не удалось распарсить ответ модели"
            }

    async def classify(self, sku: SKUModel) -> ClassificationResult:
        """Классифицирует товар по кодам КТРУ"""
        # Подготовка данных SKU для векторизации
        sku_text = self._prepare_sku_text(sku)

        # Получаем эмбеддинг для SKU
        sku_embedding = await self.get_embedding(sku_text)

        # Ищем похожие коды КТРУ
        similar_ktru = await self.search_similar_ktru(
            embedding=sku_embedding,
            limit=self.settings.search_top_k
        )

        # Если нет похожих кодов КТРУ, возвращаем результат без совпадений
        if not similar_ktru:
            return ClassificationResult(
                sku=sku,
                has_match=False,
                explanation="Не найдено подходящих кодов КТРУ"
            )

        # Используем LLM для принятия решения
        llm_decision = await self.generate_llm_decision(sku, similar_ktru)

        # Формируем результат
        result = ClassificationResult(
            sku=sku,
            has_match=llm_decision.get("has_match", False),
            explanation=llm_decision.get("explanation", "")
        )

        # Если есть совпадение, добавляем информацию о найденном коде КТРУ
        matched_ktru_code = llm_decision.get("matched_ktru_code")
        if matched_ktru_code:
            matched_item = next(
                (item for item in similar_ktru if item["ktru_code"] == matched_ktru_code),
                None
            )

            if matched_item:
                result.matched_ktru = KTRUMatch(**matched_item)

            # Остальные варианты добавляем как альтернативные
            result.alternative_matches = [
                KTRUMatch(**item)
                for item in similar_ktru
                if item["ktru_code"] != matched_ktru_code
            ]
        else:
            # Если нет совпадений, все найденные коды добавляем как альтернативные
            result.alternative_matches = [KTRUMatch(**item) for item in similar_ktru]

        return result

    def _prepare_sku_text(self, sku: SKUModel) -> str:
        """Подготавливает текст SKU для векторизации"""
        text_parts = [
            f"Название: {sku.title}",
        ]

        if sku.description:
            text_parts.append(f"Описание: {sku.description}")

        if sku.brand:
            text_parts.append(f"Производитель: {sku.brand}")

        if sku.category:
            text_parts.append(f"Категория: {sku.category}")

        if sku.attributes:
            attrs = []
            for attr in sku.attributes:
                attrs.append(f"{attr.attr_name}: {attr.attr_value}")

            if attrs:
                text_parts.append(f"Атрибуты: {'; '.join(attrs)}")

        return " ".join(text_parts)