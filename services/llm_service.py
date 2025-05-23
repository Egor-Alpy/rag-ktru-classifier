from typing import Dict, Any, Optional, List, Tuple
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from loguru import logger
from config import settings
import json


class LLMService:
    """Service for LLM interactions using only local models"""

    def __init__(self):
        self.provider = "ollama"  # Только локальный провайдер
        self.model = self._initialize_llm()
        logger.info(f"Initialized local LLM: {settings.llm_model}")

    def _initialize_llm(self):
        """Initialize local LLM"""
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_host,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens
        )

    def classify_product(
            self,
            product_info: Dict[str, Any],
            candidates: List[Tuple[Dict[str, Any], float]]
    ) -> Dict[str, Any]:
        """
        Classify product using LLM

        Args:
            product_info: Product information
            candidates: List of (ktru_info, score) candidates

        Returns:
            Classification result
        """
        system_prompt = """Вы - эксперт по классификации товаров в системе КТРУ (Каталог товаров, работ, услуг).

Ваша задача:
1. Проанализировать информацию о товаре
2. Изучить предложенные коды КТРУ
3. Выбрать ЕДИНСТВЕННЫЙ наиболее подходящий код
4. Обосновать свой выбор

Правила классификации:
- Код должен максимально точно соответствовать товару
- Учитывайте все характеристики: название, описание, категорию, атрибуты
- Если несколько кодов подходят, выберите наиболее специфичный (с большим уровнем иерархии)
- Если ни один код не подходит с уверенностью >95%, верните {"code": null, "confidence": 0}

Формат ответа - строго JSON:
{
    "code": "XX.XX.XX.XXX-XXXXXXXX",
    "confidence": 0.98,
    "reasoning": "Краткое обоснование выбора"
}"""

        # Prepare candidates information
        candidates_text = []
        for i, (ktru, score) in enumerate(candidates[:settings.max_candidates]):
            candidate_info = (
                f"{i + 1}. Код: {ktru['code']}\n"
                f"   Наименование: {ktru['name']}\n"
                f"   Описание: {ktru.get('description', 'Нет описания')}\n"
                f"   Уровень иерархии: {ktru.get('level', 'Н/Д')}\n"
                f"   Релевантность: {score:.3f}\n"
            )
            candidates_text.append(candidate_info)

        # Prepare product information
        product_text = (
            f"Название: {product_info.get('title', 'Н/Д')}\n"
            f"Описание: {product_info.get('description', 'Н/Д')}\n"
            f"Категория: {product_info.get('category', 'Н/Д')}\n"
            f"Бренд: {product_info.get('brand', 'Н/Д')}\n"
            f"Артикул: {product_info.get('article', 'Н/Д')}\n"
        )

        # Add attributes
        attributes = product_info.get('attributes', [])
        if attributes:
            attrs_text = "Характеристики:\n"
            for attr in attributes:
                if isinstance(attr, dict):
                    attrs_text += f"- {attr.get('name', 'Н/Д')}: {attr.get('value', 'Н/Д')}\n"
            product_text += attrs_text

        user_prompt = f"""Информация о товаре:
{product_text}

Найденные коды КТРУ:
{''.join(candidates_text)}

Проанализируйте и выберите наиболее подходящий код КТРУ."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.model.invoke(messages)

            # Parse JSON response
            try:
                # Ollama может возвращать ответ в разных форматах
                content = response.content if hasattr(response, 'content') else str(response)

                # Попытка найти JSON в ответе
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1

                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    # Если JSON не найден, пытаемся распарсить весь ответ
                    result = json.loads(content)

                # Проверяем наличие необходимых полей
                if 'code' not in result:
                    result['code'] = None
                if 'confidence' not in result:
                    result['confidence'] = 0
                if 'reasoning' not in result:
                    result['reasoning'] = "Нет обоснования"

                return result

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response: {content[:200]}...")
                return {
                    "code": None,
                    "confidence": 0,
                    "reasoning": "Ошибка парсинга ответа LLM"
                }

        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return {
                "code": None,
                "confidence": 0,
                "reasoning": f"Ошибка LLM: {str(e)}"
            }

    def generate_search_query(self, product_info: Dict[str, Any]) -> str:
        """Generate optimized search query for product"""
        prompt = """На основе информации о товаре создайте оптимальный поисковый запрос для поиска кода КТРУ.

Включите ключевые слова, тип товара, важные характеристики.
Запрос должен быть на русском языке, краткий и точный.

Информация о товаре:
{product_info}

Поисковый запрос:"""

        messages = [
            HumanMessage(content=prompt.format(product_info=json.dumps(product_info, ensure_ascii=False)))
        ]

        try:
            response = self.model.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            return content.strip()
        except Exception as e:
            logger.error(f"Error generating search query: {e}")
            # Возвращаем простой запрос на основе названия
            return product_info.get('title', 'товар')


# Global instance
llm_service = None


def get_llm_service() -> LLMService:
    """Get or create LLM service instance"""
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service