from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from loguru import logger
from config import settings
import json


class LLMService:
    """Service for LLM interactions"""

    def __init__(self):
        self.provider = settings.llm_provider
        self.model = self._initialize_llm()
        logger.info(f"Initialized LLM: {self.provider} - {settings.llm_model}")

    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        if self.provider == "openai":
            return ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.openai_api_key
            )
        elif self.provider == "anthropic":
            return ChatAnthropic(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.anthropic_api_key
            )
        elif self.provider == "ollama":
            return ChatOllama(
                model=settings.llm_model,
                temperature=settings.llm_temperature
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

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
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response: {response.content}")
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

        response = self.model.invoke(messages)
        return response.content.strip()


# Global instance
llm_service = None


def get_llm_service() -> LLMService:
    """Get or create LLM service instance"""
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service