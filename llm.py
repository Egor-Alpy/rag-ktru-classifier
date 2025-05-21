from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

from config import HF_TOKEN, MODEL_NAME, CONFIDENCE_THRESHOLD
from logging_config import setup_logging

logger = setup_logging("llm")


class LanguageModel:
    def __init__(self):
        """
        Инициализация класса для работы с языковой моделью.
        """
        logger.info(f"Loading language model {MODEL_NAME}")

        # Инициализация tokenizer и модели
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Загрузка модели с использованием token, если предоставлен
        if HF_TOKEN:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        self.model.to(self.device)
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=100,
            temperature=0.1,  # Низкая температура для более предсказуемых ответов
            device=0 if self.device == "cuda" else -1
        )

        logger.info(f"Language model loaded successfully")

    def create_prompt(self, product_text: str, context_documents: List[Dict[str, Any]]) -> str:
        """
        Создает промпт для классификации товара с использованием найденных документов КТРУ.
        """
        # Базовый промпт с инструкциями
        base_prompt = """Я предоставлю тебе JSON-файл с описанием товара. Твоя задача - определить единственный точный код КТРУ (Каталог товаров, работ, услуг) для этого товара. Если ты не можешь определить код с высокой уверенностью (более 95%), ответь только "код не найден".

## Правила определения:
1. Анализируй все поля JSON, особое внимание обрати на:
   - title (полное наименование товара)
   - description (описание товара)
   - attributes (ключевые характеристики)
   - brand (производитель)

2. Для корректного определения кода КТРУ обязательно учитывай:
   - Точное соответствие типа товара
   - Технические характеристики
   - Специфические особенности товара, указанные в описании

3. Код КТРУ должен иметь формат XX.XX.XX.XXX-XXXXXXXX, где первые цифры соответствуют ОКПД2, а после дефиса - уникальный идентификатор в КТРУ.

## Формат ответа:
- Если определен один точный код с уверенностью >95%, выведи только этот код КТРУ, без пояснений
- Если невозможно определить точный код, выведи только фразу "код не найден"

## Информация из базы КТРУ для анализа:
"""
        # Добавляем найденные документы КТРУ
        for i, doc in enumerate(context_documents):
            base_prompt += f"\nДокумент {i + 1}:\n{doc['text']}\n"

        # Добавляем информацию о товаре
        base_prompt += f"\n## JSON товара для классификации:\n{product_text}"

        return base_prompt

    def classify(self, prompt: str) -> str:
        """
        Классифицирует товар с использованием языковой модели.
        Возвращает код КТРУ или "код не найден".
        """
        logger.debug(f"Classifying product with prompt length: {len(prompt)}")

        result = self.pipe(prompt)[0]['generated_text']

        # Извлекаем результат из ответа модели
        # Ищем строку, которая соответствует формату кода КТРУ: XX.XX.XX.XXX-XXXXXXXX
        ktru_pattern = r'\d{2}\.\d{2}\.\d{2}\.\d{3}-\d{8}'
        found_codes = re.findall(ktru_pattern, result)

        if found_codes:
            logger.info(f"Found KTRU code: {found_codes[0]}")
            return found_codes[0]
        elif "код не найден" in result.lower():
            logger.info("No KTRU code found with high confidence")
            return "код не найден"
        else:
            logger.warning(f"Unable to extract valid KTRU code from result: {result}")
            return "код не найден"