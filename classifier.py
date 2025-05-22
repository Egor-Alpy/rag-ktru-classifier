import re
import torch
import json
import logging
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from qdrant_client import QdrantClient
from embedding import generate_embedding
from config import (
    LLM_BASE_MODEL, LLM_ADAPTER_MODEL, QDRANT_HOST, QDRANT_PORT,
    QDRANT_COLLECTION, TEMPERATURE, TOP_P, REPETITION_PENALTY,
    MAX_NEW_TOKENS, TOP_K
)

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if not torch.cuda.is_available():
    logger.warning("CUDA недоступна, используется CPU")

if not torch.cuda.is_bf16_supported():
    logger.warning("bfloat16 не поддерживается, используется float16")


class KtruClassifier:
    def __init__(self):
        """Инициализация классификатора КТРУ"""
        self.qdrant_client = self._setup_qdrant()
        self.llm, self.tokenizer = self._setup_llm()

        # Компилируем шаблон для поиска КТРУ кода
        self.ktru_pattern = re.compile(r'\d{2}\.\d{2}\.\d{2}\.\d{3}-\d{8}')

    def _setup_qdrant(self):
        """Настройка клиента Qdrant"""
        try:
            qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            return qdrant_client
        except Exception as e:
            logger.error(f"Ошибка при настройке клиента Qdrant: {e}")
            return None

    def _setup_llm(self):
        """Настройка языковой модели"""
        try:
            logger.info(f"Загрузка базовой модели: {LLM_BASE_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_MODEL, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token

            logger.info(f"Загрузка адаптера модели: {LLM_ADAPTER_MODEL}")

            # Определяем тип данных в зависимости от поддержки
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
                logger.info("Используется bfloat16")
            elif torch.cuda.is_available():
                torch_dtype = torch.float16
                logger.info("Используется float16 (bfloat16 не поддерживается)")
            else:
                torch_dtype = torch.float32
                logger.info("Используется float32 (CPU режим)")

            model = AutoPeftModelForCausalLM.from_pretrained(
                LLM_ADAPTER_MODEL,
                device_map="auto",
                torch_dtype=torch_dtype
            )

            return model, tokenizer
        except Exception as e:
            logger.error(f"Ошибка при настройке языковой модели: {e}")
            return None, None

    def create_prompt(self, sku_data, similar_ktru_entries):
        """Создание промпта для классификации"""
        prompt = """Я предоставлю тебе JSON-файл с описанием товара и список похожих товаров из каталога КТРУ. 
Твоя задача - определить единственный точный код КТРУ для этого товара.
Если ты не можешь определить код с высокой уверенностью (более 95%), ответь только "код не найден".

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
3. Код КТРУ должен иметь формат XX.XX.XX.XXX-XXXXXXXX

## Похожие товары из каталога КТРУ:
"""

        # Добавляем похожие записи КТРУ в промпт
        for i, entry in enumerate(similar_ktru_entries, 1):
            prompt += f"\n{i}. Код: {entry['payload']['ktru_code']}, Название: {entry['payload']['title']}\n"

            if entry['payload'].get('description'):
                prompt += f"   Описание: {entry['payload']['description']}\n"

            if 'attributes' in entry['payload'] and entry['payload']['attributes']:
                prompt += "   Атрибуты:\n"
                for attr in entry['payload']['attributes']:
                    prompt += f"   - {attr.get('attr_name', '')}: "

                    if 'attr_values' in attr:
                        values = [val.get('value', '') for val in attr['attr_values']]
                        prompt += f"{', '.join(values)}\n"
                    elif 'attr_value' in attr:
                        prompt += f"{attr['attr_value']}\n"

        # Добавляем данные SKU в промпт
        prompt += f"\n## JSON товара для классификации:\n{json.dumps(sku_data, ensure_ascii=False, indent=2)}\n"

        return prompt

    def classify_sku(self, sku_data, top_k=TOP_K):
        """Классификация SKU по КТРУ коду"""
        if not self.qdrant_client or not self.llm or not self.tokenizer:
            logger.error("Не инициализированы компоненты классификатора")
            return "код не найден"

        try:
            # Подготовка текста для эмбеддинга
            sku_text = f"{sku_data['title']} {sku_data.get('description', '')}"

            # Добавляем атрибуты
            if 'attributes' in sku_data and sku_data['attributes']:
                for attr in sku_data['attributes']:
                    attr_name = attr.get('attr_name', '')
                    attr_value = attr.get('attr_value', '')
                    sku_text += f" {attr_name}: {attr_value}"

            # Генерируем эмбеддинг
            sku_embedding = generate_embedding(sku_text)

            # Поиск похожих КТРУ кодов
            search_result = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=sku_embedding.tolist(),
                limit=top_k
            )

            logger.info(f"Найдено {len(search_result)} похожих КТРУ записей")

            if not search_result:
                logger.warning("Не найдено похожих КТРУ записей")
                return "код не найден"

            # Создаем промпт с извлеченным контекстом
            prompt = self.create_prompt(sku_data, search_result)

            # Токенизация промпта
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)

            # Настройка параметров генерации
            generation_config = GenerationConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True
            )

            # Генерация ответа
            with torch.no_grad():
                generated_ids = self.llm.generate(
                    **inputs,
                    generation_config=generation_config
                )

            # Декодирование ответа
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Извлечение результата после промпта
            response = response[len(prompt):].strip()

            logger.info(f"Сырой ответ модели: {response}")

            # Проверка на наличие КТРУ кода в ответе
            ktru_match = self.ktru_pattern.search(response)

            if ktru_match:
                return ktru_match.group(0)
            elif "код не найден" in response.lower():
                return "код не найден"
            else:
                # Попытка извлечь код из ответа построчно
                lines = response.split('\n')
                for line in lines:
                    ktru_match = self.ktru_pattern.search(line)
                    if ktru_match:
                        return ktru_match.group(0)

                return "код не найден"

        except Exception as e:
            logger.error(f"Ошибка при классификации SKU: {e}")
            return "код не найден"


# Создаем глобальный экземпляр классификатора
classifier = KtruClassifier()


def classify_sku(sku_data, top_k=TOP_K):
    """Функция-обертка для классификации SKU"""
    return classifier.classify_sku(sku_data, top_k)