import re
import torch
import json
import logging
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig, LlamaTokenizer
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

            # Попробуем разные способы загрузки токенизера
            tokenizer = None
            model = None

            # Способ 1: Стандартная загрузка
            try:
                logger.info("Попытка 1: Стандартная загрузка токенизера")
                tokenizer = AutoTokenizer.from_pretrained(
                    LLM_BASE_MODEL,
                    trust_remote_code=True,
                    use_fast=False  # Принудительно используем медленный токенизер
                )
                logger.info("✅ Токенизер загружен (стандартный способ)")
            except Exception as e:
                logger.warning(f"Способ 1 не сработал: {e}")

            # Способ 2: Загрузка как LlamaTokenizer
            if tokenizer is None:
                try:
                    logger.info("Попытка 2: Загрузка как LlamaTokenizer")
                    tokenizer = LlamaTokenizer.from_pretrained(
                        LLM_BASE_MODEL,
                        trust_remote_code=True
                    )
                    logger.info("✅ Токенизер загружен (LlamaTokenizer)")
                except Exception as e:
                    logger.warning(f"Способ 2 не сработал: {e}")

            # Способ 3: Загрузка с дополнительными параметрами
            if tokenizer is None:
                try:
                    logger.info("Попытка 3: Загрузка с дополнительными параметрами")
                    tokenizer = AutoTokenizer.from_pretrained(
                        LLM_BASE_MODEL,
                        trust_remote_code=True,
                        use_fast=False,
                        legacy=True,
                        padding_side="left"
                    )
                    logger.info("✅ Токенизер загружен (расширенные параметры)")
                except Exception as e:
                    logger.warning(f"Способ 3 не сработал: {e}")

            # Способ 4: Попробуем другую модель
            if tokenizer is None:
                logger.info("Попытка 4: Альтернативная модель")
                alternative_model = "microsoft/DialoGPT-medium"
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        alternative_model,
                        trust_remote_code=True
                    )
                    LLM_BASE_MODEL_ACTUAL = alternative_model
                    logger.info(f"✅ Токенизер загружен (альтернативная модель: {alternative_model})")
                except Exception as e:
                    logger.error(f"Все способы загрузки токенизера не сработали: {e}")
                    return None, None
            else:
                LLM_BASE_MODEL_ACTUAL = LLM_BASE_MODEL

            # Настройка токенизера для генерации
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Исправляем проблему с токенами для генерации
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # Убеждаемся что токены настроены корректно
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                if tokenizer.pad_token_id != tokenizer.eos_token_id:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

            logger.info(f"Загрузка адаптера модели: {LLM_ADAPTER_MODEL}")

            # Определяем тип данных в зависимости от поддержки
            device_available = torch.cuda.is_available()
            logger.info(f"CUDA доступна: {device_available}")

            if device_available:
                # Проверяем поддержку bfloat16 только если CUDA доступна
                try:
                    # Создаем тестовый тензор для проверки поддержки bfloat16
                    test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
                    torch_dtype = torch.bfloat16
                    logger.info("Используется bfloat16")
                except (RuntimeError, AssertionError, Exception):
                    torch_dtype = torch.float16
                    logger.info("Используется float16 (bfloat16 не поддерживается)")
            else:
                torch_dtype = torch.float32
                logger.info("Используется float32 (CPU режим)")

            # Загрузка модели с обработкой ошибок
            try:
                model = AutoPeftModelForCausalLM.from_pretrained(
                    LLM_ADAPTER_MODEL,
                    device_map="auto",
                    torch_dtype=torch_dtype
                )
                logger.info("✅ PEFT модель загружена успешно")
            except Exception as e:
                logger.warning(f"Ошибка загрузки PEFT модели: {e}")
                logger.info("Попытка загрузки базовой модели без адаптера...")

                try:
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        LLM_BASE_MODEL_ACTUAL,
                        device_map="auto",
                        torch_dtype=torch_dtype
                    )
                    logger.info("✅ Базовая модель загружена успешно")
                except Exception as e2:
                    logger.error(f"Ошибка загрузки базовой модели: {e2}")
                    return None, None

            return model, tokenizer

        except Exception as e:
            logger.error(f"Критическая ошибка при настройке языковой модели: {e}")
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
            payload = entry.payload  # Получаем payload как атрибут
            prompt += f"\n{i}. Код: {payload.get('ktru_code', '')}, Название: {payload.get('title', '')}\n"

            if payload.get('description'):
                prompt += f"   Описание: {payload.get('description')}\n"

            if 'attributes' in payload and payload.get('attributes'):
                prompt += "   Атрибуты:\n"
                for attr in payload['attributes']:
                    prompt += f"   - {attr.get('attr_name', '')}: "

                    if 'attr_values' in attr:
                        values = [val.get('value', '') for val in attr['attr_values']]
                        prompt += f"{', '.join(values)}\n"
                    elif 'attr_value' in attr:
                        prompt += f"{attr['attr_value']}\n"

        # Добавляем данные SKU в промпт
        prompt += f"\n## JSON товара для классификации:\n{json.dumps(sku_data, ensure_ascii=False, indent=2)}\n"

        return prompt

    def _find_ktru_title_by_code(self, ktru_code, search_results):
        """Поиск названия КТРУ по коду в результатах поиска"""
        try:
            for entry in search_results:
                payload = entry.payload
                if payload.get('ktru_code') == ktru_code:
                    return payload.get('title', '')

            # Если не найдено в результатах поиска, попробуем найти в базе
            if self.qdrant_client:
                search_result = self.qdrant_client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    scroll_filter={
                        "must": [
                            {
                                "key": "ktru_code",
                                "match": {"value": ktru_code}
                            }
                        ]
                    },
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )

                if search_result[0]:  # Если есть результаты
                    return search_result[0][0].payload.get('title', '')

            return None

        except Exception as e:
            logger.error(f"Ошибка при поиске названия КТРУ: {e}")
            return None

    def classify_sku(self, sku_data, top_k=TOP_K):
        """Классификация SKU по КТРУ коду"""
        if not self.qdrant_client:
            logger.error("Qdrant клиент не инициализирован")
            return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

        if not self.llm or not self.tokenizer:
            logger.error("LLM модель не инициализирована")
            # Можем попробовать классификацию только по эмбеддингам
            return self._classify_by_similarity_only(sku_data, top_k)

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
                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

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
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
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
                ktru_code = ktru_match.group(0)
                ktru_title = self._find_ktru_title_by_code(ktru_code, search_result)
                return {
                    "ktru_code": ktru_code,
                    "ktru_title": ktru_title,
                    "confidence": 1.0
                }
            elif "код не найден" in response.lower():
                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}
            else:
                # Попытка извлечь код из ответа построчно
                lines = response.split('\n')
                for line in lines:
                    ktru_match = self.ktru_pattern.search(line)
                    if ktru_match:
                        ktru_code = ktru_match.group(0)
                        ktru_title = self._find_ktru_title_by_code(ktru_code, search_result)
                        return {
                            "ktru_code": ktru_code,
                            "ktru_title": ktru_title,
                            "confidence": 1.0
                        }

                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

        except Exception as e:
            logger.error(f"Ошибка при классификации SKU: {e}")
            return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

    def _classify_by_similarity_only(self, sku_data, top_k=TOP_K):
        """Fallback классификация только по схожести эмбеддингов"""
        try:
            logger.info("Использование fallback классификации по схожести")

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
                limit=1  # Берем только самый похожий
            )

            if search_result and len(search_result) > 0:
                best_match = search_result[0]
                confidence = best_match.score

                # Устанавливаем порог схожести
                if confidence > 0.75:  # Высокая схожесть
                    logger.info(f"Найдено совпадение по схожести: {confidence:.3f}")
                    ktru_code = best_match.payload.get('ktru_code', 'код не найден')
                    ktru_title = best_match.payload.get('title', None)
                    return {
                        "ktru_code": ktru_code,
                        "ktru_title": ktru_title,
                        "confidence": confidence
                    }
                else:
                    logger.info(f"Схожесть слишком низкая: {confidence:.3f}")
                    return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}
            else:
                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

        except Exception as e:
            logger.error(f"Ошибка в fallback классификации: {e}")
            return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}


# Создаем глобальный экземпляр классификатора
classifier = KtruClassifier()


def classify_sku(sku_data, top_k=TOP_K):
    """Функция-обертка для классификации SKU"""
    return classifier.classify_sku(sku_data, top_k)