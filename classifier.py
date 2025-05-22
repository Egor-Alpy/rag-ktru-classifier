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
            # Проверяем подключение
            collections = qdrant_client.get_collections()
            logger.info(f"✅ Qdrant подключен, коллекций: {len(collections.collections)}")
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

    def create_simple_prompt(self, sku_data, similar_ktru_entries):
        """Упрощенный промпт для классификации"""
        prompt = """Ты эксперт по классификации товаров КТРУ. Твоя задача - найти точный код КТРУ для товара.

ТОВАР ДЛЯ КЛАССИФИКАЦИИ:
Название: """ + sku_data.get('title', '') + """
Описание: """ + sku_data.get('description', '') + """

ПОХОЖИЕ ТОВАРЫ ИЗ КТРУ:
"""

        # Добавляем похожие записи КТРУ в промпт (только топ-3 для краткости)
        for i, entry in enumerate(similar_ktru_entries[:3], 1):
            payload = entry.payload
            score = getattr(entry, 'score', 0)
            prompt += f"\n{i}. КОД: {payload.get('ktru_code', '')} | НАЗВАНИЕ: {payload.get('title', '')} | СХОЖЕСТЬ: {score:.3f}\n"

        prompt += """
ИНСТРУКЦИЯ: 
Выбери ТОЧНО ОДИН код КТРУ, который лучше всего подходит товару.
Ответь ТОЛЬКО кодом в формате XX.XX.XX.XXX-XXXXXXXX
Если нет подходящего кода, ответь: код не найден

ОТВЕТ:"""

        return prompt

    def _find_ktru_title_by_code(self, ktru_code, search_results):
        """Поиск названия КТРУ по коду в результатах поиска"""
        try:
            # Сначала ищем в результатах поиска
            for entry in search_results:
                payload = entry.payload
                if payload.get('ktru_code') == ktru_code:
                    title = payload.get('title', '')
                    logger.debug(f"Найдено название в результатах поиска: {title}")
                    return title

            # Если не найдено в результатах поиска, ищем в базе
            if self.qdrant_client:
                try:
                    scroll_result = self.qdrant_client.scroll(
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

                    points, next_page_offset = scroll_result
                    if points:  # Если есть результаты
                        title = points[0].payload.get('title', '')
                        logger.debug(f"Найдено название в базе: {title}")
                        return title
                except Exception as e:
                    logger.error(f"Ошибка при поиске в базе: {e}")

            logger.warning(f"Название для кода {ktru_code} не найдено")
            return None

        except Exception as e:
            logger.error(f"Ошибка при поиске названия КТРУ: {e}")
            return None

    def _debug_search_results(self, search_results, query_text):
        """Отладочная информация о результатах поиска"""
        logger.info(f"🔍 Отладка поиска для: '{query_text[:50]}...'")
        logger.info(f"📊 Найдено результатов: {len(search_results)}")

        for i, entry in enumerate(search_results[:5], 1):
            payload = entry.payload
            score = getattr(entry, 'score', 0)
            logger.info(f"  {i}. Код: {payload.get('ktru_code', 'N/A')}")
            logger.info(f"     Название: {payload.get('title', 'N/A')[:60]}...")
            logger.info(f"     Схожесть: {score:.3f}")
        logger.info("=" * 50)

    def classify_sku(self, sku_data, top_k=TOP_K):
        """Классификация SKU по КТРУ коду с отладкой"""
        logger.info(f"🚀 Начало классификации: {sku_data.get('title', 'Без названия')}")

        if not self.qdrant_client:
            logger.error("❌ Qdrant клиент не инициализирован")
            return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

        try:
            # Подготовка текста для эмбеддинга
            sku_text = f"{sku_data['title']} {sku_data.get('description', '')}"

            # Добавляем атрибуты
            if 'attributes' in sku_data and sku_data['attributes']:
                for attr in sku_data['attributes']:
                    attr_name = attr.get('attr_name', '')
                    attr_value = attr.get('attr_value', '')
                    sku_text += f" {attr_name}: {attr_value}"

            logger.info(f"📝 Текст для поиска: {sku_text}")

            # Генерируем эмбеддинг
            sku_embedding = generate_embedding(sku_text)
            logger.info(f"🔢 Размерность эмбеддинга: {len(sku_embedding)}")

            # Поиск похожих КТРУ кодов
            search_result = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=sku_embedding.tolist(),
                limit=top_k
            )

            # Отладочная информация
            self._debug_search_results(search_result, sku_text)

            if not search_result:
                logger.warning("⚠️ Не найдено похожих КТРУ записей")
                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

            # Проверяем лучший результат по схожести
            best_result = search_result[0]
            best_score = getattr(best_result, 'score', 0)
            best_payload = best_result.payload

            logger.info(f"🏆 Лучший результат:")
            logger.info(f"   Код: {best_payload.get('ktru_code', 'N/A')}")
            logger.info(f"   Название: {best_payload.get('title', 'N/A')}")
            logger.info(f"   Схожесть: {best_score:.3f}")

            # Если схожесть очень высокая (>0.85), возвращаем без LLM
            if best_score > 0.85:
                logger.info(f"✅ Высокая схожесть ({best_score:.3f}), возвращаем лучший результат")
                return {
                    "ktru_code": best_payload.get('ktru_code', 'код не найден'),
                    "ktru_title": best_payload.get('title', None),
                    "confidence": best_score
                }

            # Если нет LLM модели, используем fallback
            if not self.llm or not self.tokenizer:
                logger.warning("⚠️ LLM модель недоступна, используем fallback")
                return self._classify_by_similarity_only(sku_data, top_k)

            # Создаем упрощенный промпт
            prompt = self.create_simple_prompt(sku_data, search_result)
            logger.info(f"📋 Длина промпта: {len(prompt)} символов")

            # Токенизация промпта
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

            # Настройка параметров генерации (более детерминированная)
            generation_config = GenerationConfig(
                temperature=0.1,  # Меньше случайности
                top_p=0.9,
                repetition_penalty=1.1,
                max_new_tokens=50,  # Меньше токенов для ответа
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # Генерация ответа
            logger.info("🤖 Запуск LLM для классификации...")
            with torch.no_grad():
                generated_ids = self.llm.generate(
                    **inputs,
                    generation_config=generation_config
                )

            # Декодирование ответа
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            logger.info(f"🤖 Ответ LLM: '{response}'")

            # Проверка на наличие КТРУ кода в ответе
            ktru_match = self.ktru_pattern.search(response)

            if ktru_match:
                ktru_code = ktru_match.group(0)
                ktru_title = self._find_ktru_title_by_code(ktru_code, search_result)
                logger.info(f"✅ Найден код: {ktru_code}, название: {ktru_title}")
                return {
                    "ktru_code": ktru_code,
                    "ktru_title": ktru_title,
                    "confidence": 0.9  # Фиксированная высокая уверенность для LLM результатов
                }
            elif "код не найден" in response.lower():
                logger.info("❌ LLM ответил: код не найден")
                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}
            else:
                # Если LLM не дал четкого ответа, используем лучший результат поиска
                logger.warning(f"⚠️ Неопределенный ответ LLM: '{response}', используем лучший результат поиска")
                if best_score > 0.7:  # Понижен порог
                    return {
                        "ktru_code": best_payload.get('ktru_code', 'код не найден'),
                        "ktru_title": best_payload.get('title', None),
                        "confidence": best_score
                    }
                else:
                    return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

        except Exception as e:
            logger.error(f"❌ Ошибка при классификации SKU: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

    def _classify_by_similarity_only(self, sku_data, top_k=TOP_K):
        """Fallback классификация только по схожести эмбеддингов"""
        try:
            logger.info("🔄 Использование fallback классификации по схожести")

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
                confidence = getattr(best_match, 'score', 0)

                # Понижен порог схожести
                if confidence > 0.65:  # Было 0.75
                    logger.info(f"✅ Найдено совпадение по схожести: {confidence:.3f}")
                    ktru_code = best_match.payload.get('ktru_code', 'код не найден')
                    ktru_title = best_match.payload.get('title', None)
                    return {
                        "ktru_code": ktru_code,
                        "ktru_title": ktru_title,
                        "confidence": confidence
                    }
                else:
                    logger.info(f"❌ Схожесть слишком низкая: {confidence:.3f}")
                    return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}
            else:
                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

        except Exception as e:
            logger.error(f"❌ Ошибка в fallback классификации: {e}")
            return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}


# Создаем глобальный экземпляр классификатора
classifier = KtruClassifier()


def classify_sku(sku_data, top_k=TOP_K):
    """Функция-обертка для классификации SKU"""
    return classifier.classify_sku(sku_data, top_k)