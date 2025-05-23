import re
import torch
import json
import logging
from typing import List, Dict, Optional, Tuple
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig, LlamaTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from embedding import generate_embedding
from config import (
    LLM_BASE_MODEL, LLM_ADAPTER_MODEL, QDRANT_HOST, QDRANT_PORT,
    QDRANT_COLLECTION, TEMPERATURE, TOP_P, REPETITION_PENALTY,
    MAX_NEW_TOKENS, TOP_K, SIMILARITY_THRESHOLD_HIGH,
    SIMILARITY_THRESHOLD_MEDIUM, SIMILARITY_THRESHOLD_LOW,
    CLASSIFICATION_CONFIDENCE_THRESHOLD, ENABLE_ATTRIBUTE_MATCHING,
    ATTRIBUTE_WEIGHT
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

        # Словарь для нормализации атрибутов
        self.attribute_normalization = {
            "количество слоев": ["слойность", "число слоев", "слои"],
            "цвет": ["окраска", "расцветка", "оттенок"],
            "тип": ["вид", "разновидность", "категория"],
            "материал": ["состав", "сырье", "основа"],
            "размер": ["габарит", "величина", "формат"],
            "назначение": ["применение", "использование", "цель"]
        }

    def _setup_qdrant(self):
        """Настройка клиента Qdrant"""
        try:
            qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
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

            # Загрузка токенизера
            tokenizer = None
            model = None

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    LLM_BASE_MODEL,
                    trust_remote_code=True,
                    use_fast=False
                )
                logger.info("✅ Токенизер загружен")
            except Exception as e:
                logger.warning(f"Ошибка загрузки токенизера: {e}")
                try:
                    tokenizer = LlamaTokenizer.from_pretrained(
                        LLM_BASE_MODEL,
                        trust_remote_code=True
                    )
                    logger.info("✅ Токенизер загружен (LlamaTokenizer)")
                except Exception as e2:
                    logger.error(f"Не удалось загрузить токенизер: {e2}")
                    return None, None

            # Настройка токенизера
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # Определяем тип данных
            device_available = torch.cuda.is_available()
            logger.info(f"CUDA доступна: {device_available}")

            if device_available:
                try:
                    test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
                    torch_dtype = torch.bfloat16
                    logger.info("Используется bfloat16")
                except:
                    torch_dtype = torch.float16
                    logger.info("Используется float16")
            else:
                torch_dtype = torch.float32
                logger.info("Используется float32 (CPU режим)")

            # Загрузка модели
            try:
                model = AutoPeftModelForCausalLM.from_pretrained(
                    LLM_ADAPTER_MODEL,
                    device_map="auto",
                    torch_dtype=torch_dtype
                )
                logger.info("✅ PEFT модель загружена успешно")
            except Exception as e:
                logger.warning(f"Ошибка загрузки PEFT модели: {e}")
                try:
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        LLM_BASE_MODEL,
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

    def _normalize_attribute_name(self, attr_name: str) -> str:
        """Нормализация названия атрибута"""
        attr_lower = attr_name.lower().strip()

        # Проверяем словарь нормализации
        for normalized, variants in self.attribute_normalization.items():
            if attr_lower == normalized or attr_lower in variants:
                return normalized

        return attr_lower

    def _extract_attributes(self, data: Dict) -> Dict[str, str]:
        """Извлечение и нормализация атрибутов"""
        attributes = {}

        if 'attributes' in data and isinstance(data['attributes'], list):
            for attr in data['attributes']:
                if isinstance(attr, dict):
                    # Для SKU формат
                    if 'attr_name' in attr and 'attr_value' in attr:
                        name = self._normalize_attribute_name(attr['attr_name'])
                        attributes[name] = str(attr['attr_value']).lower()
                    # Для KTRU формат
                    elif 'attr_name' in attr and 'attr_values' in attr:
                        name = self._normalize_attribute_name(attr['attr_name'])
                        values = []
                        for val in attr['attr_values']:
                            if isinstance(val, dict) and 'value' in val:
                                values.append(str(val['value']).lower())
                        if values:
                            attributes[name] = '; '.join(values)

        return attributes

    def _calculate_attribute_similarity(self, sku_attrs: Dict[str, str], ktru_attrs: Dict[str, str]) -> float:
        """Расчет схожести атрибутов"""
        if not sku_attrs or not ktru_attrs:
            return 0.0

        matches = 0
        total_comparisons = 0

        # Проверяем совпадение атрибутов
        for sku_attr_name, sku_attr_value in sku_attrs.items():
            if sku_attr_name in ktru_attrs:
                ktru_value = ktru_attrs[sku_attr_name]
                total_comparisons += 1

                # Точное совпадение
                if sku_attr_value == ktru_value:
                    matches += 1
                # Частичное совпадение
                elif sku_attr_value in ktru_value or ktru_value in sku_attr_value:
                    matches += 0.5
                # Проверка на вхождение значений (для множественных значений)
                elif ';' in ktru_value:
                    ktru_values = [v.strip() for v in ktru_value.split(';')]
                    if any(sku_attr_value in v or v in sku_attr_value for v in ktru_values):
                        matches += 0.5

        if total_comparisons == 0:
            return 0.0

        return matches / total_comparisons

    def _create_advanced_prompt(self, sku_data: Dict, similar_ktru_entries: List,
                                sku_attrs: Dict[str, str]) -> str:
        """Создание продвинутого промпта для классификации"""
        prompt = """Ты эксперт по классификации товаров в системе КТРУ (Каталог товаров, работ, услуг).
Твоя задача - найти ЕДИНСТВЕННЫЙ ТОЧНЫЙ код КТРУ для товара с уверенностью не менее 95%.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. Код КТРУ должен ТОЧНО соответствовать типу товара, а не просто быть похожим
2. Проверь ВСЕ атрибуты товара - они должны соответствовать характеристикам КТРУ
3. Если есть хоть малейшие сомнения в точности - отвечай "код не найден"
4. НЕ выбирай код общей категории, если есть более специфичный код
5. Учитывай ВСЕ детали: бренд, модель, технические характеристики

ТОВАР ДЛЯ КЛАССИФИКАЦИИ:
"""

        # Добавляем информацию о товаре
        prompt += f"Название: {sku_data.get('title', '')}\n"
        if sku_data.get('description'):
            prompt += f"Описание: {sku_data.get('description', '')}\n"
        if sku_data.get('category'):
            prompt += f"Категория: {sku_data.get('category', '')}\n"
        if sku_data.get('brand'):
            prompt += f"Бренд: {sku_data.get('brand', '')}\n"

        # Добавляем атрибуты
        if sku_attrs:
            prompt += "\nАТРИБУТЫ ТОВАРА:\n"
            for attr_name, attr_value in sku_attrs.items():
                prompt += f"- {attr_name}: {attr_value}\n"

        prompt += "\nКАНДИДАТЫ ИЗ КТРУ (отсортированы по релевантности):\n"

        # Добавляем кандидатов с подробной информацией
        for i, entry in enumerate(similar_ktru_entries[:10], 1):  # Топ-10 для анализа
            payload = entry.payload
            score = getattr(entry, 'score', 0)
            ktru_attrs = self._extract_attributes(payload)
            attr_similarity = self._calculate_attribute_similarity(sku_attrs, ktru_attrs)

            prompt += f"\n{i}. КОД: {payload.get('ktru_code', '')}\n"
            prompt += f"   НАЗВАНИЕ: {payload.get('title', '')}\n"
            prompt += f"   СХОЖЕСТЬ ТЕКСТА: {score:.3f}\n"
            prompt += f"   СХОЖЕСТЬ АТРИБУТОВ: {attr_similarity:.2f}\n"

            if ktru_attrs:
                prompt += "   АТРИБУТЫ КТРУ:\n"
                for attr_name, attr_value in list(ktru_attrs.items())[:5]:  # Первые 5 атрибутов
                    prompt += f"   - {attr_name}: {attr_value}\n"

        prompt += """
ИНСТРУКЦИЯ ПО ВЫБОРУ:
1. Найди кандидата, где название И атрибуты максимально точно соответствуют товару
2. Если схожесть текста > 0.9 И схожесть атрибутов > 0.7 - это хороший кандидат
3. Проверь, что ВСЕ ключевые характеристики товара есть в описании КТРУ
4. Если несколько кандидатов подходят - выбери наиболее специфичный (не общую категорию)
5. Если ни один кандидат не подходит с уверенностью 95% - ответь "код не найден"

ФОРМАТ ОТВЕТА:
- Если найден точный код: выведи ТОЛЬКО код в формате XX.XX.XX.XXX-XXXXXXXX
- Если код не найден: выведи ТОЛЬКО фразу "код не найден"

ОТВЕТ:"""

        return prompt

    def _validate_ktru_match(self, sku_data: Dict, ktru_data: Dict,
                             text_similarity: float, attr_similarity: float) -> Tuple[bool, float]:
        """Валидация соответствия SKU и KTRU"""
        # Базовая оценка уверенности
        confidence = 0.0

        # Вклад текстовой схожести (70%)
        if text_similarity >= SIMILARITY_THRESHOLD_HIGH:
            confidence += 0.7
        elif text_similarity >= SIMILARITY_THRESHOLD_MEDIUM:
            confidence += 0.5
        elif text_similarity >= SIMILARITY_THRESHOLD_LOW:
            confidence += 0.3
        else:
            return False, 0.0

        # Вклад схожести атрибутов (30%)
        if ENABLE_ATTRIBUTE_MATCHING:
            confidence += attr_similarity * ATTRIBUTE_WEIGHT

        # Дополнительные проверки
        sku_title_lower = sku_data.get('title', '').lower()
        ktru_title_lower = ktru_data.get('title', '').lower()

        # Проверка ключевых слов
        sku_keywords = set(sku_title_lower.split())
        ktru_keywords = set(ktru_title_lower.split())

        # Должно быть хотя бы 30% общих ключевых слов
        if len(sku_keywords) > 0 and len(ktru_keywords) > 0:
            common_keywords = sku_keywords.intersection(ktru_keywords)
            keyword_overlap = len(common_keywords) / min(len(sku_keywords), len(ktru_keywords))
            if keyword_overlap < 0.3:
                confidence *= 0.7  # Снижаем уверенность

        # Проверка категории, если есть
        if 'category' in sku_data and sku_data['category']:
            sku_category = sku_data['category'].lower()
            if sku_category not in ktru_title_lower and sku_category not in ktru_data.get('description', '').lower():
                confidence *= 0.8  # Снижаем уверенность

        # Финальная проверка
        is_valid = confidence >= CLASSIFICATION_CONFIDENCE_THRESHOLD

        return is_valid, confidence

    def classify_sku(self, sku_data: Dict, top_k: int = TOP_K) -> Dict:
        """Классификация SKU по КТРУ коду с высокой точностью"""
        logger.info(f"🚀 Начало классификации: {sku_data.get('title', 'Без названия')}")

        if not self.qdrant_client:
            logger.error("❌ Qdrant клиент не инициализирован")
            return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

        try:
            # Извлечение атрибутов SKU
            sku_attrs = self._extract_attributes(sku_data)
            logger.info(f"📋 Извлечено атрибутов SKU: {len(sku_attrs)}")

            # Подготовка текста для эмбеддинга
            sku_text_parts = [
                sku_data.get('title', ''),
                sku_data.get('description', ''),
                sku_data.get('category', ''),
                sku_data.get('brand', '')
            ]

            # Добавляем атрибуты
            for attr_name, attr_value in sku_attrs.items():
                sku_text_parts.append(f"{attr_name}: {attr_value}")

            sku_text = ' '.join(filter(None, sku_text_parts))
            logger.info(f"📝 Текст для поиска: {sku_text[:100]}...")

            # Генерируем эмбеддинг
            sku_embedding = generate_embedding(sku_text)
            logger.info(f"🔢 Размерность эмбеддинга: {len(sku_embedding)}")

            # Поиск похожих КТРУ кодов
            search_result = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=sku_embedding.tolist(),
                limit=top_k
            )

            if not search_result:
                logger.warning("⚠️ Не найдено похожих КТРУ записей")
                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

            # Анализ результатов
            best_candidates = []

            for result in search_result:
                score = getattr(result, 'score', 0)
                payload = result.payload

                # Извлекаем атрибуты KTRU
                ktru_attrs = self._extract_attributes(payload)

                # Рассчитываем схожесть атрибутов
                attr_similarity = self._calculate_attribute_similarity(sku_attrs, ktru_attrs)

                # Валидация соответствия
                is_valid, confidence = self._validate_ktru_match(
                    sku_data, payload, score, attr_similarity
                )

                if is_valid:
                    best_candidates.append({
                        'result': result,
                        'confidence': confidence,
                        'text_similarity': score,
                        'attr_similarity': attr_similarity
                    })

            # Сортируем кандидатов по уверенности
            best_candidates.sort(key=lambda x: x['confidence'], reverse=True)

            # Если есть кандидат с очень высокой уверенностью
            if best_candidates and best_candidates[0]['confidence'] >= 0.98:
                best = best_candidates[0]
                payload = best['result'].payload
                logger.info(f"✅ Найдено точное совпадение с уверенностью {best['confidence']:.3f}")
                return {
                    "ktru_code": payload.get('ktru_code', 'код не найден'),
                    "ktru_title": payload.get('title', None),
                    "confidence": best['confidence']
                }

            # Если нет LLM модели и нет высокой уверенности
            if not self.llm or not self.tokenizer:
                if best_candidates:
                    best = best_candidates[0]
                    if best['confidence'] >= CLASSIFICATION_CONFIDENCE_THRESHOLD:
                        payload = best['result'].payload
                        return {
                            "ktru_code": payload.get('ktru_code', 'код не найден'),
                            "ktru_title": payload.get('title', None),
                            "confidence": best['confidence']
                        }

                logger.warning("⚠️ LLM модель недоступна и нет кандидатов с высокой уверенностью")
                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

            # Используем LLM для финального выбора
            prompt = self._create_advanced_prompt(sku_data, search_result[:20], sku_attrs)
            logger.info(f"📋 Длина промпта: {len(prompt)} символов")

            # Токенизация промпта
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

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

                # Находим этот код в результатах поиска для получения полной информации
                for candidate in best_candidates:
                    if candidate['result'].payload.get('ktru_code') == ktru_code:
                        logger.info(f"✅ LLM выбрал код: {ktru_code} с уверенностью {candidate['confidence']:.3f}")
                        return {
                            "ktru_code": ktru_code,
                            "ktru_title": candidate['result'].payload.get('title', None),
                            "confidence": candidate['confidence']
                        }

                # Если код не найден в кандидатах, ищем в базе
                ktru_title = self._find_ktru_title_by_code(ktru_code, search_result)
                return {
                    "ktru_code": ktru_code,
                    "ktru_title": ktru_title,
                    "confidence": 0.95  # Базовая уверенность для LLM
                }

            elif "код не найден" in response.lower():
                logger.info("❌ LLM не смог найти точное соответствие")
                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

            else:
                logger.warning(f"⚠️ Неопределенный ответ LLM: '{response}'")
                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

        except Exception as e:
            logger.error(f"❌ Ошибка при классификации SKU: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

    def _find_ktru_title_by_code(self, ktru_code: str, search_results: List) -> Optional[str]:
        """Поиск названия КТРУ по коду"""
        try:
            # Сначала ищем в результатах поиска
            for entry in search_results:
                payload = entry.payload
                if payload.get('ktru_code') == ktru_code:
                    return payload.get('title', '')

            # Если не найдено, ищем в базе
            if self.qdrant_client:
                try:
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=QDRANT_COLLECTION,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="ktru_code",
                                    match=MatchValue(value=ktru_code)
                                )
                            ]
                        ),
                        limit=1,
                        with_payload=True,
                        with_vectors=False
                    )

                    points, _ = scroll_result
                    if points:
                        return points[0].payload.get('title', '')
                except Exception as e:
                    logger.error(f"Ошибка при поиске в базе: {e}")

            return None

        except Exception as e:
            logger.error(f"Ошибка при поиске названия КТРУ: {e}")
            return None


# Создаем глобальный экземпляр классификатора
classifier = KtruClassifier()


def classify_sku(sku_data: Dict, top_k: int = TOP_K) -> Dict:
    """Функция-обертка для классификации SKU"""
    return classifier.classify_sku(sku_data, top_k)