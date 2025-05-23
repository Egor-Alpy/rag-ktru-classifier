import re
import torch
import json
import logging
from typing import List, Dict, Optional, Tuple, Set
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
            # Общие атрибуты
            "количество слоев": ["слойность", "число слоев", "слои", "слойный"],
            "цвет": ["окраска", "расцветка", "оттенок", "цвет бумаги"],
            "тип": ["вид", "разновидность", "категория"],
            "материал": ["состав", "сырье", "основа"],
            "размер": ["габарит", "величина", "формат", "размеры"],
            "назначение": ["применение", "использование", "цель", "для"],

            # Специфичные атрибуты
            "объем": ["объём", "емкость", "вместимость", "объём реагента"],
            "процессор": ["cpu", "чип", "микропроцессор"],
            "память": ["ram", "озу", "оперативная память"],
            "торговая марка": ["бренд", "производитель", "марка", "brand"],

            # Единицы измерения
            "штука": ["шт", "единица", "ед"],
            "набор": ["комплект", "set", "kit"],
            "упаковка": ["пачка", "пакет", "уп"],
        }

        # Важные слова для категорий товаров
        self.category_keywords = {
            "компьютер": ["ноутбук", "laptop", "notebook", "портативный компьютер", "пк", "pc"],
            "бумага": ["туалетная", "офисная", "писчая", "копировальная"],
            "реагент": ["ивд", "диагностика", "лабораторный", "анализатор"],
            "канцелярия": ["ручка", "карандаш", "маркер", "фломастер"],
            "мебель": ["стол", "стул", "шкаф", "тумба", "кресло"],
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

    def _extract_attributes(self, data: Dict) -> Dict[str, Set[str]]:
        """Извлечение и нормализация атрибутов с поддержкой множественных значений"""
        attributes = {}

        if 'attributes' in data and isinstance(data['attributes'], list):
            for attr in data['attributes']:
                if isinstance(attr, dict):
                    # Для SKU формат
                    if 'attr_name' in attr and 'attr_value' in attr:
                        name = self._normalize_attribute_name(attr['attr_name'])
                        value = str(attr['attr_value']).lower().strip()
                        if name not in attributes:
                            attributes[name] = set()
                        attributes[name].add(value)

                    # Для KTRU формат
                    elif 'attr_name' in attr and 'attr_values' in attr:
                        name = self._normalize_attribute_name(attr['attr_name'])
                        if name not in attributes:
                            attributes[name] = set()

                        for val in attr['attr_values']:
                            if isinstance(val, dict) and 'value' in val:
                                value = str(val['value']).lower().strip()
                                # Добавляем единицу измерения если есть
                                if 'value_unit' in val and val['value_unit']:
                                    unit = val['value_unit'].strip()
                                    if unit:
                                        value = f"{value} {unit}"
                                attributes[name].add(value)

        # Конвертируем sets в строки для удобства
        result = {}
        for name, values in attributes.items():
            result[name] = '; '.join(sorted(values))

        return result

    def _calculate_attribute_similarity(self, sku_attrs: Dict[str, str], ktru_attrs: Dict[str, str]) -> float:
        """Улучшенный расчет схожести атрибутов"""
        if not sku_attrs or not ktru_attrs:
            return 0.0

        matches = 0
        total_comparisons = 0
        critical_matches = 0  # Для критически важных атрибутов

        # Определяем критические атрибуты для разных категорий
        critical_attributes = {
            "тип", "количество слоев", "материал", "процессор", "память",
            "объем", "назначение", "размер"
        }

        # Проверяем совпадение атрибутов
        for sku_attr_name, sku_attr_value in sku_attrs.items():
            if sku_attr_name in ktru_attrs:
                ktru_value = ktru_attrs[sku_attr_name]
                total_comparisons += 1
                is_critical = sku_attr_name in critical_attributes

                # Разбиваем значения на компоненты для сравнения
                sku_values = set(v.strip() for v in sku_attr_value.split(';'))
                ktru_values = set(v.strip() for v in ktru_value.split(';'))

                # Точное совпадение хотя бы одного значения
                if sku_values.intersection(ktru_values):
                    matches += 1
                    if is_critical:
                        critical_matches += 1
                # Частичное совпадение
                else:
                    partial_match = False
                    for sv in sku_values:
                        for kv in ktru_values:
                            # Проверка вхождения
                            if sv in kv or kv in sv:
                                partial_match = True
                                break
                            # Проверка числовых диапазонов
                            if self._check_numeric_match(sv, kv):
                                partial_match = True
                                break
                        if partial_match:
                            break

                    if partial_match:
                        matches += 0.7 if is_critical else 0.5

        if total_comparisons == 0:
            return 0.0

        # Базовая схожесть
        base_similarity = matches / total_comparisons

        # Бонус за совпадение критических атрибутов
        critical_bonus = critical_matches * 0.1

        return min(1.0, base_similarity + critical_bonus)

    def _check_numeric_match(self, value1: str, value2: str) -> bool:
        """Проверка совпадения числовых значений с учетом диапазонов"""
        try:
            # Извлекаем числа из строк
            nums1 = re.findall(r'[\d.]+', value1)
            nums2 = re.findall(r'[\d.]+', value2)

            if not nums1 or not nums2:
                return False

            # Проверяем диапазоны (≥, ≤, и т.д.)
            if '≥' in value2 or '>=' in value2:
                return float(nums1[0]) >= float(nums2[0])
            elif '≤' in value2 or '<=' in value2:
                return float(nums1[0]) <= float(nums2[0])
            else:
                # Точное совпадение с допуском 10%
                num1 = float(nums1[0])
                num2 = float(nums2[0])
                return abs(num1 - num2) / max(num1, num2) < 0.1
        except:
            return False

    def _create_search_text(self, sku_data: Dict, sku_attrs: Dict[str, str]) -> str:
        """Создание оптимизированного текста для векторного поиска"""
        text_parts = []

        # Основная информация
        if sku_data.get('title'):
            text_parts.append(sku_data['title'])

        if sku_data.get('category'):
            text_parts.append(f"категория: {sku_data['category']}")

        if sku_data.get('brand'):
            text_parts.append(f"бренд: {sku_data['brand']}")

        if sku_data.get('description'):
            text_parts.append(sku_data['description'])

        # Добавляем все атрибуты в структурированном виде
        for attr_name, attr_value in sku_attrs.items():
            text_parts.append(f"{attr_name}: {attr_value}")

        # Добавляем ключевые слова категории
        text_lower = ' '.join(text_parts).lower()
        for category, keywords in self.category_keywords.items():
            if any(kw in text_lower for kw in keywords):
                text_parts.append(f"категория товара: {category}")

        return ' '.join(text_parts)

    def _create_classification_prompt(self, sku_data: Dict, similar_ktru_entries: List,
                                      sku_attrs: Dict[str, str]) -> str:
        """Создание улучшенного промпта для точной классификации"""
        prompt = """Ты эксперт по классификации товаров в системе КТРУ. Твоя задача - найти ЕДИНСТВЕННЫЙ ТОЧНЫЙ код КТРУ.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Код должен ТОЧНО соответствовать товару по ВСЕМ характеристикам
2. Проверь соответствие: тип товара, технические параметры, назначение, материалы
3. НЕ выбирай похожий код - только ТОЧНОЕ соответствие
4. Если есть сомнения - ответь "код не найден"

ТОВАР ДЛЯ КЛАССИФИКАЦИИ:
"""
        # Информация о товаре
        prompt += f"Название: {sku_data.get('title', '')}\n"
        if sku_data.get('description'):
            prompt += f"Описание: {sku_data['description']}\n"
        if sku_data.get('category'):
            prompt += f"Категория: {sku_data['category']}\n"
        if sku_data.get('brand'):
            prompt += f"Бренд: {sku_data['brand']}\n"

        # Атрибуты товара
        if sku_attrs:
            prompt += "\nХАРАКТЕРИСТИКИ ТОВАРА:\n"
            for attr_name, attr_value in sorted(sku_attrs.items()):
                prompt += f"• {attr_name}: {attr_value}\n"

        prompt += "\nКАНДИДАТЫ КТРУ (отсортированы по релевантности):\n"

        # Добавляем топ-5 кандидатов с детальным анализом
        for i, entry in enumerate(similar_ktru_entries[:5], 1):
            payload = entry.payload
            score = getattr(entry, 'score', 0)
            ktru_attrs = self._extract_attributes(payload)
            attr_similarity = self._calculate_attribute_similarity(sku_attrs, ktru_attrs)

            prompt += f"\n{i}. Код: {payload.get('ktru_code', '')}\n"
            prompt += f"   Название: {payload.get('title', '')}\n"
            prompt += f"   Схожесть: текст={score:.3f}, атрибуты={attr_similarity:.2f}\n"

            if ktru_attrs:
                prompt += "   Характеристики КТРУ:\n"
                for attr_name, attr_value in sorted(ktru_attrs.items())[:5]:
                    prompt += f"   • {attr_name}: {attr_value}\n"

        prompt += """
АЛГОРИТМ ВЫБОРА:
1. Найди кандидата где ВСЕ ключевые характеристики совпадают
2. Проверь что категория товара соответствует
3. Убедись что технические параметры идентичны
4. Если полного соответствия нет - код не найден

ОТВЕТ (только код или "код не найден"):"""

        return prompt

    def _validate_ktru_match(self, sku_data: Dict, ktru_data: Dict,
                             text_similarity: float, attr_similarity: float) -> Tuple[bool, float]:
        """Строгая валидация соответствия SKU и KTRU"""
        confidence = 0.0

        # Требуем высокую текстовую схожесть
        if text_similarity >= 0.95:
            confidence += 0.6
        elif text_similarity >= 0.90:
            confidence += 0.4
        elif text_similarity >= 0.85:
            confidence += 0.2
        else:
            return False, 0.0

        # Атрибуты должны хорошо совпадать
        if attr_similarity >= 0.8:
            confidence += 0.4
        elif attr_similarity >= 0.6:
            confidence += 0.2
        else:
            confidence *= 0.5  # Снижаем общую уверенность

        # Проверка ключевых слов категории
        sku_text = f"{sku_data.get('title', '')} {sku_data.get('description', '')}".lower()
        ktru_text = f"{ktru_data.get('title', '')} {ktru_data.get('description', '')}".lower()

        # Ищем общие значимые слова (длиннее 3 символов)
        sku_words = set(w for w in sku_text.split() if len(w) > 3)
        ktru_words = set(w for w in ktru_text.split() if len(w) > 3)

        if sku_words and ktru_words:
            word_overlap = len(sku_words.intersection(ktru_words)) / min(len(sku_words), len(ktru_words))
            if word_overlap < 0.3:
                confidence *= 0.6

        # Проверка категории
        if 'category' in sku_data and sku_data['category']:
            category_found = False
            for cat_key, keywords in self.category_keywords.items():
                if any(kw in sku_text for kw in keywords):
                    if any(kw in ktru_text for kw in keywords):
                        category_found = True
                        break

            if not category_found:
                confidence *= 0.7

        # Требуем очень высокую уверенность
        is_valid = confidence >= 0.95

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

            # Создаем оптимизированный текст для поиска
            sku_text = self._create_search_text(sku_data, sku_attrs)
            logger.info(f"📝 Текст для поиска (первые 200 символов): {sku_text[:200]}...")

            # Генерируем эмбеддинг
            sku_embedding = generate_embedding(sku_text)
            logger.info(f"🔢 Размерность эмбеддинга: {len(sku_embedding)}")

            # Поиск похожих КТРУ кодов
            search_result = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=sku_embedding.tolist(),
                limit=top_k  # Берем больше кандидатов для анализа
            )

            if not search_result:
                logger.warning("⚠️ Не найдено похожих КТРУ записей")
                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

            # Анализ результатов с более строгими критериями
            best_candidates = []

            for result in search_result:
                score = getattr(result, 'score', 0)
                payload = result.payload

                # Пропускаем результаты с низкой схожестью
                if score < 0.8:
                    continue

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

            # Если есть кандидат с очень высокой уверенностью (>=0.98)
            if best_candidates and best_candidates[0]['confidence'] >= 0.98:
                best = best_candidates[0]
                payload = best['result'].payload
                logger.info(f"✅ Найдено точное совпадение с уверенностью {best['confidence']:.3f}")
                return {
                    "ktru_code": payload.get('ktru_code', 'код не найден'),
                    "ktru_title": payload.get('title', None),
                    "confidence": best['confidence']
                }

            # Используем LLM для финального выбора среди кандидатов
            if self.llm and self.tokenizer and len(search_result) > 0:
                # Берем только топ результаты для LLM
                top_results = search_result[:10]

                prompt = self._create_classification_prompt(sku_data, top_results, sku_attrs)
                logger.info(f"📋 Используем LLM для финальной классификации")

                # Токенизация промпта
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

                # Настройка параметров генерации для максимальной точности
                generation_config = GenerationConfig(
                    temperature=0.1,  # Очень низкая температура для детерминированности
                    top_p=0.9,
                    repetition_penalty=1.15,
                    max_new_tokens=50,
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
                response = response[len(prompt):].strip()

                logger.info(f"🤖 Ответ LLM: '{response}'")

                # Проверка ответа
                ktru_match = self.ktru_pattern.search(response)

                if ktru_match:
                    ktru_code = ktru_match.group(0)

                    # Находим информацию о выбранном коде
                    for candidate in best_candidates:
                        if candidate['result'].payload.get('ktru_code') == ktru_code:
                            return {
                                "ktru_code": ktru_code,
                                "ktru_title": candidate['result'].payload.get('title', None),
                                "confidence": candidate['confidence']
                            }

                    # Если код не в кандидатах, ищем в результатах поиска
                    for result in search_result:
                        if result.payload.get('ktru_code') == ktru_code:
                            return {
                                "ktru_code": ktru_code,
                                "ktru_title": result.payload.get('title', None),
                                "confidence": 0.95
                            }

            # Если ничего не найдено с высокой уверенностью
            logger.info("❌ Не найдено точное соответствие КТРУ")
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