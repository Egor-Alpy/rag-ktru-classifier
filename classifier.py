import re
import torch
import json
import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
from difflib import SequenceMatcher
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig, LlamaTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
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


class UtilityKtruClassifier:
    def __init__(self):
        """Инициализация утилитарного классификатора КТРУ"""
        self.qdrant_client = self._setup_qdrant()
        self.llm, self.tokenizer = self._setup_llm()

        # Компилируем шаблон для поиска КТРУ кода
        self.ktru_pattern = re.compile(r'\d{2}\.\d{2}\.\d{2}\.\d{3}-\d{8}')

        # Создаем индекс ключевых слов для быстрого поиска
        self.keyword_index = self._build_keyword_index()

        # Карта категорий товаров и их типичных кодов
        self.category_patterns = {
            # Компьютерная техника
            'компьютер': {'codes': ['26.20.11', '26.20.13', '26.20.14'], 'weight': 1.0},
            'ноутбук': {'codes': ['26.20.11', '26.20.13'], 'weight': 1.0},
            'laptop': {'codes': ['26.20.11', '26.20.13'], 'weight': 1.0},
            'пк': {'codes': ['26.20.11', '26.20.13'], 'weight': 0.9},
            'системный блок': {'codes': ['26.20.11'], 'weight': 1.0},
            'монитор': {'codes': ['26.20.17'], 'weight': 1.0},
            'клавиатура': {'codes': ['26.20.16'], 'weight': 1.0},
            'мышь': {'codes': ['26.20.16'], 'weight': 1.0},
            'принтер': {'codes': ['26.20.16', '30.20'], 'weight': 1.0},
            'сканер': {'codes': ['26.20.16'], 'weight': 1.0},
            'мфу': {'codes': ['26.20.16'], 'weight': 1.0},

            # Канцелярские товары
            'ручка': {'codes': ['32.99.12', '32.99.13'], 'weight': 1.0},
            'карандаш': {'codes': ['32.99.15'], 'weight': 1.0},
            'маркер': {'codes': ['32.99.12'], 'weight': 1.0},
            'фломастер': {'codes': ['32.99.12'], 'weight': 1.0},
            'ластик': {'codes': ['22.19.71'], 'weight': 1.0},
            'линейка': {'codes': ['32.99.15'], 'weight': 1.0},
            'тетрадь': {'codes': ['17.23.13'], 'weight': 1.0},
            'блокнот': {'codes': ['17.23.13'], 'weight': 1.0},
            'степлер': {'codes': ['25.99.23'], 'weight': 1.0},
            'скрепки': {'codes': ['25.93.18'], 'weight': 1.0},
            'скотч': {'codes': ['22.21.21'], 'weight': 1.0},
            'клей': {'codes': ['20.52'], 'weight': 1.0},

            # Бумажная продукция
            'бумага': {'codes': ['17.12.14', '17.23.12'], 'weight': 1.0},
            'туалетная бумага': {'codes': ['17.22.12'], 'weight': 1.0},
            'салфетки': {'codes': ['17.22.13'], 'weight': 1.0},
            'полотенца бумажные': {'codes': ['17.22.13'], 'weight': 1.0},
            'картон': {'codes': ['17.12.42'], 'weight': 1.0},

            # Мебель
            'стол': {'codes': ['31.01.11', '31.09.11'], 'weight': 1.0},
            'стул': {'codes': ['31.01.11', '31.09.12'], 'weight': 1.0},
            'кресло': {'codes': ['31.01.12', '31.09.12'], 'weight': 1.0},
            'шкаф': {'codes': ['31.01.13', '31.09.13'], 'weight': 1.0},
            'тумба': {'codes': ['31.09.13'], 'weight': 1.0},
            'полка': {'codes': ['31.09.13'], 'weight': 1.0},
            'диван': {'codes': ['31.09.11'], 'weight': 1.0},

            # Медицинские товары
            'шприц': {'codes': ['32.50.13'], 'weight': 1.0},
            'бинт': {'codes': ['21.20.24'], 'weight': 1.0},
            'маска': {'codes': ['32.50.22', '14.12.30'], 'weight': 1.0},
            'перчатки': {'codes': ['22.19.60', '15.20.32'], 'weight': 1.0},
            'термометр': {'codes': ['26.51.53'], 'weight': 1.0},
            'тонометр': {'codes': ['26.60.12'], 'weight': 1.0},

            # Продукты питания
            'молоко': {'codes': ['10.51.11', '10.51.12'], 'weight': 1.0},
            'хлеб': {'codes': ['10.71.11'], 'weight': 1.0},
            'мясо': {'codes': ['10.11', '10.13'], 'weight': 1.0},
            'рыба': {'codes': ['10.20'], 'weight': 1.0},
            'овощи': {'codes': ['01.13'], 'weight': 1.0},
            'фрукты': {'codes': ['01.24', '01.25'], 'weight': 1.0},
        }

        # Словарь синонимов
        self.synonyms = {
            'ноутбук': ['лэптоп', 'портативный компьютер', 'ноут', 'notebook', 'laptop'],
            'компьютер': ['пк', 'персональный компьютер', 'комп', 'системный блок', 'десктоп'],
            'ручка': ['авторучка', 'шариковая ручка', 'гелевая ручка', 'ручка для письма'],
            'бумага': ['листы', 'бумажные листы', 'офисная бумага', 'писчая бумага'],
            'стол': ['письменный стол', 'рабочий стол', 'офисный стол', 'парта'],
            'принтер': ['печатающее устройство', 'лазерный принтер', 'струйный принтер'],
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
            logger.info(f"Загрузка LLM для финальной классификации...")
            tokenizer = AutoTokenizer.from_pretrained(LLM_BASE_MODEL, trust_remote_code=True)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Упрощенная загрузка модели
            try:
                model = AutoPeftModelForCausalLM.from_pretrained(
                    LLM_ADAPTER_MODEL,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                logger.info("✅ LLM загружена")
            except:
                logger.warning("⚠️ LLM недоступна, будет использоваться только rule-based подход")
                return None, None

            return model, tokenizer
        except Exception as e:
            logger.error(f"Ошибка загрузки LLM: {e}")
            return None, None

    def _build_keyword_index(self):
        """Строим индекс ключевых слов из базы КТРУ"""
        keyword_index = defaultdict(list)

        if not self.qdrant_client:
            return keyword_index

        try:
            # Получаем все записи для построения индекса
            offset = None
            batch_size = 100

            while True:
                records, offset = self.qdrant_client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                if not records:
                    break

                for record in records:
                    payload = record.payload
                    ktru_code = payload.get('ktru_code', '')
                    title = payload.get('title', '').lower()

                    # Извлекаем ключевые слова из названия
                    words = re.findall(r'\b[а-яё]+\b', title, re.IGNORECASE)
                    for word in words:
                        if len(word) > 2:  # Игнорируем короткие слова
                            keyword_index[word].append({
                                'code': ktru_code,
                                'title': payload.get('title', ''),
                                'full_data': payload
                            })

                if offset is None:
                    break

            logger.info(f"✅ Построен индекс из {len(keyword_index)} ключевых слов")

        except Exception as e:
            logger.error(f"Ошибка при построении индекса: {e}")

        return keyword_index

    def _extract_keywords(self, text):
        """Извлечение ключевых слов из текста"""
        text_lower = text.lower()

        # Ищем важные слова для классификации
        keywords = []

        # Проверяем паттерны категорий
        for pattern, info in self.category_patterns.items():
            if pattern in text_lower:
                keywords.append((pattern, info['weight']))

        # Проверяем синонимы
        for main_word, synonyms in self.synonyms.items():
            if main_word in text_lower:
                keywords.append((main_word, 1.0))
            for synonym in synonyms:
                if synonym in text_lower:
                    keywords.append((main_word, 0.9))  # Синонимы имеют чуть меньший вес

        # Извлекаем все существенные слова
        words = re.findall(r'\b[а-яёa-z]{3,}\b', text_lower, re.IGNORECASE)
        for word in words:
            if word not in [kw[0] for kw in keywords]:
                keywords.append((word, 0.5))

        return keywords

    def _search_by_keywords(self, keywords, limit=20):
        """Поиск КТРУ по ключевым словам"""
        candidates = defaultdict(float)

        for keyword, weight in keywords:
            # Ищем в индексе ключевых слов
            if keyword in self.keyword_index:
                for item in self.keyword_index[keyword]:
                    candidates[item['code']] += weight

            # Ищем по паттернам категорий
            if keyword in self.category_patterns:
                pattern_info = self.category_patterns[keyword]
                for code_prefix in pattern_info['codes']:
                    # Ищем все коды, начинающиеся с этого префикса
                    for kw_items in self.keyword_index.values():
                        for item in kw_items:
                            if item['code'].startswith(code_prefix):
                                candidates[item['code']] += pattern_info[
                                                                'weight'] * 2  # Удваиваем вес для точных категорий

        # Сортируем по весу
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        # Получаем полную информацию о топ кандидатах
        result = []
        for code, score in sorted_candidates[:limit]:
            # Находим полную информацию о коде
            for kw_items in self.keyword_index.values():
                for item in kw_items:
                    if item['code'] == code:
                        result.append({
                            'code': code,
                            'score': score,
                            'data': item['full_data']
                        })
                        break
                if len(result) > len([r for r in result if r['code'] != code]):
                    break

        return result

    def _calculate_title_similarity(self, sku_title, ktru_title):
        """Расчет схожести названий"""
        sku_words = set(re.findall(r'\b[а-яёa-z]+\b', sku_title.lower(), re.IGNORECASE))
        ktru_words = set(re.findall(r'\b[а-яёa-z]+\b', ktru_title.lower(), re.IGNORECASE))

        if not sku_words or not ktru_words:
            return 0.0

        # Прямое совпадение слов
        common_words = sku_words.intersection(ktru_words)
        word_similarity = len(common_words) / min(len(sku_words), len(ktru_words))

        # Проверка последовательности символов
        sequence_similarity = SequenceMatcher(None, sku_title.lower(), ktru_title.lower()).ratio()

        # Комбинированная оценка
        return word_similarity * 0.7 + sequence_similarity * 0.3

    def _hybrid_search(self, sku_data, top_k=50):
        """Гибридный поиск: ключевые слова + векторный поиск"""
        title = sku_data.get('title', '')
        description = sku_data.get('description', '')

        # Извлекаем ключевые слова
        keywords = self._extract_keywords(f"{title} {description}")

        # Поиск по ключевым словам
        keyword_results = self._search_by_keywords(keywords, limit=30)

        # Векторный поиск для дополнения результатов
        search_text = f"{title} {description}"
        if sku_data.get('category'):
            search_text += f" категория: {sku_data['category']}"
        if sku_data.get('brand'):
            search_text += f" бренд: {sku_data['brand']}"

        embedding = generate_embedding(search_text)

        vector_results = []
        if self.qdrant_client:
            try:
                search_result = self.qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=embedding.tolist(),
                    limit=top_k
                )

                for result in search_result:
                    vector_results.append({
                        'code': result.payload.get('ktru_code', ''),
                        'score': getattr(result, 'score', 0),
                        'data': result.payload
                    })
            except Exception as e:
                logger.error(f"Ошибка векторного поиска: {e}")

        # Объединяем результаты
        all_results = {}

        # Сначала добавляем результаты по ключевым словам (они приоритетнее)
        for item in keyword_results:
            code = item['code']
            all_results[code] = {
                'code': code,
                'keyword_score': item['score'],
                'vector_score': 0,
                'data': item['data']
            }

        # Добавляем векторные результаты
        for item in vector_results:
            code = item['code']
            if code in all_results:
                all_results[code]['vector_score'] = item['score']
            else:
                all_results[code] = {
                    'code': code,
                    'keyword_score': 0,
                    'vector_score': item['score'],
                    'data': item['data']
                }

        # Вычисляем финальный скор
        final_results = []
        for code, scores in all_results.items():
            # Приоритет отдаем поиску по ключевым словам
            final_score = scores['keyword_score'] * 0.7 + scores['vector_score'] * 0.3

            # Бонус за совпадение названий
            ktru_title = scores['data'].get('title', '')
            title_similarity = self._calculate_title_similarity(title, ktru_title)
            final_score += title_similarity * 0.5

            final_results.append({
                'code': code,
                'score': final_score,
                'data': scores['data'],
                'keyword_score': scores['keyword_score'],
                'vector_score': scores['vector_score'],
                'title_similarity': title_similarity
            })

        # Сортируем по финальному скору
        final_results.sort(key=lambda x: x['score'], reverse=True)

        return final_results[:top_k]

    def classify_sku(self, sku_data, top_k=TOP_K):
        """Классификация SKU по КТРУ коду"""
        logger.info(f"🚀 Классификация: {sku_data.get('title', 'Без названия')}")

        if not self.qdrant_client:
            logger.error("❌ Qdrant клиент не инициализирован")
            return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

        try:
            # Гибридный поиск
            search_results = self._hybrid_search(sku_data, top_k=top_k)

            if not search_results:
                logger.warning("⚠️ Не найдено подходящих КТРУ кодов")
                return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

            # Логируем топ результаты
            logger.info(f"📊 Топ-5 кандидатов:")
            for i, result in enumerate(search_results[:5]):
                logger.info(f"   {i + 1}. {result['code']} | Скор: {result['score']:.3f} | "
                            f"KW: {result['keyword_score']:.2f} | Vec: {result['vector_score']:.2f} | "
                            f"Title: {result['title_similarity']:.2f} | {result['data'].get('title', '')[:50]}...")

            # Проверяем топ результат
            best_match = search_results[0]

            # Если есть явный победитель по ключевым словам
            if best_match['keyword_score'] > 2.0 and best_match['title_similarity'] > 0.5:
                confidence = min(0.98, 0.7 + best_match['title_similarity'] * 0.3)
                logger.info(f"✅ Найдено совпадение по ключевым словам с уверенностью {confidence:.3f}")
                return {
                    "ktru_code": best_match['code'],
                    "ktru_title": best_match['data'].get('title', None),
                    "confidence": confidence
                }

            # Если есть хорошее совпадение по названию
            if best_match['title_similarity'] > 0.7:
                confidence = min(0.95, 0.6 + best_match['title_similarity'] * 0.35)
                logger.info(f"✅ Найдено совпадение по названию с уверенностью {confidence:.3f}")
                return {
                    "ktru_code": best_match['code'],
                    "ktru_title": best_match['data'].get('title', None),
                    "confidence": confidence
                }

            # Если нужна дополнительная проверка с LLM
            if self.llm and self.tokenizer and len(search_results) > 1:
                # Используем LLM только для неоднозначных случаев
                logger.info("🤖 Используем LLM для уточнения...")

                prompt = self._create_simple_prompt(sku_data, search_results[:5])

                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                    inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

                    generation_config = GenerationConfig(
                        temperature=0.1,
                        top_p=0.9,
                        max_new_tokens=50,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                    with torch.no_grad():
                        generated_ids = self.llm.generate(**inputs, generation_config=generation_config)

                    response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    response = response[len(prompt):].strip()

                    logger.info(f"🤖 Ответ LLM: '{response}'")

                    # Проверяем ответ
                    ktru_match = self.ktru_pattern.search(response)
                    if ktru_match:
                        ktru_code = ktru_match.group(0)
                        # Проверяем, что код есть в наших кандидатах
                        for result in search_results[:5]:
                            if result['code'] == ktru_code:
                                return {
                                    "ktru_code": ktru_code,
                                    "ktru_title": result['data'].get('title', None),
                                    "confidence": 0.90
                                }
                except Exception as e:
                    logger.error(f"Ошибка при использовании LLM: {e}")

            # Если уверенность недостаточна, но есть результат
            if best_match['score'] > 1.0:
                confidence = min(0.85, 0.5 + best_match['score'] * 0.1)
                logger.info(f"⚠️ Найдено возможное совпадение с уверенностью {confidence:.3f}")
                return {
                    "ktru_code": best_match['code'],
                    "ktru_title": best_match['data'].get('title', None),
                    "confidence": confidence
                }

            # Если ничего не подходит
            logger.info("❌ Не найдено подходящего КТРУ кода")
            return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

        except Exception as e:
            logger.error(f"❌ Ошибка при классификации: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"ktru_code": "код не найден", "ktru_title": None, "confidence": 0.0}

    def _create_simple_prompt(self, sku_data, candidates):
        """Создание упрощенного промпта для LLM"""
        prompt = f"""Определи КТРУ код для товара.

ТОВАР: {sku_data.get('title', '')}
{f"Описание: {sku_data.get('description', '')}" if sku_data.get('description') else ""}

КАНДИДАТЫ КТРУ:
"""
        for i, candidate in enumerate(candidates, 1):
            prompt += f"{i}. {candidate['code']} - {candidate['data'].get('title', '')}\n"

        prompt += "\nВыбери НАИБОЛЕЕ ПОДХОДЯЩИЙ код из списка выше. Ответь только кодом:"

        return prompt


# Создаем глобальный экземпляр классификатора
classifier = UtilityKtruClassifier()


def classify_sku(sku_data: Dict, top_k: int = TOP_K) -> Dict:
    """Функция-обертка для классификации SKU"""
    return classifier.classify_sku(sku_data, top_k)