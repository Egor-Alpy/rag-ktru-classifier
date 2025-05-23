"""
RAG KTRU Классификатор с векторной БД и LLM
Оптимизирован для RunPod с 24GB VRAM
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from rapidfuzz import fuzz, process
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import (
    LLM_MODEL, LLM_MAX_LENGTH, LLM_TEMPERATURE, LLM_TOP_P, LLM_TOP_K,
    SEARCH_TOP_K, RERANK_TOP_K, MIN_CONFIDENCE, WEIGHTS,
    CATEGORY_MAPPINGS, STOP_WORDS, DEVICE
)
from embeddings import embedding_manager
from vector_db import vector_db

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Результат классификации"""
    ktru_code: str
    ktru_title: str
    confidence: float
    method: str  # Метод, который дал результат
    details: Dict = None


class KTRUClassifier:
    """RAG классификатор KTRU с векторной БД и LLM"""

    def __init__(self, use_llm: bool = True):
        """
        Инициализация классификатора

        Args:
            use_llm: Использовать ли LLM для финальной классификации
        """
        self.use_llm = use_llm
        self.llm = None
        self.tokenizer = None

        # Паттерн для извлечения KTRU кода
        self.ktru_pattern = re.compile(r'\d{2}\.\d{2}\.\d{2}\.\d{3}-\d{8}')

        # Загружаем LLM если нужно
        if self.use_llm:
            self._load_llm()

        logger.info("✅ Классификатор инициализирован")

    def _load_llm(self):
        """Загрузка квантизированной LLM модели"""
        try:
            logger.info(f"Загрузка LLM модели: {LLM_MODEL}")

            # Конфигурация квантизации для экономии памяти
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Загружаем модель с квантизацией
            self.llm = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

            # Переводим в eval режим
            self.llm.eval()

            logger.info("✅ LLM модель загружена с 4-bit квантизацией")

        except Exception as e:
            logger.error(f"Ошибка загрузки LLM: {e}")
            logger.warning("Продолжаем без LLM")
            self.use_llm = False
            self.llm = None
            self.tokenizer = None

    def _extract_keywords(self, text: str) -> List[str]:
        """Извлечение ключевых слов из текста"""
        text_lower = text.lower()

        # Удаляем знаки препинания
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower)

        # Разбиваем на слова
        words = text_clean.split()

        # Фильтруем стоп-слова и короткие слова
        keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]

        # Добавляем биграммы для важных слов
        bigrams = []
        for i in range(len(words) - 1):
            if words[i] not in STOP_WORDS and words[i + 1] not in STOP_WORDS:
                bigrams.append(f"{words[i]} {words[i + 1]}")

        return keywords + bigrams

    def _calculate_keyword_score(self, product_text: str, ktru_data: Dict) -> float:
        """Расчет совпадения по ключевым словам"""
        product_keywords = set(self._extract_keywords(product_text))

        # Извлекаем ключевые слова из KTRU
        ktru_text = f"{ktru_data.get('title', '')} {ktru_data.get('description', '')}"
        ktru_keywords = set(self._extract_keywords(ktru_text))

        # Добавляем явные ключевые слова если есть
        if ktru_data.get('keywords'):
            ktru_keywords.update([kw.lower() for kw in ktru_data['keywords']])

        # Считаем пересечение
        if not ktru_keywords:
            return 0.0

        intersection = len(product_keywords & ktru_keywords)
        union = len(product_keywords | ktru_keywords)

        return intersection / union if union > 0 else 0.0

    def _calculate_fuzzy_score(self, product_title: str, ktru_title: str) -> float:
        """Расчет нечеткого совпадения названий"""
        # Нормализуем названия
        product_title = product_title.lower().strip()
        ktru_title = ktru_title.lower().strip()

        # Используем различные метрики
        ratio = fuzz.ratio(product_title, ktru_title) / 100
        partial_ratio = fuzz.partial_ratio(product_title, ktru_title) / 100
        token_sort = fuzz.token_sort_ratio(product_title, ktru_title) / 100

        # Комбинируем метрики
        return max(ratio, partial_ratio * 0.9, token_sort * 0.85)

    def _check_category_match(self, product_data: Dict, ktru_code: str) -> float:
        """Проверка соответствия категории"""
        product_text = f"{product_data.get('title', '')} {product_data.get('category', '')}".lower()

        # Проверяем известные категории
        for category_key, code_prefixes in CATEGORY_MAPPINGS.items():
            if category_key in product_text:
                for prefix in code_prefixes:
                    if ktru_code.startswith(prefix):
                        return 1.0

        return 0.0

    def _rerank_candidates(self, product_data: Dict, candidates: List[Dict]) -> List[Tuple[Dict, float]]:
        """Переранжирование кандидатов с учетом всех метрик"""
        product_text = embedding_manager.prepare_product_text(product_data)
        product_title = product_data.get('title', '')

        reranked = []

        for candidate in candidates:
            ktru_data = candidate['payload']

            # Векторное сходство (уже есть)
            vector_score = candidate['score']

            # Совпадение ключевых слов
            keyword_score = self._calculate_keyword_score(product_text, ktru_data)

            # Нечеткое совпадение названий
            fuzzy_score = self._calculate_fuzzy_score(product_title, ktru_data.get('title', ''))

            # Проверка категории
            category_score = self._check_category_match(product_data, ktru_data.get('ktru_code', ''))

            # Комбинированный скор
            final_score = (
                    vector_score * WEIGHTS['vector_similarity'] +
                    keyword_score * WEIGHTS['keyword_match'] +
                    fuzzy_score * WEIGHTS['fuzzy_match'] +
                    category_score * WEIGHTS['category_match']
            )

            # Бонус за точное совпадение слов в названии
            title_words = set(product_title.lower().split())
            ktru_title_words = set(ktru_data.get('title', '').lower().split())
            exact_matches = len(title_words & ktru_title_words)
            if exact_matches > 1:
                final_score *= (1 + 0.1 * exact_matches)

            reranked.append((ktru_data, final_score))

        # Сортируем по финальному скору
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:RERANK_TOP_K]

    def _classify_with_llm(self, product_data: Dict, candidates: List[Tuple[Dict, float]]) -> Optional[str]:
        """Классификация с помощью LLM"""
        if not self.llm or not candidates:
            return None

        try:
            # Формируем промпт
            prompt = self._create_classification_prompt(product_data, candidates)

            # Токенизация
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=LLM_MAX_LENGTH
            ).to(self.llm.device)

            # Генерация
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=LLM_TEMPERATURE,
                    top_p=LLM_TOP_P,
                    top_k=LLM_TOP_K,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Декодирование
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            # Извлекаем KTRU код из ответа
            match = self.ktru_pattern.search(response)
            if match:
                suggested_code = match.group(0)

                # Проверяем, что код есть среди кандидатов
                for ktru_data, _ in candidates:
                    if ktru_data.get('ktru_code') == suggested_code:
                        return suggested_code

            return None

        except Exception as e:
            logger.error(f"Ошибка при классификации с LLM: {e}")
            return None

    def _create_classification_prompt(self, product_data: Dict, candidates: List[Tuple[Dict, float]]) -> str:
        """Создание промпта для LLM"""
        prompt = f"""Задача: Определить точный код КТРУ для товара.

ТОВАР:
Название: {product_data.get('title', '')}
Описание: {product_data.get('description', '')}
Категория: {product_data.get('category', '')}

КАНДИДАТЫ КТРУ (отсортированы по релевантности):
"""

        for i, (ktru_data, score) in enumerate(candidates[:5], 1):
            prompt += f"\n{i}. Код: {ktru_data.get('ktru_code', '')}"
            prompt += f"\n   Название: {ktru_data.get('title', '')}"
            prompt += f"\n   Релевантность: {score:.2f}"

        prompt += "\n\nВыбери НАИБОЛЕЕ ПОДХОДЯЩИЙ код КТРУ из списка выше. Ответь только кодом:"

        return prompt

    def classify(self, product_data: Dict) -> ClassificationResult:
        """
        Основной метод классификации товара

        Args:
            product_data: Словарь с данными товара

        Returns:
            ClassificationResult с результатом классификации
        """
        logger.info(f"Классификация товара: {product_data.get('title', 'Без названия')}")

        # Подготовка текста для поиска
        search_text = embedding_manager.prepare_product_text(product_data)

        # Векторный поиск
        logger.debug("Выполняем векторный поиск...")
        search_results = vector_db.search(search_text, top_k=SEARCH_TOP_K)

        if not search_results:
            logger.warning("Векторный поиск не дал результатов")
            return ClassificationResult(
                ktru_code="код не найден",
                ktru_title="",
                confidence=0.0,
                method="no_results"
            )

        # Переранжирование кандидатов
        logger.debug("Переранжирование кандидатов...")
        reranked_candidates = self._rerank_candidates(product_data, search_results)

        # Логируем топ кандидатов
        logger.info("Топ-5 кандидатов после переранжирования:")
        for i, (ktru_data, score) in enumerate(reranked_candidates[:5], 1):
            logger.info(f"  {i}. {ktru_data.get('ktru_code')} | {score:.3f} | {ktru_data.get('title', '')[:50]}...")

        # Проверяем лучший результат
        best_ktru, best_score = reranked_candidates[0]

        # Если скор очень высокий, возвращаем сразу
        if best_score >= 0.9:
            return ClassificationResult(
                ktru_code=best_ktru.get('ktru_code', ''),
                ktru_title=best_ktru.get('title', ''),
                confidence=min(best_score, 0.99),
                method="high_confidence",
                details={'score': best_score}
            )

        # Если скор средний и есть LLM, используем её для уточнения
        if self.use_llm and self.llm and best_score >= 0.6:
            logger.debug("Используем LLM для уточнения...")
            llm_code = self._classify_with_llm(product_data, reranked_candidates)

            if llm_code:
                # Находим данные для кода от LLM
                for ktru_data, score in reranked_candidates:
                    if ktru_data.get('ktru_code') == llm_code:
                        return ClassificationResult(
                            ktru_code=llm_code,
                            ktru_title=ktru_data.get('title', ''),
                            confidence=min(score * 1.1, 0.95),  # Boost за подтверждение LLM
                            method="llm_confirmed",
                            details={'original_score': score, 'llm_boost': True}
                        )

        # Возвращаем лучший результат если он выше порога
        if best_score >= MIN_CONFIDENCE:
            return ClassificationResult(
                ktru_code=best_ktru.get('ktru_code', ''),
                ktru_title=best_ktru.get('title', ''),
                confidence=best_score,
                method="threshold_passed",
                details={'score': best_score}
            )

        # Не найдено подходящего кода
        return ClassificationResult(
            ktru_code="код не найден",
            ktru_title="",
            confidence=best_score,
            method="below_threshold",
            details={'best_score': best_score, 'best_code': best_ktru.get('ktru_code', '')}
        )


# Глобальный экземпляр классификатора
classifier = KTRUClassifier(use_llm=True)


def classify_product(product_data: Dict) -> Dict:
    """Функция-обертка для обратной совместимости"""
    result = classifier.classify(product_data)

    return {
        'ktru_code': result.ktru_code,
        'ktru_title': result.ktru_title,
        'confidence': result.confidence
    }