#!/usr/bin/env python3
"""
Скрипт для диагностики проблем с классификацией КТРУ
"""

import sys
import os
import requests
import json
import numpy as np
from qdrant_client import QdrantClient

# Переходим в директорию проекта, если нужно
project_dir = "/workspace/rag-ktru-classifier"
if os.path.exists(project_dir) and os.getcwd() != project_dir:
    os.chdir(project_dir)
    print(f"Переход в директорию проекта: {project_dir}")

from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, API_HOST, API_PORT
from embedding import generate_embedding, test_embeddings


def check_qdrant_data():
    """Проверка данных в Qdrant"""
    print("\n🔍 1. ДИАГНОСТИКА ДАННЫХ В QDRANT")
    print("=" * 50)

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Получаем статистику коллекции
        collection_info = client.get_collection(QDRANT_COLLECTION)
        count_info = client.count(QDRANT_COLLECTION)

        print(f"✅ Коллекция: {QDRANT_COLLECTION}")
        print(f"📊 Количество векторов: {count_info.count:,}")
        print(f"📏 Размерность векторов: {collection_info.config.params.vectors.size}")
        print(f"📐 Метрика расстояния: {collection_info.config.params.vectors.distance.name}")

        # Получаем примеры записей
        examples = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=5,
            with_payload=True,
            with_vectors=False
        )

        print(f"\n📋 Примеры записей:")
        for i, point in enumerate(examples[0], 1):
            payload = point.payload
            print(f"  {i}. {payload.get('ktru_code', 'N/A')} - {payload.get('title', 'N/A')[:60]}...")

        # Поиск записей с компьютерной техникой
        print(f"\n🔍 Поиск записей с компьютерной техникой:")
        computer_examples = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter={
                "should": [
                    {"key": "title", "match": {"text": "компьютер"}},
                    {"key": "title", "match": {"text": "ноутбук"}},
                    {"key": "title", "match": {"text": "laptop"}},
                    {"key": "ktru_code", "match": {"text": "26.20"}},  # Компьютеры обычно 26.20
                ]
            },
            limit=5,
            with_payload=True,
            with_vectors=False
        )

        if computer_examples[0]:
            for i, point in enumerate(computer_examples[0], 1):
                payload = point.payload
                print(f"  {i}. {payload.get('ktru_code', 'N/A')} - {payload.get('title', 'N/A')[:60]}...")
        else:
            print("  ❌ Записи с компьютерной техникой не найдены!")

        return True

    except Exception as e:
        print(f"❌ Ошибка при проверке Qdrant: {e}")
        return False


def check_embeddings():
    """Проверка качества эмбеддингов"""
    print("\n🔍 2. ДИАГНОСТИКА ЭМБЕДДИНГОВ")
    print("=" * 50)

    try:
        # Тестируем эмбеддинги
        test_embeddings()

        # Тестируем поиск по эмбеддингам
        print(f"\n🔍 Тест поиска по эмбеддингам:")

        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        test_queries = [
            "ноутбук компьютер портативный",
            "ручка шариковая канцелярские",
            "стол письменный офисный мебель"
        ]

        for query in test_queries:
            print(f"\n  🔎 Запрос: '{query}'")

            # Генерируем эмбеддинг
            embedding = generate_embedding(query)
            print(f"     Размерность эмбеддинга: {len(embedding)}")
            print(f"     Норма вектора: {np.linalg.norm(embedding):.3f}")

            # Ищем похожие
            results = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=embedding.tolist(),
                limit=3
            )

            for i, result in enumerate(results, 1):
                score = getattr(result, 'score', 0)
                payload = result.payload
                print(
                    f"     {i}. {payload.get('ktru_code', 'N/A')} | {score:.3f} | {payload.get('title', 'N/A')[:50]}...")

        return True

    except Exception as e:
        print(f"❌ Ошибка при проверке эмбеддингов: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


def test_api_classification():
    """Тестирование API классификации"""
    print("\n🔍 3. ТЕСТИРОВАНИЕ API КЛАССИФИКАЦИИ")
    print("=" * 50)

    try:
        test_cases = [
            {
                "name": "Ноутбук (простой)",
                "data": {
                    "title": "Ноутбук",
                    "description": "Портативный компьютер"
                }
            },
            {
                "name": "Компьютер персональный",
                "data": {
                    "title": "Компьютер персональный",
                    "description": "Настольный компьютер для офиса"
                }
            },
            {
                "name": "Ручка шариковая",
                "data": {
                    "title": "Ручка шариковая",
                    "description": "Канцелярская принадлежность"
                }
            },
            {
                "name": "Бумага для принтера",
                "data": {
                    "title": "Бумага белая А4",
                    "description": "Бумага офисная для печати"
                }
            }
        ]

        successful = 0
        total = len(test_cases)

        for test_case in test_cases:
            print(f"\n🧪 {test_case['name']}")

            try:
                response = requests.post(
                    f"http://{API_HOST}:{API_PORT}/classify",
                    json=test_case['data'],
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    ktru_code = result.get('ktru_code', 'N/A')
                    ktru_title = result.get('ktru_title', 'N/A')
                    confidence = result.get('confidence', 0)
                    processing_time = result.get('processing_time', 0)

                    print(f"   ✅ Код: {ktru_code}")
                    if ktru_title and ktru_title != 'N/A':
                        print(f"   📋 Название: {ktru_title}")
                    print(f"   🎯 Уверенность: {confidence:.3f}")
                    print(f"   ⏱️  Время: {processing_time:.2f}с")

                    # Оценка корректности (примерная)
                    is_correct = False
                    if "ноутбук" in test_case['name'].lower() or "компьютер" in test_case['name'].lower():
                        if "26.20" in ktru_code or "компьютер" in ktru_title.lower():
                            is_correct = True
                    elif "ручка" in test_case['name'].lower():
                        if "ручка" in ktru_title.lower() or "30.19" in ktru_code:
                            is_correct = True
                    elif "бумага" in test_case['name'].lower():
                        if "бумага" in ktru_title.lower() or "17.23" in ktru_code:
                            is_correct = True

                    if is_correct:
                        print(f"   ✅ Результат выглядит корректно")
                        successful += 1
                    else:
                        print(f"   ❌ Результат выглядит некорректно")
                else:
                    print(f"   ❌ HTTP {response.status_code}: {response.text}")

            except Exception as e:
                print(f"   ❌ Ошибка: {e}")

        print(f"\n📊 Итого корректных результатов: {successful}/{total}")
        return successful > total // 2  # Больше половины корректных

    except Exception as e:
        print(f"❌ Ошибка при тестировании API: {e}")
        return False


def check_sample_data():
    """Проверка конкретных данных в базе"""
    print("\n🔍 4. ПРОВЕРКА КОНКРЕТНЫХ ДАННЫХ")
    print("=" * 50)

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Поиск конкретных категорий
        categories_to_check = [
            ("компьютер", "26.20"),
            ("ноутбук", "26.20"),
            ("ручка", "30.19"),
            ("бумага", "17.23"),
            ("стол", "31.09"),
            ("принтер", "30.20")
        ]

        for keyword, expected_code_prefix in categories_to_check:
            print(f"\n🔍 Поиск '{keyword}' (ожидается код {expected_code_prefix}.*)")

            # Поиск по тексту
            results = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter={
                    "should": [
                        {"key": "title", "match": {"text": keyword}},
                        {"key": "description", "match": {"text": keyword}},
                        {"key": "ktru_code", "match": {"text": expected_code_prefix}}
                    ]
                },
                limit=3,
                with_payload=True,
                with_vectors=False
            )

            if results[0]:
                for i, point in enumerate(results[0], 1):
                    payload = point.payload
                    code = payload.get('ktru_code', 'N/A')
                    title = payload.get('title', 'N/A')
                    print(f"   {i}. {code} - {title[:60]}...")
            else:
                print(f"   ❌ Записи с '{keyword}' не найдены")

        return True

    except Exception as e:
        print(f"❌ Ошибка при проверке данных: {e}")
        return False


def generate_recommendations():
    """Генерирует рекомендации по улучшению"""
    print("\n💡 5. РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ")
    print("=" * 50)

    recommendations = [
        "1. 🔧 Обновить файлы classifier.py и embedding.py исправленными версиями",
        "2. 📊 Проверить качество данных КТРУ в векторной базе",
        "3. 🎯 Понизить порог схожести с 0.75 до 0.65 для увеличения отзыва",
        "4. 📝 Упростить промпт для LLM модели",
        "5. 🔍 Добавить больше отладочной информации в логи",
        "6. 📈 Настроить более агрессивные параметры поиска",
        "7. 🧪 Добавить unit-тесты для проверки качества классификации",
        "8. 📋 Проверить, что данные КТРУ содержат правильные категории товаров"
    ]

    for rec in recommendations:
        print(f"   {rec}")

    print(f"\n🚀 ПЕРВООЧЕРЕДНЫЕ ДЕЙСТВИЯ:")
    print(f"   1. Замените classifier.py на исправленную версию")
    print(f"   2. Замените embedding.py на исправленную версию")
    print(f"   3. Перезапустите систему")
    print(f"   4. Повторите тесты")


def main():
    """Основная функция диагностики"""
    print("🔍 ДИАГНОСТИКА СИСТЕМЫ КЛАССИФИКАЦИИ КТРУ")
    print("=" * 60)

    results = []

    # 1. Проверка данных в Qdrant
    results.append(("Данные в Qdrant", check_qdrant_data()))

    # 2. Проверка эмбеддингов
    results.append(("Качество эмбеддингов", check_embeddings()))

    # 3. Тестирование API
    results.append(("API классификация", test_api_classification()))

    # 4. Проверка конкретных данных
    results.append(("Конкретные данные", check_sample_data()))

    # Итоговый отчет
    print(f"\n📋 ИТОГОВЫЙ ОТЧЕТ ДИАГНОСТИКИ")
    print("=" * 60)

    all_good = True
    for name, status in results:
        icon = "✅" if status else "❌"
        print(f"   {icon} {name}")
        if not status:
            all_good = False

    if not all_good:
        print(f"\n⚠️  ОБНАРУЖЕНЫ ПРОБЛЕМЫ!")
        generate_recommendations()
    else:
        print(f"\n🎉 Все проверки пройдены успешно!")

    return 0 if all_good else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)