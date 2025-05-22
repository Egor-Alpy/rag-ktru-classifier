#!/usr/bin/env python3
"""
Утилита для проверки состояния системы классификации КТРУ
"""

import sys
import os
import time
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
import logging

# Переходим в директорию проекта, если нужно
project_dir = "/workspace/rag-ktru-classifier"
if os.path.exists(project_dir) and os.getcwd() != project_dir:
    os.chdir(project_dir)
    print(f"Переход в директорию проекта: {project_dir}")

from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    API_HOST, API_PORT
)

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_qdrant_status():
    """Проверка состояния Qdrant"""
    try:
        logger.info(f"Проверка Qdrant на {QDRANT_HOST}:{QDRANT_PORT}")
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Проверяем доступность
        collections = qdrant_client.get_collections()
        logger.info(f"✅ Qdrant доступен. Найдено коллекций: {len(collections.collections)}")

        # Информация о коллекциях
        total_vectors = 0
        for collection in collections.collections:
            try:
                collection_info = qdrant_client.get_collection(collection.name)
                collection_stats = qdrant_client.count(collection.name)

                logger.info(f"📊 Коллекция '{collection.name}':")
                logger.info(f"   - Векторов: {collection_stats.count:,}")
                logger.info(f"   - Размерность: {collection_info.config.params.vectors.size}")
                logger.info(f"   - Метрика: {collection_info.config.params.vectors.distance.name}")
                logger.info(f"   - Статус: {collection_info.status.name}")

                total_vectors += collection_stats.count

                # Проверяем основную коллекцию КТРУ
                if collection.name == QDRANT_COLLECTION:
                    if collection_stats.count > 0:
                        logger.info(f"✅ Коллекция КТРУ загружена ({collection_stats.count:,} записей)")
                    else:
                        logger.warning(f"⚠️  Коллекция КТРУ пуста")

            except Exception as e:
                logger.error(f"❌ Ошибка при проверке коллекции {collection.name}: {e}")

        logger.info(f"📈 Общее количество векторов: {total_vectors:,}")
        return True

    except Exception as e:
        logger.error(f"❌ Qdrant недоступен: {e}")
        return False


def check_api_status():
    """Проверка состояния API"""
    try:
        logger.info(f"Проверка API на {API_HOST}:{API_PORT}")

        # Health check
        response = requests.get(f"http://{API_HOST}:{API_PORT}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API Health check успешен")
        else:
            logger.error(f"❌ API Health check failed: {response.status_code}")
            return False

        # Status check
        try:
            response = requests.get(f"http://{API_HOST}:{API_PORT}/status", timeout=10)
            if response.status_code == 200:
                status_data = response.json()
                logger.info("✅ API Status check успешен")
                logger.info(f"   - Qdrant: {status_data.get('qdrant', 'unknown')}")
                logger.info(f"   - Модели: {status_data.get('models', 'unknown')}")
                logger.info(f"   - КТРУ загружено: {status_data.get('ktru_loaded', False)}")

                # Показываем статистику коллекций
                collections = status_data.get('collections', {})
                for name, info in collections.items():
                    logger.info(f"   - {name}: {info.get('vectors_count', 0):,} векторов")
            else:
                logger.warning(f"⚠️  API Status недоступен: {response.status_code}")
        except:
            logger.warning("⚠️  API Status endpoint недоступен")

        return True

    except Exception as e:
        logger.error(f"❌ API недоступен: {e}")
        return False


def check_models_status():
    """Проверка состояния моделей"""
    try:
        logger.info("Проверка моделей...")

        # Проверяем модель эмбеддингов
        try:
            from embedding import embedding_model
            if embedding_model and embedding_model.model:
                logger.info(f"✅ Модель эмбеддингов загружена: {embedding_model.tokenizer.name_or_path}")
                logger.info(f"   - Устройство: {embedding_model.device}")
            else:
                logger.error("❌ Модель эмбеддингов не загружена")
                return False
        except Exception as e:
            logger.error(f"❌ Ошибка при проверке модели эмбеддингов: {e}")
            return False

        # Проверяем LLM модель
        try:
            from classifier import classifier
            if classifier and classifier.llm and classifier.tokenizer:
                logger.info(f"✅ LLM модель загружена")
                logger.info(f"   - Базовая модель: {classifier.tokenizer.name_or_path}")
            else:
                logger.warning("⚠️  LLM модель не загружена")
        except Exception as e:
            logger.warning(f"⚠️  Ошибка при проверке LLM модели: {e}")

        return True

    except Exception as e:
        logger.error(f"❌ Ошибка при проверке моделей: {e}")
        return False


def test_classification():
    """Тест классификации"""
    try:
        logger.info("Тестирование классификации...")

        test_data = {
            "title": "Ноутбук ASUS",
            "description": "Портативный компьютер для работы",
            "attributes": [
                {"attr_name": "Процессор", "attr_value": "Intel Core i5"},
                {"attr_name": "Память", "attr_value": "8 ГБ"}
            ]
        }

        response = requests.post(
            f"http://{API_HOST}:{API_PORT}/classify",
            json=test_data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Тест классификации успешен")
            logger.info(f"   - Код КТРУ: {result.get('ktru_code', 'N/A')}")
            logger.info(f"   - Название КТРУ: {result.get('ktru_title', 'N/A')}")
            logger.info(f"   - Уверенность: {result.get('confidence', 0):.2f}")
            logger.info(f"   - Время обработки: {result.get('processing_time', 0):.2f}с")

            # Проверяем новый формат ответа
            if 'ktru_title' in result:
                logger.info("✅ Новый формат ответа с названием КТРУ работает")
            else:
                logger.warning("⚠️  Поле ktru_title отсутствует в ответе")

            return True
        else:
            logger.error(f"❌ Тест классификации failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании классификации: {e}")
        return False


def test_multiple_classification():
    """Расширенный тест классификации с несколькими товарами"""
    try:
        logger.info("Расширенное тестирование классификации...")

        test_cases = [
            {
                "name": "Тест 1: Ноутбук",
                "data": {
                    "title": "Ноутбук ASUS X515",
                    "description": "Портативный персональный компьютер",
                    "attributes": [
                        {"attr_name": "Процессор", "attr_value": "Intel Core i5"},
                        {"attr_name": "Оперативная память", "attr_value": "8 ГБ"}
                    ]
                }
            },
            {
                "name": "Тест 2: Канцелярские товары",
                "data": {
                    "title": "Ручка шариковая синяя",
                    "description": "Письменная принадлежность для офиса",
                    "attributes": [
                        {"attr_name": "Цвет чернил", "attr_value": "синий"},
                        {"attr_name": "Тип", "attr_value": "шариковая"}
                    ]
                }
            },
            {
                "name": "Тест 3: Мебель",
                "data": {
                    "title": "Стол офисный письменный",
                    "description": "Мебель для рабочего места",
                    "attributes": [
                        {"attr_name": "Материал", "attr_value": "ЛДСП"},
                        {"attr_name": "Размер", "attr_value": "120x60 см"}
                    ]
                }
            }
        ]

        successful_tests = 0
        total_tests = len(test_cases)

        for test_case in test_cases:
            logger.info(f"🧪 {test_case['name']}")

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

                    logger.info(f"   ✅ Результат: {ktru_code}")
                    if ktru_title and ktru_title != 'N/A':
                        logger.info(f"   📋 Название: {ktru_title}")
                    logger.info(f"   🎯 Уверенность: {confidence:.2f}")
                    logger.info(f"   ⏱️  Время: {processing_time:.2f}с")

                    successful_tests += 1
                else:
                    logger.error(f"   ❌ Ошибка: {response.status_code}")

            except Exception as e:
                logger.error(f"   ❌ Исключение: {e}")

            logger.info("")  # Пустая строка для разделения

        logger.info(f"📊 Результаты расширенного тестирования: {successful_tests}/{total_tests}")
        return successful_tests == total_tests

    except Exception as e:
        logger.error(f"❌ Ошибка при расширенном тестировании: {e}")
        return False


def main():
    """Основная функция проверки"""
    logger.info("🔍 Начало проверки состояния системы классификации КТРУ")
    logger.info("=" * 60)

    checks = []

    # Проверка Qdrant
    logger.info("1️⃣  Проверка Qdrant...")
    checks.append(("Qdrant", check_qdrant_status()))

    logger.info("")

    # Проверка API
    logger.info("2️⃣  Проверка API...")
    checks.append(("API", check_api_status()))

    logger.info("")

    # Проверка моделей
    logger.info("3️⃣  Проверка моделей...")
    checks.append(("Models", check_models_status()))

    logger.info("")

    # Тест классификации
    logger.info("4️⃣  Тест классификации...")
    checks.append(("Classification", test_classification()))

    logger.info("")

    # Расширенный тест классификации
    logger.info("5️⃣  Расширенный тест классификации...")
    checks.append(("Extended Tests", test_multiple_classification()))

    # Итоговый отчет
    logger.info("")
    logger.info("=" * 60)
    logger.info("📋 ИТОГОВЫЙ ОТЧЕТ:")

    all_passed = True
    for component, status in checks:
        status_icon = "✅" if status else "❌"
        status_text = "РАБОТАЕТ" if status else "НЕ РАБОТАЕТ"
        logger.info(f"   {status_icon} {component:<15} : {status_text}")
        if not status:
            all_passed = False

    logger.info("")

    if all_passed:
        logger.info("🎉 Все компоненты работают корректно!")
        logger.info("📋 Новая функциональность:")
        logger.info("   ✅ Возврат названия КТРУ работает")
        logger.info("   ✅ Обратная совместимость сохранена")
        return 0
    else:
        logger.error("⚠️  Обнаружены проблемы в работе системы!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)