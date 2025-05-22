#!/usr/bin/env python3
"""
Утилита для ручной загрузки данных КТРУ из JSON файла
Использование: python load_ktru_json.py --json_file path/to/ktru_data.json
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path

# Переходим в директорию проекта, если нужно
project_dir = "/workspace/rag-ktru-classifier"
if os.path.exists(project_dir) and os.getcwd() != project_dir:
    os.chdir(project_dir)
    print(f"Переход в директорию проекта: {project_dir}")

# Импортируем после смены директории
from process_ktru_json import process_ktru_json
from config import KTRU_JSON_PATH, DATA_DIR

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_json_file(json_path):
    """Проверка корректности JSON файла с данными КТРУ"""
    try:
        if not os.path.exists(json_path):
            logger.error(f"Файл не найден: {json_path}")
            return False

        # Проверяем размер файла
        file_size = os.path.getsize(json_path)
        if file_size < 100:
            logger.error(f"Файл слишком мал: {file_size} байт")
            return False

        logger.info(f"Размер файла: {file_size / (1024 * 1024):.2f} МБ")

        # Пытаемся загрузить и проверить структуру
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("JSON файл должен содержать массив объектов")
            return False

        if len(data) == 0:
            logger.error("JSON файл пуст")
            return False

        logger.info(f"Количество записей в JSON: {len(data):,}")

        # Проверяем структуру первой записи
        first_record = data[0]
        required_fields = ['ktru_code', 'title']

        for field in required_fields:
            if field not in first_record:
                logger.error(f"Отсутствует обязательное поле: {field}")
                return False

        logger.info(f"Пример записи: {first_record.get('ktru_code')} - {first_record.get('title')[:50]}...")

        return True

    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"Ошибка при проверке файла: {e}")
        return False


def copy_json_to_data_dir(source_path, target_path):
    """Копирование JSON файла в директорию данных"""
    try:
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Копируем файл
        import shutil
        shutil.copy2(source_path, target_path)

        logger.info(f"Файл скопирован: {source_path} -> {target_path}")
        return True

    except Exception as e:
        logger.error(f"Ошибка при копировании файла: {e}")
        return False


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Загрузка данных КТРУ из JSON файла')
    parser.add_argument('--json_file', type=str, help='Путь к JSON файлу с данными КТРУ')
    parser.add_argument('--copy_to_data', action='store_true',
                        help='Скопировать файл в директорию данных проекта')
    parser.add_argument('--force', action='store_true',
                        help='Принудительная загрузка без подтверждения')

    args = parser.parse_args()

    # Определяем путь к JSON файлу
    if args.json_file:
        json_path = args.json_file
    else:
        json_path = KTRU_JSON_PATH

    logger.info("🔄 Загрузка данных КТРУ из JSON файла")
    logger.info(f"Путь к файлу: {json_path}")

    # Проверяем файл
    if not validate_json_file(json_path):
        logger.error("❌ Проверка JSON файла не пройдена")
        return 1

    # Копируем файл в директорию данных, если нужно
    if args.copy_to_data and json_path != KTRU_JSON_PATH:
        if not copy_json_to_data_dir(json_path, KTRU_JSON_PATH):
            logger.error("❌ Не удалось скопировать файл")
            return 1
        json_path = KTRU_JSON_PATH

    # Запрашиваем подтверждение, если не используется --force
    if not args.force:
        response = input(f"\n🤔 Загрузить данные из {json_path}? Это может занять время. (y/N): ")
        if response.lower() not in ['y', 'yes', 'да']:
            logger.info("❌ Загрузка отменена пользователем")
            return 0

    # Запускаем загрузку
    try:
        logger.info("🚀 Начинаем загрузку данных в векторную базу...")
        process_ktru_json(json_path)
        logger.info("✅ Данные успешно загружены!")

        # Показываем статистику
        try:
            from qdrant_client import QdrantClient
            from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            collection_info = client.get_collection(QDRANT_COLLECTION)
            count = client.count(QDRANT_COLLECTION)

            logger.info(f"📊 Статистика векторной базы:")
            logger.info(f"   - Коллекция: {QDRANT_COLLECTION}")
            logger.info(f"   - Векторов: {count.count:,}")
            logger.info(f"   - Размерность: {collection_info.config.params.vectors.size}")

        except Exception as e:
            logger.warning(f"Не удалось получить статистику: {e}")

        return 0

    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке данных: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)