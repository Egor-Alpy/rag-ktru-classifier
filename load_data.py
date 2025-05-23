"""
Скрипт для загрузки данных KTRU из JSON в векторную БД
"""

import argparse
import logging
import sys
from pathlib import Path
import json

from config import KTRU_JSON_PATH, DATA_DIR
from vector_db import vector_db

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_json_file(json_path: Path) -> bool:
    """Проверка корректности JSON файла"""
    try:
        if not json_path.exists():
            logger.error(f"Файл не найден: {json_path}")
            return False

        # Проверяем размер
        file_size = json_path.stat().st_size
        if file_size < 100:
            logger.error(f"Файл слишком мал: {file_size} байт")
            return False

        logger.info(f"Размер файла: {file_size / (1024 * 1024):.2f} МБ")

        # Пробуем загрузить
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("JSON должен содержать массив объектов")
            return False

        if not data:
            logger.error("JSON файл пуст")
            return False

        logger.info(f"Найдено записей: {len(data):,}")

        # Проверяем структуру первой записи
        first = data[0]
        required_fields = ['ktru_code', 'title']

        for field in required_fields:
            if field not in first:
                logger.error(f"Отсутствует обязательное поле: {field}")
                return False

        logger.info(f"Пример записи: {first['ktru_code']} - {first['title'][:50]}...")

        # Статистика по полям
        fields_stats = {}
        for record in data[:1000]:  # Проверяем первые 1000 записей
            for field in record:
                fields_stats[field] = fields_stats.get(field, 0) + 1

        logger.info("Статистика полей:")
        for field, count in sorted(fields_stats.items()):
            percent = count / min(len(data), 1000) * 100
            logger.info(f"  - {field}: {percent:.1f}%")

        return True

    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"Ошибка при проверке файла: {e}")
        return False


def create_sample_json():
    """Создание примера JSON файла с данными KTRU"""
    sample_data = [
        {
            "ktru_code": "26.20.11.110-00000001",
            "title": "Компьютеры портативные массой не более 10 кг, такие как ноутбуки",
            "description": "Портативные компьютеры (ноутбуки) для офисной работы",
            "unit": "Штука",
            "keywords": ["ноутбук", "компьютер", "портативный"],
            "attributes": [
                {
                    "attr_name": "Тип процессора",
                    "attr_values": [
                        {"value": "Intel Core i5", "value_unit": ""},
                        {"value": "Intel Core i7", "value_unit": ""},
                        {"value": "AMD Ryzen 5", "value_unit": ""}
                    ]
                },
                {
                    "attr_name": "Объем оперативной памяти",
                    "attr_values": [
                        {"value": "8", "value_unit": "ГБ"},
                        {"value": "16", "value_unit": "ГБ"}
                    ]
                }
            ]
        },
        {
            "ktru_code": "32.99.12.110-00000001",
            "title": "Ручки шариковые",
            "description": "Ручки шариковые для письма",
            "unit": "Штука",
            "keywords": ["ручка", "канцелярия", "письменные принадлежности"],
            "attributes": [
                {
                    "attr_name": "Цвет чернил",
                    "attr_values": [
                        {"value": "Синий", "value_unit": ""},
                        {"value": "Черный", "value_unit": ""},
                        {"value": "Красный", "value_unit": ""}
                    ]
                }
            ]
        },
        {
            "ktru_code": "17.12.14.110-00000001",
            "title": "Бумага для печати",
            "description": "Бумага офисная для печати и копирования",
            "unit": "Пачка",
            "keywords": ["бумага", "офисная", "печать"],
            "attributes": [
                {
                    "attr_name": "Формат",
                    "attr_values": [
                        {"value": "А4", "value_unit": ""},
                        {"value": "А3", "value_unit": ""}
                    ]
                },
                {
                    "attr_name": "Плотность",
                    "attr_values": [
                        {"value": "80", "value_unit": "г/м2"}
                    ]
                }
            ]
        }
    ]

    sample_path = DATA_DIR / "ktru_sample.json"

    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Создан пример файла: {sample_path}")
    logger.info("Добавьте больше записей в этот файл или используйте свой JSON")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description='Загрузка данных KTRU в векторную БД'
    )
    parser.add_argument(
        '--json-file',
        type=str,
        help='Путь к JSON файлу с данными KTRU'
    )
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='Пересоздать коллекцию (удалить существующие данные)'
    )
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Создать пример JSON файла'
    )

    args = parser.parse_args()

    # Создание примера
    if args.create_sample:
        create_sample_json()
        return 0

    # Определяем путь к файлу
    if args.json_file:
        json_path = Path(args.json_file)
    else:
        json_path = KTRU_JSON_PATH

    logger.info("=" * 60)
    logger.info("ЗАГРУЗКА ДАННЫХ KTRU В ВЕКТОРНУЮ БД")
    logger.info("=" * 60)

    # Проверка файла
    logger.info(f"Проверка файла: {json_path}")
    if not validate_json_file(json_path):
        logger.error("❌ Проверка не пройдена")
        return 1

    # Подтверждение
    if not args.recreate:
        response = input("\nПродолжить загрузку? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Загрузка отменена")
            return 0

    try:
        # Загрузка данных
        logger.info("Начинаем загрузку...")
        vector_db.load_ktru_from_json(json_path, recreate=args.recreate)

        logger.info("✅ Загрузка успешно завершена!")

        # Показываем статистику
        stats = vector_db.get_statistics()
        logger.info("\nСтатистика:")
        logger.info(f"  - Всего векторов: {stats.get('total_vectors', 0):,}")
        logger.info(f"  - Категорий: {len(stats.get('categories', {}))}")

        # Топ категорий
        if stats.get('categories'):
            logger.info("\nТоп-10 категорий:")
            categories = stats['categories']
            for cat, count in sorted(
                    categories.items(),
                    key=lambda x: x[1],
                    reverse=True
            )[:10]:
                logger.info(f"  - {cat}: {count:,} записей")

        return 0

    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
