import time
import json
import numpy as np
import argparse
from tqdm import tqdm
import logging
import re
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from embedding import generate_embedding, generate_batch_embeddings
from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, VECTOR_SIZE,
    BATCH_SIZE
)

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ktru_data_from_json(json_file_path):
    """Загрузка данных КТРУ из JSON-файла"""
    try:
        logger.info(f"Загрузка данных КТРУ из файла: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            ktru_data = json.load(f)
        logger.info(f"Успешно загружено записей: {len(ktru_data)}")
        return ktru_data
    except Exception as e:
        logger.error(f"Ошибка загрузки данных из JSON-файла: {e}")
        return None


def extract_category_from_code(ktru_code):
    """Извлечение категории из КТРУ кода"""
    if not ktru_code:
        return None

    # КТРУ код имеет формат XX.XX.XX.XXX-XXXXXXXX
    parts = ktru_code.split('-')[0].split('.')
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return None


def normalize_text(text):
    """Нормализация текста для лучшего поиска"""
    if not text:
        return ""

    text = text.lower()
    # Заменяем множественные пробелы одним
    text = re.sub(r'\s+', ' ', text)
    # Удаляем специальные символы, оставляя буквы, цифры и основную пунктуацию
    text = re.sub(r'[^\w\s\-.,;:()]', ' ', text)
    return text.strip()


def extract_keywords_from_title(title):
    """Извлечение ключевых слов из названия"""
    if not title:
        return []

    normalized = normalize_text(title)
    # Извлекаем слова длиннее 2 символов
    words = re.findall(r'\b[а-яёa-z0-9]{3,}\b', normalized, re.IGNORECASE)

    # Фильтруем стоп-слова
    stop_words = {'для', 'или', 'под', 'над', 'все', 'при', 'без'}
    keywords = [w for w in words if w not in stop_words]

    return keywords


def prepare_ktru_text_optimized(ktru_entry):
    """Оптимизированная подготовка текста для эмбеддинга с упором на название"""
    text_parts = []

    # НАЗВАНИЕ - самый важный элемент
    title = ktru_entry.get('title', '')
    if title:
        # Добавляем название несколько раз для увеличения веса
        text_parts.append(f"название: {title}")
        text_parts.append(f"товар: {title}")

        # Добавляем ключевые слова из названия
        keywords = extract_keywords_from_title(title)
        if keywords:
            text_parts.append(f"ключевые слова: {' '.join(keywords)}")

    # Код КТРУ и категория
    ktru_code = ktru_entry.get('ktru_code', '')
    if ktru_code:
        text_parts.append(f"код КТРУ: {ktru_code}")

        # Извлекаем категорию
        category = extract_category_from_code(ktru_code)
        if category:
            text_parts.append(f"категория: {category}")

    # Описание (если есть)
    description = ktru_entry.get('description', '')
    if description and description.strip():
        # Сокращаем длинные описания
        desc_normalized = normalize_text(description)
        if len(desc_normalized) > 200:
            desc_normalized = desc_normalized[:200] + "..."
        text_parts.append(f"описание: {desc_normalized}")

    # Единица измерения
    unit = ktru_entry.get('unit', '')
    if unit and unit not in ['Нет данных', '']:
        text_parts.append(f"единица: {unit}")

    # Атрибуты (только важные)
    if 'attributes' in ktru_entry and ktru_entry['attributes']:
        important_attrs = []

        for attr in ktru_entry['attributes'][:5]:  # Берем только первые 5 атрибутов
            attr_name = attr.get('attr_name', '')

            # Обрабатываем значения атрибута
            values = []
            if 'attr_values' in attr and attr['attr_values']:
                for val in attr['attr_values'][:3]:  # Максимум 3 значения
                    value_text = val.get('value', '')
                    if value_text:
                        values.append(value_text)
            elif 'attr_value' in attr:
                values.append(attr['attr_value'])

            if values and attr_name:
                attr_text = f"{attr_name}: {', '.join(values)}"
                important_attrs.append(attr_text)

        if important_attrs:
            text_parts.append(f"характеристики: {'; '.join(important_attrs)}")

    # Создаем поисковые вариации для лучшего matching
    if title:
        # Добавляем вариации названия
        title_words = title.split()
        if len(title_words) > 2:
            # Добавляем первые и последние слова
            text_parts.append(f"начало: {' '.join(title_words[:2])}")
            text_parts.append(f"конец: {' '.join(title_words[-2:])}")

    # Объединяем все части
    full_text = ' | '.join(text_parts)

    return full_text


def create_search_index(ktru_data):
    """Создание поискового индекса для быстрого поиска по названиям"""
    search_index = {
        'by_keyword': defaultdict(list),
        'by_category': defaultdict(list),
        'by_code': {},
        'full_data': {}
    }

    for idx, entry in enumerate(ktru_data):
        ktru_code = entry.get('ktru_code', '')
        title = entry.get('title', '')

        # Индекс по коду
        search_index['by_code'][ktru_code] = idx
        search_index['full_data'][ktru_code] = entry

        # Индекс по категории
        category = extract_category_from_code(ktru_code)
        if category:
            search_index['by_category'][category].append(ktru_code)

        # Индекс по ключевым словам
        keywords = extract_keywords_from_title(title)
        for keyword in keywords:
            search_index['by_keyword'][keyword].append(ktru_code)

    logger.info(f"Создан поисковый индекс: {len(search_index['by_keyword'])} ключевых слов, "
                f"{len(search_index['by_category'])} категорий")

    return search_index


def setup_qdrant_collection():
    """Настройка коллекции Qdrant с оптимальными параметрами"""
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Проверяем, существует ли коллекция
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if QDRANT_COLLECTION not in collection_names:
            logger.info(f"Создание коллекции {QDRANT_COLLECTION} в Qdrant")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=rest.VectorParams(
                    size=VECTOR_SIZE,
                    distance=rest.Distance.COSINE
                ),
                # Оптимизация для поиска
                optimizers_config=rest.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=20000
                )
            )
        else:
            logger.info(f"Коллекция {QDRANT_COLLECTION} уже существует")

            # Опционально: пересоздаем коллекцию для чистой загрузки
            response = input("Хотите пересоздать коллекцию? (y/N): ")
            if response.lower() == 'y':
                logger.info("Удаление существующей коллекции...")
                qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION)

                logger.info("Создание новой коллекции...")
                qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=rest.VectorParams(
                        size=VECTOR_SIZE,
                        distance=rest.Distance.COSINE
                    ),
                    optimizers_config=rest.OptimizersConfigDiff(
                        default_segment_number=5,
                        indexing_threshold=20000
                    )
                )

        return qdrant_client
    except Exception as e:
        logger.error(f"Ошибка при настройке Qdrant: {e}")
        return None


def process_ktru_json(json_file_path):
    """Основная функция обработки данных КТРУ из JSON-файла"""
    start_time = time.time()

    # Загрузка данных
    ktru_data = load_ktru_data_from_json(json_file_path)
    if not ktru_data:
        logger.error("Не удалось загрузить данные КТРУ из JSON-файла")
        return

    # Создание поискового индекса
    search_index = create_search_index(ktru_data)

    # Настройка Qdrant
    qdrant_client = setup_qdrant_collection()
    if not qdrant_client:
        logger.error("Не удалось настроить Qdrant")
        return

    try:
        total_records = len(ktru_data)
        logger.info(f"Всего записей КТРУ для обработки: {total_records}")

        # Обработка пакетами
        batch_size = BATCH_SIZE
        processed_count = 0

        # Прогресс-бар
        with tqdm(total=total_records, desc="Обработка КТРУ") as pbar:
            for i in range(0, total_records, batch_size):
                batch = ktru_data[i:i + batch_size]
                batch_texts = []
                batch_ids = []

                # Подготовка текстов
                for idx, ktru_entry in enumerate(batch):
                    text_to_embed = prepare_ktru_text_optimized(ktru_entry)
                    batch_texts.append(text_to_embed)
                    batch_ids.append(i + idx)

                # Генерация эмбеддингов
                batch_embeddings = generate_batch_embeddings(batch_texts)

                # Создание точек для Qdrant
                points = []
                for idx, (ktru_entry, embedding) in enumerate(zip(batch, batch_embeddings)):
                    # Подготовка payload
                    payload = {
                        "ktru_code": ktru_entry.get('ktru_code', ''),
                        "title": ktru_entry.get('title', ''),
                        "description": ktru_entry.get('description', ''),
                        "unit": ktru_entry.get('unit', ''),
                        "version": ktru_entry.get('version', ''),
                        "keywords": extract_keywords_from_title(ktru_entry.get('title', '')),
                        "category": extract_category_from_code(ktru_entry.get('ktru_code', '')),
                        "attributes": ktru_entry.get('attributes', []),
                        "source_link": ktru_entry.get('source_link', ''),
                        "updated_at": ktru_entry.get('updated_at', ''),
                        "_search_text": batch_texts[idx][:500]  # Для отладки
                    }

                    points.append(rest.PointStruct(
                        id=batch_ids[idx],
                        vector=embedding.tolist(),
                        payload=payload
                    ))

                # Загрузка в Qdrant
                qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points
                )

                processed_count += len(batch)
                pbar.update(len(batch))

        # Создание текстового индекса в Qdrant для полнотекстового поиска
        logger.info("Создание индексов для оптимизации поиска...")
        try:
            # Создаем payload индекс для быстрого поиска по коду
            qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="ktru_code",
                field_schema=rest.PayloadSchemaType.KEYWORD
            )

            # Индекс для категории
            qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="category",
                field_schema=rest.PayloadSchemaType.KEYWORD
            )

            logger.info("✅ Индексы созданы")
        except Exception as e:
            logger.warning(f"Не удалось создать индексы (возможно уже существуют): {e}")

        elapsed_time = time.time() - start_time
        logger.info(f"Обработка завершена за {elapsed_time:.2f} секунд")
        logger.info(f"Обработано {processed_count} записей КТРУ")

        # Статистика
        collection_info = qdrant_client.get_collection(QDRANT_COLLECTION)
        count = qdrant_client.count(QDRANT_COLLECTION)
        logger.info(f"📊 Финальная статистика коллекции {QDRANT_COLLECTION}:")
        logger.info(f"   - Векторов в базе: {count.count:,}")
        logger.info(f"   - Размерность: {collection_info.config.params.vectors.size}")
        logger.info(f"   - Статус: {collection_info.status}")

    except Exception as e:
        logger.error(f"Ошибка при обработке данных КТРУ: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def analyze_ktru_data(json_file_path):
    """Анализ данных КТРУ для понимания структуры"""
    ktru_data = load_ktru_data_from_json(json_file_path)
    if not ktru_data:
        return

    logger.info("Анализ структуры данных КТРУ...")

    # Статистика по категориям
    categories = defaultdict(int)
    title_lengths = []
    has_description = 0
    has_attributes = 0

    for entry in ktru_data:
        # Категории
        category = extract_category_from_code(entry.get('ktru_code', ''))
        if category:
            categories[category] += 1

        # Длина названий
        title = entry.get('title', '')
        if title:
            title_lengths.append(len(title))

        # Наличие описаний
        if entry.get('description', '').strip():
            has_description += 1

        # Наличие атрибутов
        if entry.get('attributes', []):
            has_attributes += 1

    # Вывод статистики
    logger.info(f"📊 Статистика данных КТРУ:")
    logger.info(f"   - Всего записей: {len(ktru_data)}")
    logger.info(f"   - Уникальных категорий: {len(categories)}")
    logger.info(f"   - Топ-10 категорий:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"     * {cat}: {count} записей")
    logger.info(f"   - Средняя длина названия: {np.mean(title_lengths):.1f} символов")
    logger.info(f"   - Записей с описанием: {has_description} ({has_description / len(ktru_data) * 100:.1f}%)")
    logger.info(f"   - Записей с атрибутами: {has_attributes} ({has_attributes / len(ktru_data) * 100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обработка данных КТРУ из JSON-файла')
    parser.add_argument('--json_file', type=str, required=True, help='Путь к JSON-файлу с данными КТРУ')
    parser.add_argument('--analyze', action='store_true', help='Только анализ данных без загрузки')
    args = parser.parse_args()

    if args.analyze:
        analyze_ktru_data(args.json_file)
    else:
        process_ktru_json(args.json_file)