import time
import json
import numpy as np
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from tqdm import tqdm
import logging
from embedding import generate_embedding, generate_batch_embeddings
from config import (
    MONGO_EXTERNAL_URI, MONGO_DB_NAME, MONGO_COLLECTION,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, VECTOR_SIZE,
    BATCH_SIZE
)

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def connect_to_mongodb(uri, db_name, collection_name):
    """Подключение к MongoDB и получение коллекции"""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Проверка подключения
        client.admin.command('ping')
        logger.info(f"Успешное подключение к MongoDB: {uri}")

        db = client[db_name]
        collection = db[collection_name]
        return client, collection
    except Exception as e:
        logger.error(f"Ошибка подключения к MongoDB: {e}")
        return None, None


def prepare_ktru_text_enhanced(ktru_entry):
    """Улучшенная подготовка текста для эмбеддинга из записи КТРУ"""
    text_parts = []

    # Код и название - самые важные
    if ktru_entry.get('ktru_code'):
        text_parts.append(f"код КТРУ: {ktru_entry['ktru_code']}")

    if ktru_entry.get('title'):
        text_parts.append(f"название: {ktru_entry['title']}")

    # Описание
    if ktru_entry.get('description'):
        text_parts.append(f"описание: {ktru_entry['description']}")

    # Единица измерения
    if ktru_entry.get('unit'):
        text_parts.append(f"единица измерения: {ktru_entry['unit']}")

    # Ключевые слова
    if ktru_entry.get('keywords') and ktru_entry['keywords']:
        keywords_text = ', '.join(ktru_entry['keywords'])
        text_parts.append(f"ключевые слова: {keywords_text}")

    # Обработка атрибутов с полной информацией
    if 'attributes' in ktru_entry and ktru_entry['attributes']:
        attr_texts = []

        for attr in ktru_entry['attributes']:
            attr_name = attr.get('attr_name', '')

            # Обрабатываем атрибуты с attr_values (формат KTRU)
            if 'attr_values' in attr and attr['attr_values']:
                values = []
                for val in attr['attr_values']:
                    value_text = val.get('value', '')
                    value_unit = val.get('value_unit', '')

                    # Формируем полное значение с единицей измерения
                    if value_unit and value_unit not in ['', 'Нет данных']:
                        # Обрабатываем составные единицы измерения
                        units = value_unit.split(';')
                        unit_text = ' или '.join(u.strip() for u in units if u.strip())
                        full_value = f"{value_text} {unit_text}"
                    else:
                        full_value = value_text

                    values.append(full_value)

                if values:
                    attr_text = f"{attr_name}: {', '.join(values)}"
                    attr_texts.append(attr_text)

            # Обрабатываем атрибуты с attr_value (альтернативный формат)
            elif 'attr_value' in attr and attr['attr_value']:
                attr_text = f"{attr_name}: {attr['attr_value']}"
                attr_texts.append(attr_text)

        if attr_texts:
            text_parts.append("характеристики: " + '; '.join(attr_texts))

    # Добавляем метаданные для улучшения поиска
    # Извлекаем категорию из кода KTRU (первые цифры)
    if ktru_entry.get('ktru_code'):
        code_parts = ktru_entry['ktru_code'].split('.')
        if len(code_parts) >= 2:
            category_code = f"{code_parts[0]}.{code_parts[1]}"
            text_parts.append(f"категория кода: {category_code}")

    # Объединяем все части
    full_text = ' | '.join(text_parts)

    return full_text


def setup_qdrant_collection():
    """Настройка коллекции Qdrant"""
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
                )
            )
        else:
            logger.info(f"Коллекция {QDRANT_COLLECTION} уже существует в Qdrant")

        return qdrant_client
    except Exception as e:
        logger.error(f"Ошибка при настройке Qdrant: {e}")
        return None


def process_ktru_data():
    """Основная функция обработки данных КТРУ"""
    start_time = time.time()

    # Подключение к MongoDB
    mongo_client, ktru_collection = connect_to_mongodb(
        MONGO_EXTERNAL_URI, MONGO_DB_NAME, MONGO_COLLECTION)
    if not mongo_client or not ktru_collection:
        logger.error("Не удалось подключиться к MongoDB. Завершение процесса.")
        return

    # Настройка Qdrant
    qdrant_client = setup_qdrant_collection()
    if not qdrant_client:
        logger.error("Не удалось настроить Qdrant. Завершение процесса.")
        mongo_client.close()
        return

    try:
        # Получаем общее количество записей для tqdm
        total_records = ktru_collection.count_documents({})
        logger.info(f"Всего записей КТРУ для обработки: {total_records}")

        if total_records == 0:
            logger.warning("Нет данных КТРУ для обработки.")
            mongo_client.close()
            return

        # Обработка данных пакетами
        batch_size = BATCH_SIZE
        processed_count = 0

        # Получаем курсор на все записи
        cursor = ktru_collection.find({})

        while True:
            batch = []
            batch_texts = []
            batch_ids = []

            # Собираем пакет записей
            for _ in range(batch_size):
                try:
                    ktru_entry = next(cursor)
                    # Используем улучшенную функцию подготовки текста
                    text_to_embed = prepare_ktru_text_enhanced(ktru_entry)
                    batch_texts.append(text_to_embed)
                    batch.append(ktru_entry)
                    batch_ids.append(processed_count)
                    processed_count += 1
                except StopIteration:
                    break

            if not batch:
                break

            # Генерируем эмбеддинги для пакета
            batch_embeddings = generate_batch_embeddings(batch_texts)

            # Создаем записи для Qdrant
            points = []
            for i, (ktru_entry, embedding) in enumerate(zip(batch, batch_embeddings)):
                # Сохраняем полную информацию в payload
                payload = {
                    "ktru_code": ktru_entry.get('ktru_code', ''),
                    "title": ktru_entry.get('title', ''),
                    "description": ktru_entry.get('description', ''),
                    "unit": ktru_entry.get('unit', ''),
                    "version": ktru_entry.get('version', ''),
                    "keywords": ktru_entry.get('keywords', []),
                    "attributes": ktru_entry.get('attributes', []),
                    "source_link": ktru_entry.get('source_link', ''),
                    "updated_at": str(ktru_entry.get('updated_at', ''))
                }

                # Добавляем дополнительное поле с текстом для поиска (для отладки)
                payload['_search_text'] = batch_texts[i][:500]  # Первые 500 символов

                points.append(rest.PointStruct(
                    id=batch_ids[i],
                    vector=embedding.tolist(),
                    payload=payload
                ))

            # Загружаем точки в Qdrant
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )

            logger.info(f"Обработано {processed_count}/{total_records} записей КТРУ")

        elapsed_time = time.time() - start_time
        logger.info(f"Обработка КТРУ завершена. Всего обработано {processed_count} записей.")
        logger.info(f"Время обработки: {elapsed_time:.2f} секунд")

        # Проверяем результат
        collection_info = qdrant_client.get_collection(QDRANT_COLLECTION)
        count = qdrant_client.count(QDRANT_COLLECTION)
        logger.info(f"📊 Финальная статистика коллекции {QDRANT_COLLECTION}:")
        logger.info(f"   - Векторов в базе: {count.count:,}")
        logger.info(f"   - Размерность: {collection_info.config.params.vectors.size}")

    except Exception as e:
        logger.error(f"Ошибка при обработке данных КТРУ: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

    finally:
        mongo_client.close()


if __name__ == "__main__":
    process_ktru_data()