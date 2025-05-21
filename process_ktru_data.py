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


def prepare_ktru_text(ktru_entry):
    """Подготовка текста для эмбеддинга из записи КТРУ"""
    text_to_embed = f"{ktru_entry.get('ktru_code', '')} {ktru_entry.get('title', '')}"

    # Добавляем описание, если оно есть
    if 'description' in ktru_entry and ktru_entry['description']:
        text_to_embed += f" {ktru_entry['description']}"

    # Добавляем информацию об атрибутах
    if 'attributes' in ktru_entry and ktru_entry['attributes']:
        for attr in ktru_entry['attributes']:
            attr_name = attr.get('attr_name', '')

            # Обрабатываем атрибуты с attr_values
            if 'attr_values' in attr and attr['attr_values']:
                for val in attr['attr_values']:
                    value = val.get('value', '')
                    value_unit = val.get('value_unit', '')
                    text_to_embed += f" {attr_name}: {value} {value_unit}".strip()

            # Обрабатываем атрибуты с attr_value
            elif 'attr_value' in attr and attr['attr_value']:
                text_to_embed += f" {attr_name}: {attr['attr_value']}"

    return text_to_embed


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
                    text_to_embed = prepare_ktru_text(ktru_entry)
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
                points.append(rest.PointStruct(
                    id=batch_ids[i],
                    vector=embedding.tolist(),
                    payload={
                        "ktru_code": ktru_entry.get('ktru_code', ''),
                        "title": ktru_entry.get('title', ''),
                        "description": ktru_entry.get('description', ''),
                        "unit": ktru_entry.get('unit', ''),
                        "version": ktru_entry.get('version', ''),
                        "attributes": ktru_entry.get('attributes', [])
                    }
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

    except Exception as e:
        logger.error(f"Ошибка при обработке данных КТРУ: {e}")

    finally:
        mongo_client.close()


if __name__ == "__main__":
    process_ktru_data()