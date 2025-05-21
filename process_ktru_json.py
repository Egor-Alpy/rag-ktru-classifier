import time
import json
import numpy as np
import argparse
from tqdm import tqdm
import logging
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


def process_ktru_json(json_file_path):
    """Основная функция обработки данных КТРУ из JSON-файла"""
    start_time = time.time()

    # Загрузка данных из JSON-файла
    ktru_data = load_ktru_data_from_json(json_file_path)
    if not ktru_data:
        logger.error("Не удалось загрузить данные КТРУ из JSON-файла. Завершение процесса.")
        return

    # Настройка Qdrant
    qdrant_client = setup_qdrant_collection()
    if not qdrant_client:
        logger.error("Не удалось настроить Qdrant. Завершение процесса.")
        return

    try:
        # Получаем общее количество записей
        total_records = len(ktru_data)
        logger.info(f"Всего записей КТРУ для обработки: {total_records}")

        if total_records == 0:
            logger.warning("Нет данных КТРУ для обработки.")
            return

        # Обработка данных пакетами
        batch_size = BATCH_SIZE
        processed_count = 0

        # Обработка данных пакетами
        for i in range(0, total_records, batch_size):
            batch = ktru_data[i:i+batch_size]
            batch_texts = []
            batch_ids = []

            # Подготовка текстов для эмбеддингов
            for idx, ktru_entry in enumerate(batch):
                text_to_embed = prepare_ktru_text(ktru_entry)
                batch_texts.append(text_to_embed)
                batch_ids.append(i + idx)  # Используем индекс как ID

            # Генерируем эмбеддинги для пакета
            batch_embeddings = generate_batch_embeddings(batch_texts)

            # Создаем записи для Qdrant
            points = []
            for idx, (ktru_entry, embedding) in enumerate(zip(batch, batch_embeddings)):
                points.append(rest.PointStruct(
                    id=batch_ids[idx],
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

            processed_count += len(batch)
            logger.info(f"Обработано {processed_count}/{total_records} записей КТРУ")

        elapsed_time = time.time() - start_time
        logger.info(f"Обработка КТРУ завершена. Всего обработано {processed_count} записей.")
        logger.info(f"Время обработки: {elapsed_time:.2f} секунд")

    except Exception as e:
        logger.error(f"Ошибка при обработке данных КТРУ: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обработка данных КТРУ из JSON-файла')
    parser.add_argument('--json_file', type=str, required=True, help='Путь к JSON-файлу с данными КТРУ')
    args = parser.parse_args()

    process_ktru_json(args.json_file)